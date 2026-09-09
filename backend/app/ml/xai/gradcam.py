"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainability
...
"""
import hashlib
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None       # type: ignore
    F = None           # type: ignore
    transforms = None  # type: ignore


class GradCAM:
    """
    Grad-CAM implementation for visual explanations.

    Highlights regions in the fundus image that influenced the model's
    prediction via gradient-weighted class activation mapping.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate Class Activation Map for the given target class."""
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam     = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def generate_heatmap_overlay(
        self,
        original_image_path: str,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create heatmap overlay on original image."""
        original = cv2.imread(original_image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        h, w     = original.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay     = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
        return overlay


class ExplainabilityEngine:
    """
    High-level explainability engine for DR predictions.
    Handles both real model explanations (Grad-CAM) and demo mode.
    """

    def __init__(self, model=None, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.model     = model

        if not demo_mode and model is not None and _TORCH_AVAILABLE:
            target_layer  = self._get_target_layer(model)
            self.gradcam  = GradCAM(model, target_layer)
        else:
            self.gradcam = None

        if _TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = None

    def _get_target_layer(self, model):
        """Return the last convolutional layer (EfficientNet-B0 features[-1])."""
        return model.features[-1]

    def generate_explanation(
        self,
        image_path:    str,
        predicted_class: int,
        severity_label:  str,
        save_path:       Optional[str] = None,
    ) -> Dict:
        if self.demo_mode:
            return self._generate_demo_explanation(
                image_path, predicted_class, severity_label, save_path
            )
        return self._generate_real_explanation(
            image_path, predicted_class, severity_label, save_path
        )

    # ──────────────────────────────────────────────────────────────────
    # Real Grad-CAM
    # ──────────────────────────────────────────────────────────────────

    def _generate_real_explanation(
        self,
        image_path:    str,
        predicted_class: int,
        severity_label:  str,
        save_path:       Optional[str] = None,
    ) -> Dict:
        image        = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        cam          = self.gradcam.generate_cam(image_tensor, predicted_class)
        overlay      = self.gradcam.generate_heatmap_overlay(image_path, cam, alpha=0.5)

        if save_path:
            Image.fromarray(overlay).save(save_path)

        attention_regions = self._identify_attention_regions(cam)
        summary           = self._generate_explanation_summary(
            severity_label, attention_regions, predicted_class
        )

        return {
            "has_explanation":  True,
            "explanation_path": save_path,
            "attention_regions": attention_regions,
            "summary":           summary,
        }

    # ──────────────────────────────────────────────────────────────────
    # Demo explanation — deterministic, seeded from image content
    # ──────────────────────────────────────────────────────────────────

    def _generate_demo_explanation(
        self,
        image_path:    str,
        predicted_class: int,
        severity_label:  str,
        save_path:       Optional[str] = None,
    ) -> Dict:
        """
        Generate synthetic visual explanation for demo mode.

        Reproducibility: the random seed is derived from a hash of the
        image file bytes, so the SAME image always produces the SAME
        heatmap regardless of how many times explain() is called.

        IMPORTANT: This is a synthetic attention map for demonstration.
        It does NOT reflect actual model gradients in demo mode.
        """
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        h, w     = original.shape[:2]

        # ── Deterministic seed from image content ─────────────────
        seed = self._image_seed(image_path)
        cam  = self._create_synthetic_attention(h, w, predicted_class, seed)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

        if save_path:
            Image.fromarray(overlay).save(save_path)

        attention_regions = self._generate_demo_attention_regions(predicted_class)
        summary           = self._generate_explanation_summary(
            severity_label, attention_regions, predicted_class
        )

        return {
            "has_explanation":  True,
            "explanation_path": save_path,
            "attention_regions": attention_regions,
            "summary":           summary,
        }

    @staticmethod
    def _image_seed(image_path: str) -> int:
        """
        Compute a deterministic integer seed from the image file bytes.
        Returns a stable seed so the same image always produces the
        same synthetic attention map.
        """
        try:
            with open(image_path, "rb") as f:
                # Read first 4 KB — fast, stable across reads
                digest = hashlib.md5(f.read(4096)).hexdigest()
            return int(digest[:8], 16) % (2 ** 31)
        except Exception:
            return 42  # safe fallback

    def _create_synthetic_attention(
        self, h: int, w: int, severity: int, seed: int
    ) -> np.ndarray:
        """
        Create a synthetic attention map using seeded random Gaussian blobs.
        More severe cases have more distributed attention.
        """
        rng = np.random.RandomState(seed)
        cam = np.zeros((h, w), dtype=np.float32)
        num_regions = min(severity + 1, 3)

        for _ in range(num_regions):
            cx     = rng.randint(w // 4, 3 * w // 4)
            cy     = rng.randint(h // 4, 3 * h // 4)
            radius = min(h, w) // 6
            y_g, x_g = np.ogrid[:h, :w]
            mask    = ((x_g - cx) ** 2 + (y_g - cy) ** 2) <= radius ** 2
            cam[mask] += float(rng.uniform(0.5, 1.0))

        return np.clip(cam, 0, 1)

    # ──────────────────────────────────────────────────────────────────
    # Attention region identification (real model)
    # ──────────────────────────────────────────────────────────────────

    def _identify_attention_regions(self, cam: np.ndarray) -> list:
        threshold    = 0.7
        high_attention = cam > threshold
        regions = []
        if high_attention.sum() > (cam.size * 0.05):
            regions.append("Central retinal regions")
        if high_attention.sum() > (cam.size * 0.15):
            regions.append("Multiple vascular regions")
        if high_attention.sum() > (cam.size * 0.25):
            regions.append("Widespread retinal regions")
        return regions if regions else ["Localised retinal regions"]

    def _generate_demo_attention_regions(self, severity: int) -> list:
        return {
            0: ["General retinal regions"],
            1: ["Minor vascular regions"],
            2: ["Multiple retinal regions with vascular changes"],
            3: ["Widespread retinal regions", "Abnormal vascular patterns"],
            4: ["Extensive retinal regions", "Severe vascular abnormalities",
                "Proliferative changes"],
        }.get(severity, ["General retinal regions"])

    # ──────────────────────────────────────────────────────────────────
    # Explanation summary (careful clinical wording)
    # ──────────────────────────────────────────────────────────────────

    def _generate_explanation_summary(
        self,
        severity_label:   str,
        attention_regions: list,
        severity:          int,
    ) -> str:
        """
        Generate a human-readable explanation summary.

        Wording is carefully chosen to avoid overclaiming:
        - Does NOT say "the disease is located here"
        - Does NOT claim Grad-CAM performs lesion segmentation
        - Does describe what the visualisation represents
        """
        if severity == 0:
            base = (
                "The model analysed the retinal image and found patterns "
                "consistent with the absence of detectable DR-associated changes."
            )
        else:
            base = (
                f"The model identified patterns associated with {severity_label}."
            )

        regions_text = ", ".join(attention_regions).lower()

        return (
            f"{base} "
            f"Model attention was concentrated on {regions_text}. "
            "The highlighted regions show image areas that influenced this prediction. "
            "This visualisation represents model attention — not a clinical lesion map "
            "or confirmed diagnostic finding."
        )
