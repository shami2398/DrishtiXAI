"""
Diabetic Retinopathy Classification Model
Supports both real trained models and demo mode.
...
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict
from pathlib import Path

# Torch is only required in real (non-demo) mode.
# Import lazily so the server can start without GPU/torch in demo mode.
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore


class DRClassifier:
    """
    DR Classification Model using EfficientNet-B0.

    Demo mode uses computer-vision heuristics on the actual image pixels so
    the same image always produces the same, image-content-driven result.
    """

    CLASS_LABELS = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR",
    }

    # ── Carefully worded screening messages (not diagnostic statements) ──
    # Wording follows the three-state model:
    #   severity 0 → STATE A (no supported abnormality)
    #   severity 1–4 → STATE B (possible abnormality detected)
    #
    # "No supported abnormality detected" means the model did not find
    # patterns associated with DR in this image.  It does NOT mean
    # "the patient has no eye disease" or "the eye is healthy".
    SCREENING_MESSAGES = {
        0: "No supported abnormality detected in this image",
        1: "Possible mild diabetic retinopathy — early signs present",
        2: "Possible moderate diabetic retinopathy — significant changes detected",
        3: "Possible severe diabetic retinopathy — prompt clinical review recommended",
        4: "Possible proliferative diabetic retinopathy — urgent clinical review recommended",
    }

    # ── Legacy key preserved for any callers expecting diabetic_status ──
    DIABETIC_STATUS = SCREENING_MESSAGES

    def __init__(self, model_path: str = None, demo_mode: bool = True):
        self.demo_mode  = demo_mode
        self.model_path = model_path
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu") if _TORCH_AVAILABLE else None
        self.model      = None

        if not demo_mode and model_path and Path(model_path).exists() and _TORCH_AVAILABLE:
            self._load_model(model_path)
        else:
            self.demo_mode = True   # force demo if weights unavailable or torch missing

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

    # ──────────────────────────────────────────────────────────────────
    # Model loading (real mode)
    # ──────────────────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        self.model = models.efficientnet_b0(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 5)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # ──────────────────────────────────────────────────────────────────
    # Public predict
    # ──────────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> Dict:
        if self.demo_mode:
            return self._image_based_predict(image_path)
        return self._real_predict(image_path)

    # ──────────────────────────────────────────────────────────────────
    # Real model inference
    # ──────────────────────────────────────────────────────────────────

    def _real_predict(self, image_path: str) -> Dict:
        image        = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs       = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted     = torch.argmax(probabilities, dim=1).item()
            confidence    = probabilities[0][predicted].item()

        class_probs = {
            self.CLASS_LABELS[i]: float(probabilities[0][i].item())
            for i in range(5)
        }

        return self._build_result(predicted, confidence, class_probs)

    # ──────────────────────────────────────────────────────────────────
    # Image-content-based demo prediction
    # ──────────────────────────────────────────────────────────────────

    def _image_based_predict(self, image_path: str) -> Dict:
        """
        Analyses real image pixels to produce a clinically-plausible DR
        severity score.  Checks features typically associated with DR:

          • Red lesion density  — microaneurysms, haemorrhages (red channel)
          • Dark spot density   — hard exudates, drusen (dark regions)
          • Vascular complexity — neovascularisation proxy (edge density)
          • Overall brightness  — retinal illumination quality
          • Contrast            — image dynamic range

        These are combined into a score that maps to severity 0–4.

        IMPORTANT: This is a pixel-analysis heuristic for demonstration.
        It is NOT a trained DR classifier and NOT for clinical use.
        """
        img = cv2.imread(image_path)
        if img is None:
            # Cannot read → safe fallback: No DR, low confidence
            return self._build_result(0, 0.55, self._uniform_probs(0, 0.55))

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r_ch, g_ch, b_ch = img[:, :, 2], img[:, :, 1], img[:, :, 0]  # BGR→RGB

        # ── Feature 1: Red lesion score ──────────────────────────
        red_dominance   = r_ch.astype(np.float32) - g_ch.astype(np.float32)
        red_lesion_mask = (red_dominance > 30) & (r_ch > 60) & (r_ch < 180)
        red_lesion_ratio = red_lesion_mask.sum() / (h * w)

        # ── Feature 2: Dark spot density ─────────────────────────
        mean_bright  = float(np.mean(gray))
        dark_thresh  = max(20, mean_bright * 0.4)
        dark_mask    = gray < dark_thresh
        cy, cx       = h // 2, w // 2
        radius       = min(h, w) // 2 - 10
        y_grid, x_grid = np.ogrid[:h, :w]
        circle_mask    = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= radius ** 2
        dark_in_circle = (dark_mask & circle_mask).sum()
        circle_area    = circle_mask.sum()
        dark_spot_ratio = dark_in_circle / max(circle_area, 1)

        # ── Feature 3: Vascular / edge complexity ────────────────
        blur         = cv2.GaussianBlur(g_ch, (3, 3), 0)
        edges        = cv2.Canny(blur, 30, 80)
        edge_density = edges.sum() / (255.0 * h * w)

        # ── Feature 4: Contrast ───────────────────────────────────
        contrast = float(np.std(gray)) / 128.0

        # ── Composite severity score [0, 1] ──────────────────────
        score = (
            red_lesion_ratio * 5.0 +
            dark_spot_ratio  * 3.0 +
            edge_density     * 4.0 +
            max(0, 0.5 - contrast) * 2.0
        )

        # ── Map score → severity (0–4) ────────────────────────────
        if score < 0.08:
            severity = 0
        elif score < 0.18:
            severity = 1
        elif score < 0.32:
            severity = 2
        elif score < 0.50:
            severity = 3
        else:
            severity = 4

        # ── Confidence: higher when features are unambiguous ─────
        boundaries = [0.08, 0.18, 0.32, 0.50]
        distances  = [abs(score - b) for b in boundaries]
        min_dist   = min(distances)
        # Scale to [0.65, 0.95]
        confidence = 0.65 + min(min_dist / 0.10, 1.0) * 0.30

        class_probs = self._build_softmax_probs(severity, confidence)

        return self._build_result(severity, round(confidence, 3), class_probs)

    # ──────────────────────────────────────────────────────────────────
    # Result builder
    # ──────────────────────────────────────────────────────────────────

    def _build_result(self, severity: int, confidence: float,
                      class_probs: Dict[str, float]) -> Dict:
        return {
            "severity":              severity,
            "severity_label":        self.CLASS_LABELS[severity],
            # screening_message uses STATE A / STATE B safe wording
            "screening_message":     self.SCREENING_MESSAGES[severity],
            # diabetic_status kept for backwards compatibility
            "diabetic_status":       self.SCREENING_MESSAGES[severity],
            "is_diabetic":           severity > 0,
            "confidence":            float(confidence),
            "class_probabilities":   class_probs,
            "requires_human_review": confidence < 0.70,
        }

    def _build_softmax_probs(self, predicted: int,
                              confidence: float) -> Dict[str, float]:
        """
        Build a probability distribution centred on `predicted` with the
        remaining probability spread over adjacent classes.
        """
        probs     = [0.0] * 5
        remaining = 1.0 - confidence
        probs[predicted] = confidence

        weights = []
        for i in range(5):
            if i == predicted:
                continue
            dist = abs(i - predicted)
            weights.append((i, 1.0 / (dist ** 2 + 1)))

        total_w = sum(w for _, w in weights)
        for i, w in weights:
            probs[i] = remaining * (w / total_w)

        total = sum(probs)
        probs = [p / total for p in probs]

        return {self.CLASS_LABELS[i]: round(probs[i], 4) for i in range(5)}

    def _uniform_probs(self, predicted: int, confidence: float) -> Dict[str, float]:
        return self._build_softmax_probs(predicted, confidence)

    def get_model_info(self) -> Dict:
        return {
            "architecture":  "EfficientNet-B0",
            "num_classes":   5,
            "class_labels":  self.CLASS_LABELS,
            "demo_mode":     self.demo_mode,
            "device":        str(self.device),
        }
