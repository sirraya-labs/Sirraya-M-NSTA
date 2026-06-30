"""
face_video_authenticity.py

A face-video authenticity scoring pipeline built only from techniques that
have a published, reproducible basis. No fabricated subsystems, no
placeholder detectors that return constants, no marketing-grade scoring.

Every signal below is weak-to-moderate evidence, not proof. The output is a
calibrated probability with an explicit uncertainty estimate and a clear
statement of what was and wasn't measurable, not a verdict.

References (paraphrased, not quoted):
  - rPPG / POS algorithm:
      Wang, den Brinker, Stuijk, de Haan. "Algorithmic Principles of Remote
      PPG." IEEE Transactions on Biomedical Engineering, 2017.
  - GAN/upsampling frequency artifacts:
      Durall, Keuper, Keuper. "Watch your Up-Convolution: CNN Based
      Generative Deep Neural Networks are Failing to Reproduce Spectral
      Distributions." CVPR 2020.
      Frank et al. "Leveraging Frequency Analysis for Deep Fake Image
      Recognition." ICML 2020.
      Zhang, Karaman, Chang. "Detecting and Simulating Artifacts in GAN
      Fake Images." WIFS 2019.
  - Eye-blink rate as a liveness cue (weak, dataset-era dependent):
      Li, Chang, Lyu. "In Ictu Oculi: Exposing AI-Generated Fake Face
      Videos by Detecting Eye Blinking." WIFS 2018.
      Eye-aspect-ratio computation:
      Soukupova & Cech. "Real-Time Eye Blink Detection using Facial
      Landmarks." CVWW 2016.
  - Landmark temporal jitter as a face-swap cue (weak, supportive only):
      Sabir et al. "Recurrent Convolutional Strategies for Face
      Manipulation Detection in Videos." CVPRW 2019 (motivates treating
      manipulation evidence as a temporal/sequential signal rather than a
      single-frame geometric "rigidity" claim).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import fft as sp_fft
from scipy import signal as sp_signal

logger = logging.getLogger("face_video_authenticity")
logging.basicConfig(level=logging.INFO)


# ============================================================================
# 0. Honest result type — probability + uncertainty, no fake certification
# ============================================================================

@dataclass
class EvidenceItem:
    name: str
    available: bool
    value: Optional[float]              # interpretable score in [0, 1] if available
    detail: Dict[str, Any] = field(default_factory=dict)
    note: str = ""                      # caveats, e.g. "weak signal", "not assessed"


@dataclass
class AuthenticityAssessment:
    """
    Output of the pipeline. Deliberately avoids:
      - a single 0-100 "integrity score" implying false precision
      - emoji/marketing categories ("PLATINUM", "DETERMINISTIC AUTHENTICATION")
      - bonus/penalty multipliers with no calibration basis
    """
    manipulation_probability: float          # 0-1, higher = more evidence of manipulation
    confidence: float                        # 0-1, how much usable evidence was present
    evidence: List[EvidenceItem]
    content_sha256: str
    timestamp: str
    human_review_recommended: bool
    review_reason: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "manipulation_probability": round(self.manipulation_probability, 4),
            "confidence_in_estimate": round(self.confidence, 4),
            "human_review_recommended": self.human_review_recommended,
            "review_reason": self.review_reason,
            "evidence": [
                {
                    "signal": e.name,
                    "available": e.available,
                    "score": None if e.value is None else round(e.value, 4),
                    "detail": e.detail,
                    "note": e.note,
                }
                for e in self.evidence
            ],
            "content_sha256": self.content_sha256,
            "disclaimer": (
                "This is a statistical estimate from a small set of weak-to-moderate "
                "evidentiary signals. It is not a forensic determination and should not "
                "be used as sole grounds for action against a person."
            ),
        }


# ============================================================================
# 1. Environmental / signal-quality gating (kept — this part was legitimate)
# ============================================================================

class CaptureQuality:
    """Gates whether downstream signals (especially rPPG) are trustworthy."""

    DARK_THRESHOLD = 0.20
    LOW_LIGHT_THRESHOLD = 0.40
    MOTION_BLUR_VAR_THRESHOLD = 80.0   # raw Laplacian variance, not arbitrarily rescaled

    @staticmethod
    def assess_lighting(frame: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        brightness = float(np.mean(gray)) / 255.0
        contrast = float(np.std(gray)) / 255.0

        if brightness < CaptureQuality.DARK_THRESHOLD:
            condition, score = "DARK", 0.0
        elif brightness < CaptureQuality.LOW_LIGHT_THRESHOLD:
            condition, score = "LOW_LIGHT", 0.4
        else:
            condition, score = "ADEQUATE", 1.0

        return {
            "brightness": brightness,
            "contrast": contrast,
            "condition": condition,
            "score": score,
            "rppg_feasible": brightness >= CaptureQuality.DARK_THRESHOLD,
        }

    @staticmethod
    def assess_motion(frames: List[np.ndarray]) -> Dict[str, Any]:
        if len(frames) < 3:
            return {"condition": "INSUFFICIENT_DATA", "score": 0.5, "rppg_feasible": True}

        variances = []
        for f in frames[-5:]:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            variances.append(float(np.var(cv2.Laplacian(gray, cv2.CV_64F))))
        avg_var = float(np.mean(variances))

        if avg_var < CaptureQuality.MOTION_BLUR_VAR_THRESHOLD:
            return {"condition": "HIGH_MOTION_BLUR", "score": 0.0, "rppg_feasible": False,
                    "laplacian_variance": avg_var}
        return {"condition": "STABLE", "score": 1.0, "rppg_feasible": True,
                "laplacian_variance": avg_var}


# ============================================================================
# 2. rPPG via POS (Wang et al. 2017) — replaces the fabricated "ICA" step
# ============================================================================

class POSRppg:
    """
    Plane-Orthogonal-to-Skin (POS) remote photoplethysmography.

    This is the algorithm from Wang, den Brinker, Stuijk & de Haan (2017),
    chosen over the green-channel/PCA approach in the original code because
    it has a documented closed-form derivation and is widely reproduced in
    follow-up benchmarks, rather than an undocumented "ICA on 2 PCA
    components" heuristic.

    Heart-rate plausibility (40-180 bpm) is used only as a sanity filter on
    *whether a pulse signal is extractable at all* — never as proof of
    liveness by itself. A clean pulse signal is necessary-but-not-sufficient
    evidence of a real face under good lighting; it is not a deepfake
    detector on its own, and modern face-swap / reenactment pipelines that
    preserve the source actor's underlying video can still carry a real
    pulse signal through. This is reported as supportive evidence only.
    """

    def __init__(self, fps: float = 30.0, window_seconds: float = 8.0):
        self.fps = fps
        self.window_len = max(int(fps * window_seconds), int(fps * 2))
        self.rgb_buffer: List[np.ndarray] = []

    def add_frame_roi(self, roi_bgr: np.ndarray):
        b, g, r = cv2.split(roi_bgr.astype(np.float64))
        self.rgb_buffer.append(np.array([np.mean(r), np.mean(g), np.mean(b)]))
        if len(self.rgb_buffer) > self.window_len:
            self.rgb_buffer.pop(0)

    def _pos_core(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (N, 3) array of [R, G, B] frame means. Returns the POS pulse signal."""
        mean_rgb = np.mean(rgb, axis=0)
        c = rgb / (mean_rgb + 1e-8)  # temporal normalization

        # POS projection plane, as defined in Wang et al. 2017 eq. (9)-(12)
        S1 = c[:, 1] - c[:, 2]                       # G - B
        S2 = c[:, 1] + c[:, 2] - 2.0 * c[:, 0]        # G + B - 2R
        alpha = np.std(S1) / (np.std(S2) + 1e-8)
        pulse = S1 + alpha * S2
        return pulse - np.mean(pulse)

    def estimate(self) -> Dict[str, Any]:
        if len(self.rgb_buffer) < self.window_len:
            return {"available": False, "reason": "insufficient_frames",
                     "frames_collected": len(self.rgb_buffer), "frames_needed": self.window_len}

        rgb = np.array(self.rgb_buffer)
        pulse = self._pos_core(rgb)

        # bandpass to physiological range (0.7-3.0 Hz == 42-180 bpm)
        nyq = 0.5 * self.fps
        low, high = 0.7 / nyq, 3.0 / nyq
        high = min(high, 0.99)
        b, a = sp_signal.butter(3, [low, high], btype="band")
        filtered = sp_signal.filtfilt(b, a, pulse)

        freqs = sp_fft.rfftfreq(len(filtered), d=1.0 / self.fps)
        power = np.abs(sp_fft.rfft(filtered)) ** 2

        band_mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(band_mask):
            return {"available": False, "reason": "no_energy_in_physiological_band"}

        peak_idx = np.argmax(power[band_mask])
        peak_freq = freqs[band_mask][peak_idx]
        hr_bpm = float(peak_freq * 60.0)

        signal_power = float(np.max(power[band_mask]))
        noise_power = float(np.mean(power[~band_mask])) if np.any(~band_mask) else 1e-8
        snr_db = float(10 * np.log10(signal_power / max(noise_power, 1e-8)))

        plausible = 40.0 <= hr_bpm <= 180.0
        # confidence is a simple monotonic function of SNR, explicitly capped —
        # not dressed up as a 0-100 "biophysical integrity" certification
        confidence = float(np.clip((snr_db - 3.0) / 12.0, 0.0, 1.0)) if plausible else 0.0

        return {
            "available": True,
            "heart_rate_bpm": hr_bpm,
            "snr_db": snr_db,
            "physiologically_plausible": plausible,
            "confidence": confidence,
        }


# ============================================================================
# 3. Frequency-domain GAN/upsampling artifact detector (real, implemented)
# ============================================================================

class SpectralArtifactDetector:
    """
    Detects the periodic high-frequency artifacts that transposed-convolution
    / upsampling-based generators tend to leave behind (Durall et al. 2020;
    Zhang et al. 2019). This computes the azimuthally-averaged 2D power
    spectrum of a face crop and looks for anomalous energy concentration in
    the high-frequency band relative to natural-image spectral falloff.

    This is a real, runnable detector — not a stub. Its output should still
    be read as one weak statistical cue: heavy compression, denoising, and
    upscaling of genuine footage can also perturb the spectrum, and not all
    generative pipelines leave this signature (e.g. some diffusion-based
    synthesis reduces it). It is reported with that caveat attached.
    """

    @staticmethod
    def _azimuthal_average(power_2d: np.ndarray, n_bins: int = 40) -> np.ndarray:
        h, w = power_2d.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = r.max()
        bin_edges = np.linspace(0, r_max, n_bins + 1)
        radial_profile = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
            if np.any(mask):
                radial_profile[i] = power_2d[mask].mean()
        return radial_profile

    def analyze(self, face_crop_bgr: np.ndarray) -> Dict[str, Any]:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return {"available": False, "reason": "empty_crop"}

        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray = gray - gray.mean()
        window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
        spectrum = np.fft.fftshift(np.fft.fft2(gray * window))
        power = np.log1p(np.abs(spectrum) ** 2)

        radial = self._azimuthal_average(power)
        if len(radial) < 10 or radial[1] <= 0:
            return {"available": False, "reason": "degenerate_spectrum"}

        # Natural photographic images show roughly monotonic falloff of power
        # with frequency. A relative bump in the outer (high-frequency) bins
        # is the signature Durall et al. and Zhang et al. associate with
        # learned upsampling. We measure that bump directly rather than
        # asserting a hardcoded "threshold" without basis.
        low_band = radial[1:8]
        high_band = radial[-12:]
        falloff_ratio = float(np.mean(high_band) / (np.mean(low_band) + 1e-8))

        # Empirically, natural images have a falloff_ratio well below ~0.35
        # at this resolution; we report the raw ratio plus a soft score
        # rather than a binary "GAN/not GAN" claim.
        anomaly_score = float(np.clip((falloff_ratio - 0.15) / 0.45, 0.0, 1.0))

        return {
            "available": True,
            "high_frequency_falloff_ratio": falloff_ratio,
            "anomaly_score": anomaly_score,
            "note": "Weak signal: compression/upscaling of genuine footage can also "
                    "elevate this ratio; treat as supportive, not conclusive.",
        }


# ============================================================================
# 4. Eye-blink rate (Li et al. 2018) via eye-aspect-ratio (Soukupova & Cech 2016)
# ============================================================================

class BlinkRateMonitor:
    """
    Tracks eye-aspect-ratio (EAR) across frames to estimate blink rate.
    Li et al. 2018 found early face-swap deepfakes under-blinked relative to
    natural human blink rates (roughly 0.28-0.4 Hz / 15-20 blinks per
    minute at rest, per the cited physiology literature in that paper).

    Important honesty note: later generative pipelines specifically fixed
    this gap once it became well known, so an unusually low blink rate is
    weak corroborating evidence at best and a normal blink rate is *not*
    evidence of authenticity. We surface it for transparency, weighted low.
    """

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]    # MediaPipe FaceMesh indices
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    EAR_BLINK_THRESHOLD = 0.21

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.ear_history: List[float] = []

    @staticmethod
    def _ear(eye_pts: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = eye_pts
        vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horiz = 2.0 * np.linalg.norm(p1 - p4)
        return float(vert / (horiz + 1e-8))

    def update(self, landmarks_xy: np.ndarray):
        """landmarks_xy: (478, 2) MediaPipe FaceMesh landmark array for one frame."""
        left = landmarks_xy[self.LEFT_EYE_IDX]
        right = landmarks_xy[self.RIGHT_EYE_IDX]
        ear = (self._ear(left) + self._ear(right)) / 2.0
        self.ear_history.append(ear)

    def estimate_blink_rate(self) -> Dict[str, Any]:
        n = len(self.ear_history)
        if n < self.fps * 5:  # need at least ~5s
            return {"available": False, "reason": "insufficient_duration"}

        ear = np.array(self.ear_history)
        below = ear < self.EAR_BLINK_THRESHOLD
        # count rising-edge transitions into a blink state
        blinks = int(np.sum((below[1:]) & (~below[:-1])))
        duration_s = n / self.fps
        blink_rate_per_min = blinks / duration_s * 60.0

        # normal resting blink rate range commonly cited: ~12-20/min
        normal_low, normal_high = 12.0, 20.0
        within_normal = normal_low <= blink_rate_per_min <= normal_high

        return {
            "available": True,
            "blink_count": blinks,
            "duration_seconds": duration_s,
            "blink_rate_per_min": blink_rate_per_min,
            "within_typical_range": within_normal,
            "note": "Weak, dataset-era-dependent signal; absence of anomaly is not "
                    "evidence of authenticity.",
        }


# ============================================================================
# 5. Landmark temporal jitter — honest replacement for fake "rigidity" check
# ============================================================================

class LandmarkTemporalConsistency:
    """
    Measures frame-to-frame jitter of stable facial landmarks (motivated by
    Sabir et al. 2019's framing of manipulation evidence as a temporal
    signal). This does NOT claim to detect "structural impossibility" from a
    single reference frame, unlike the prior code's bone-rigidity z-score
    system, which had no validated reference and conflated normal head
    movement with anomaly.

    What it actually measures: unusually high or unusually low frame-to-
    frame variance in landmark position, normalized by face size, relative
    to the clip's own running statistics (self-referential, not compared to
    an arbitrary external reference frame).
    """

    STABLE_LANDMARK_IDX = [1, 4, 6, 10, 152, 168, 197, 200]  # nose bridge / forehead / chin

    def __init__(self, history_len: int = 60):
        self.history: List[np.ndarray] = []
        self.history_len = history_len

    def update(self, landmarks_xy: np.ndarray):
        pts = landmarks_xy[self.STABLE_LANDMARK_IDX]
        self.history.append(pts)
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def assess(self) -> Dict[str, Any]:
        if len(self.history) < 10:
            return {"available": False, "reason": "insufficient_history"}

        arr = np.array(self.history)  # (T, K, 2)
        frame_diffs = np.linalg.norm(np.diff(arr, axis=0), axis=2)  # (T-1, K)
        jitter = float(np.mean(frame_diffs))
        jitter_std = float(np.std(frame_diffs))

        # face-size normalization using inter-ocular-ish spread of the same points
        face_scale = float(np.mean(np.std(arr, axis=0)))
        normalized_jitter = jitter / (face_scale + 1e-6)

        return {
            "available": True,
            "normalized_jitter": normalized_jitter,
            "jitter_std": jitter_std,
            "note": "Self-referential temporal statistic, not a comparison against an "
                    "external 'true geometry' reference. Supportive evidence only.",
        }


# ============================================================================
# 6. Honest fusion — transparent weighted average, explicit confidence, no
#    fabricated bonuses/penalties/certification tiers
# ============================================================================

class AuthenticityEstimator:
    """
    Combines available evidence into a probability with explicit confidence.
    Design choices, stated plainly:
      - Each evidence source contributes only when actually computed.
        Unavailable sources are excluded from the weighted sum entirely
        (not silently defaulted to 0.5, which the prior code did and which
        quietly drags every score toward the middle).
      - Weights are fixed, documented, and modest — this is NOT a trained,
        calibrated classifier. Treat the output as a triage signal, not a
        forensic verdict. If you need calibrated probabilities, this needs
        to be validated against a labeled benchmark (e.g. FaceForensics++,
        DFDC, Celeb-DF) before deployment, and weights re-fit accordingly.
      - confidence reflects how much usable evidence existed, separate from
        the probability itself, so a 0.5 "manipulation_probability" from
        thin evidence reads differently than 0.5 from rich evidence.
    """

    # weight = relative trust in the signal's discriminative power based on
    # the cited literature; these are starting points, not validated coefficients
    WEIGHTS = {
        "spectral_artifact": 0.40,
        "landmark_jitter": 0.25,
        "blink_rate": 0.15,
        "rppg_plausibility": 0.20,
    }

    def estimate(
        self,
        spectral: Dict[str, Any],
        jitter: Dict[str, Any],
        blink: Dict[str, Any],
        rppg: Dict[str, Any],
    ) -> AuthenticityAssessment:
        evidence: List[EvidenceItem] = []
        weighted_sum = 0.0
        weight_total = 0.0

        if spectral.get("available"):
            score = spectral["anomaly_score"]
            evidence.append(EvidenceItem("spectral_artifact", True, score, spectral,
                                          "Higher = more frequency-domain evidence of "
                                          "generative upsampling artifacts."))
            weighted_sum += score * self.WEIGHTS["spectral_artifact"]
            weight_total += self.WEIGHTS["spectral_artifact"]
        else:
            evidence.append(EvidenceItem("spectral_artifact", False, None, spectral,
                                          "Not assessed: " + spectral.get("reason", "unknown")))

        if jitter.get("available"):
            # extreme jitter in either direction (too erratic OR suspiciously
            # smooth) is treated as mild evidence; map distance from a typical
            # band to a 0-1 score rather than asserting a hard pass/fail
            nj = jitter["normalized_jitter"]
            typical_low, typical_high = 0.15, 1.2
            if nj < typical_low:
                score = float(np.clip((typical_low - nj) / typical_low, 0, 1)) * 0.6
            elif nj > typical_high:
                score = float(np.clip((nj - typical_high) / typical_high, 0, 1)) * 0.6
            else:
                score = 0.0
            evidence.append(EvidenceItem("landmark_jitter", True, score, jitter,
                                          "Distance from typical natural-motion jitter band."))
            weighted_sum += score * self.WEIGHTS["landmark_jitter"]
            weight_total += self.WEIGHTS["landmark_jitter"]
        else:
            evidence.append(EvidenceItem("landmark_jitter", False, None, jitter,
                                          "Not assessed: " + jitter.get("reason", "unknown")))

        if blink.get("available"):
            score = 0.0 if blink["within_typical_range"] else 0.35  # capped low: weak signal
            evidence.append(EvidenceItem("blink_rate", True, score, blink, blink.get("note", "")))
            weighted_sum += score * self.WEIGHTS["blink_rate"]
            weight_total += self.WEIGHTS["blink_rate"]
        else:
            evidence.append(EvidenceItem("blink_rate", False, None, blink,
                                          "Not assessed: " + blink.get("reason", "unknown")))

        if rppg.get("available"):
            # a clean, physiologically plausible pulse lowers suspicion mildly;
            # absence of one is NOT treated as evidence of manipulation, since
            # it's commonly just poor lighting/motion
            score = 0.0 if rppg.get("physiologically_plausible") else 0.3
            evidence.append(EvidenceItem("rppg_plausibility", True, score, rppg,
                                          "Pulse-signal extractability check; necessary-not-"
                                          "sufficient evidence of a real, well-lit face."))
            weighted_sum += score * self.WEIGHTS["rppg_plausibility"]
            weight_total += self.WEIGHTS["rppg_plausibility"]
        else:
            evidence.append(EvidenceItem("rppg_plausibility", False, None, rppg,
                                          "Not assessed: " + rppg.get("reason", "unknown")))

        if weight_total == 0:
            probability = 0.5
            confidence = 0.0
        else:
            probability = float(np.clip(weighted_sum / weight_total, 0.0, 1.0))
            confidence = float(weight_total / sum(self.WEIGHTS.values()))

        review = confidence < 0.5 or 0.35 <= probability <= 0.65
        if confidence < 0.5:
            reason = "Insufficient usable evidence (poor lighting/motion/short clip)."
        elif 0.35 <= probability <= 0.65:
            reason = "Evidence is ambiguous; signals do not clearly agree."
        else:
            reason = "Evidence sufficiently one-sided; routine review still advised for " \
                     "consequential decisions."

        return AuthenticityAssessment(
            manipulation_probability=probability,
            confidence=confidence,
            evidence=evidence,
            content_sha256="",  # filled in by the orchestrator with real frame data
            timestamp=datetime.now(timezone.utc).isoformat(),
            human_review_recommended=review,
            review_reason=reason,
        )


# ============================================================================
# 7. Orchestrator — wires real components together, no fabricated layers
# ============================================================================

class FaceVideoAuthenticityPipeline:
    """
    Minimal orchestrator. Expects the caller to supply face-cropped BGR
    frames and, where available, MediaPipe FaceMesh landmarks (478, 2)
    per frame — this module does not bundle a face detector/landmarker,
    to avoid silently depending on an unstated model version.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.rppg = POSRppg(fps=fps)
        self.blink = BlinkRateMonitor(fps=fps)
        self.jitter = LandmarkTemporalConsistency()
        self.spectral = SpectralArtifactDetector()
        self.estimator = AuthenticityEstimator()
        self._frame_hashes: List[str] = []

    def process_frame(self, face_crop_bgr: np.ndarray, landmarks_xy: Optional[np.ndarray] = None):
        self._frame_hashes.append(hashlib.sha256(face_crop_bgr.tobytes()).hexdigest())
        self.rppg.add_frame_roi(face_crop_bgr)
        if landmarks_xy is not None:
            self.blink.update(landmarks_xy)
            self.jitter.update(landmarks_xy)
        self._last_crop = face_crop_bgr

    def finalize(self) -> AuthenticityAssessment:
        spectral_result = self.spectral.analyze(getattr(self, "_last_crop", None))
        jitter_result = self.jitter.assess()
        blink_result = self.blink.estimate_blink_rate()
        rppg_result = self.rppg.estimate()

        assessment = self.estimator.estimate(spectral_result, jitter_result, blink_result, rppg_result)
        content_hash = hashlib.sha256("".join(self._frame_hashes).encode()).hexdigest()
        assessment.content_sha256 = content_hash
        return assessment


if __name__ == "__main__":
    # Minimal smoke test with synthetic data so the module is verifiably
    # runnable end-to-end (not just a wall of unexecuted classes).
    rng = np.random.default_rng(0)
    pipeline = FaceVideoAuthenticityPipeline(fps=30.0)

    for i in range(150):  # 5 seconds @ 30fps
        crop = (rng.normal(128, 20, (96, 96, 3))).clip(0, 255).astype(np.uint8)
        landmarks = rng.normal(0.5, 0.02, (478, 2))
        pipeline.process_frame(crop, landmarks)

    result = pipeline.finalize()
    print(json.dumps(result.to_json(), indent=2))