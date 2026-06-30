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


@dataclass
class EvidenceItem:
    name: str
    available: bool
    value: Optional[float]
    detail: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class AuthenticityAssessment:
    manipulation_probability: float
    confidence: float
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


@dataclass
class FrameAssessment:
    frame_idx: int
    spectral_score: Optional[float]
    landmark_count: int
    lighting_quality: float
    motion_blur: float


class CaptureQuality:
    DARK_THRESHOLD = 0.20
    LOW_LIGHT_THRESHOLD = 0.40
    MOTION_BLUR_VAR_THRESHOLD = 80.0

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
            return {
                "condition": "HIGH_MOTION_BLUR",
                "score": 0.0,
                "rppg_feasible": False,
                "laplacian_variance": avg_var,
            }
        return {
            "condition": "STABLE",
            "score": 1.0,
            "rppg_feasible": True,
            "laplacian_variance": avg_var,
        }


class POSRppg:
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
        mean_rgb = np.mean(rgb, axis=0)
        c = rgb / (mean_rgb + 1e-8)

        S1 = c[:, 1] - c[:, 2]
        S2 = c[:, 1] + c[:, 2] - 2.0 * c[:, 0]
        alpha = np.std(S1) / (np.std(S2) + 1e-8)
        pulse = S1 + alpha * S2
        return pulse - np.mean(pulse)

    def estimate(self) -> Dict[str, Any]:
        if len(self.rgb_buffer) < self.window_len:
            return {
                "available": False,
                "reason": "insufficient_frames",
                "frames_collected": len(self.rgb_buffer),
                "frames_needed": self.window_len,
            }

        rgb = np.array(self.rgb_buffer)
        pulse = self._pos_core(rgb)

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
        confidence = float(np.clip((snr_db - 3.0) / 12.0, 0.0, 1.0)) if plausible else 0.0

        return {
            "available": True,
            "heart_rate_bpm": hr_bpm,
            "snr_db": snr_db,
            "physiologically_plausible": plausible,
            "confidence": confidence,
        }


class SpectralArtifactDetector:
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

        low_band = radial[1:8]
        high_band = radial[-12:]
        falloff_ratio = float(np.mean(high_band) / (np.mean(low_band) + 1e-8))

        anomaly_score = float(np.clip((falloff_ratio - 0.15) / 0.45, 0.0, 1.0))

        return {
            "available": True,
            "high_frequency_falloff_ratio": falloff_ratio,
            "anomaly_score": anomaly_score,
            "note": "Weak signal: compression/upscaling of genuine footage can also elevate this ratio; treat as supportive, not conclusive.",
        }


class BlinkRateMonitor:
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
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
        left = landmarks_xy[self.LEFT_EYE_IDX]
        right = landmarks_xy[self.RIGHT_EYE_IDX]
        ear = (self._ear(left) + self._ear(right)) / 2.0
        self.ear_history.append(ear)

    def estimate_blink_rate(self) -> Dict[str, Any]:
        n = len(self.ear_history)
        if n < self.fps * 5:
            return {"available": False, "reason": "insufficient_duration"}

        ear = np.array(self.ear_history)
        below = ear < self.EAR_BLINK_THRESHOLD
        blinks = int(np.sum((below[1:]) & (~below[:-1])))
        duration_s = n / self.fps
        blink_rate_per_min = blinks / duration_s * 60.0

        normal_low, normal_high = 12.0, 20.0
        within_normal = normal_low <= blink_rate_per_min <= normal_high

        return {
            "available": True,
            "blink_count": blinks,
            "duration_seconds": duration_s,
            "blink_rate_per_min": blink_rate_per_min,
            "within_typical_range": within_normal,
            "note": "Weak, dataset-era-dependent signal; absence of anomaly is not evidence of authenticity.",
        }


class LandmarkTemporalConsistency:
    STABLE_LANDMARK_IDX = [1, 4, 6, 10, 152, 168, 197, 200]

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

        arr = np.array(self.history)
        frame_diffs = np.linalg.norm(np.diff(arr, axis=0), axis=2)
        jitter = float(np.mean(frame_diffs))
        jitter_std = float(np.std(frame_diffs))

        face_scale = float(np.mean(np.std(arr, axis=0)))
        normalized_jitter = jitter / (face_scale + 1e-6)

        return {
            "available": True,
            "normalized_jitter": normalized_jitter,
            "jitter_std": jitter_std,
            "note": "Self-referential temporal statistic, not a comparison against an external true geometry reference. Supportive evidence only.",
        }


class TemporalArtifactDetector:
    def analyze(self, face_crops: List[np.ndarray], fps: float = 30.0) -> Dict[str, Any]:
        if len(face_crops) < 30:
            return {"available": False, "reason": "insufficient_frames"}

        crops_array = np.array([c.astype(np.float32) for c in face_crops])
        diffs = np.mean(np.abs(np.diff(crops_array, axis=0)), axis=(1, 2, 3))

        fft_diffs = np.abs(np.fft.rfft(diffs - np.mean(diffs)))
        freqs = np.fft.rfftfreq(len(diffs), d=1.0 / fps)

        gop_freq_mask = (freqs > 2.0) & (freqs < 15.0)
        if np.any(gop_freq_mask):
            gop_energy = np.mean(fft_diffs[gop_freq_mask])
            low_freq_energy = np.mean(fft_diffs[freqs < 2.0])
            flicker_score = float(np.clip(gop_energy / (low_freq_energy + 1e-8) - 0.5, 0, 1))
        else:
            flicker_score = 0.0

        return {
            "available": True,
            "flicker_score": flicker_score,
            "mean_temporal_variance": float(np.mean(diffs)),
            "note": "Weak signal; video compression and re-encoding can also cause temporal artifacts. Interpret alongside other evidence.",
        }


class UncertaintyTracker:
    def __init__(self):
        self.frame_assessments: List[FrameAssessment] = []

    def add_frame(self, assessment: FrameAssessment):
        self.frame_assessments.append(assessment)

    def get_evidence_quality_report(self) -> Dict[str, Any]:
        if not self.frame_assessments:
            return {"available": False, "total_frames": 0}

        n_frames = len(self.frame_assessments)
        good_lighting = (
            sum(1 for a in self.frame_assessments if a.lighting_quality > 0.6) / n_frames
        )
        low_motion = (
            sum(1 for a in self.frame_assessments if a.motion_blur < 100) / n_frames
        )

        return {
            "available": True,
            "total_frames": n_frames,
            "fraction_good_lighting": good_lighting,
            "fraction_low_motion": low_motion,
            "overall_quality": (good_lighting + low_motion) / 2,
            "recommendation": (
                "Evidence quality adequate"
                if good_lighting > 0.5 and low_motion > 0.5
                else "Improve lighting and reduce motion for better evidence"
            ),
        }


class CalibrationSuite:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.calibration_curve = None

    def calibrate(self, videos: List[Tuple[str, bool]]) -> Dict[str, Any]:
        scores = []
        labels = []

        for path, is_fake in videos:
            assessment = self._process_video_file(path)
            if assessment:
                scores.append(assessment.manipulation_probability)
                labels.append(1 if is_fake else 0)

        if len(scores) < 50:
            return {
                "error": "insufficient_calibration_data",
                "minimum_required": 50,
                "provided": len(scores),
            }

        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.metrics import roc_auc_score, brier_score_loss

            self.calibration_curve = IsotonicRegression(out_of_bounds="clip")
            self.calibration_curve.fit(scores, labels)
            calibrated_scores = self.calibration_curve.predict(scores)

            return {
                "auc_roc": roc_auc_score(labels, calibrated_scores),
                "brier_score": brier_score_loss(labels, calibrated_scores),
                "n_samples": len(scores),
                "calibration_curve_points": list(
                    zip(
                        self.calibration_curve.X_thresholds_,
                        self.calibration_curve.y_thresholds_,
                    )
                ),
            }
        except ImportError:
            logger.warning("scikit-learn not available; skipping calibration")
            return {"error": "sklearn_not_installed"}

    def get_calibrated_probability(self, raw_probability: float) -> float:
        if self.calibration_curve is None:
            logger.warning("Using uncalibrated probability; run calibrate() first")
            return raw_probability
        return float(self.calibration_curve.predict([raw_probability])[0])

    def _process_video_file(self, path: str) -> Optional[AuthenticityAssessment]:
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            temp_pipeline = FaceVideoAuthenticityPipeline(fps=fps)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 3 == 0:
                    face_crop = cv2.resize(frame, (96, 96))
                    landmarks = np.random.randn(478, 2) * 0.02 + 0.5
                    temp_pipeline.process_frame(face_crop, landmarks)
                frame_idx += 1

            cap.release()
            return temp_pipeline.finalize()
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return None


class AuthenticityEstimator:
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
        temporal: Optional[Dict[str, Any]] = None,
    ) -> AuthenticityAssessment:
        evidence: List[EvidenceItem] = []
        weighted_sum = 0.0
        weight_total = 0.0

        if spectral.get("available"):
            score = spectral["anomaly_score"]
            evidence.append(
                EvidenceItem(
                    "spectral_artifact",
                    True,
                    score,
                    spectral,
                    "Higher = more frequency-domain evidence of generative upsampling artifacts.",
                )
            )
            weighted_sum += score * self.WEIGHTS["spectral_artifact"]
            weight_total += self.WEIGHTS["spectral_artifact"]
        else:
            evidence.append(
                EvidenceItem(
                    "spectral_artifact",
                    False,
                    None,
                    spectral,
                    "Not assessed: " + spectral.get("reason", "unknown"),
                )
            )

        if jitter.get("available"):
            nj = jitter["normalized_jitter"]
            typical_low, typical_high = 0.15, 1.2
            if nj < typical_low:
                score = float(np.clip((typical_low - nj) / typical_low, 0, 1)) * 0.6
            elif nj > typical_high:
                score = float(np.clip((nj - typical_high) / typical_high, 0, 1)) * 0.6
            else:
                score = 0.0
            evidence.append(
                EvidenceItem(
                    "landmark_jitter",
                    True,
                    score,
                    jitter,
                    "Distance from typical natural-motion jitter band.",
                )
            )
            weighted_sum += score * self.WEIGHTS["landmark_jitter"]
            weight_total += self.WEIGHTS["landmark_jitter"]
        else:
            evidence.append(
                EvidenceItem(
                    "landmark_jitter",
                    False,
                    None,
                    jitter,
                    "Not assessed: " + jitter.get("reason", "unknown"),
                )
            )

        if blink.get("available"):
            score = 0.0 if blink["within_typical_range"] else 0.35
            evidence.append(
                EvidenceItem(
                    "blink_rate", True, score, blink, blink.get("note", "")
                )
            )
            weighted_sum += score * self.WEIGHTS["blink_rate"]
            weight_total += self.WEIGHTS["blink_rate"]
        else:
            evidence.append(
                EvidenceItem(
                    "blink_rate",
                    False,
                    None,
                    blink,
                    "Not assessed: " + blink.get("reason", "unknown"),
                )
            )

        if rppg.get("available"):
            score = 0.0 if rppg.get("physiologically_plausible") else 0.3
            evidence.append(
                EvidenceItem(
                    "rppg_plausibility",
                    True,
                    score,
                    rppg,
                    "Pulse-signal extractability check; necessary-not-sufficient evidence of a real, well-lit face.",
                )
            )
            weighted_sum += score * self.WEIGHTS["rppg_plausibility"]
            weight_total += self.WEIGHTS["rppg_plausibility"]
        else:
            evidence.append(
                EvidenceItem(
                    "rppg_plausibility",
                    False,
                    None,
                    rppg,
                    "Not assessed: " + rppg.get("reason", "unknown"),
                )
            )

        if temporal and temporal.get("available"):
            flicker_score = temporal["flicker_score"]
            temporal_weight = 0.10
            evidence.append(
                EvidenceItem(
                    "temporal_artifact",
                    True,
                    flicker_score,
                    temporal,
                    temporal.get("note", ""),
                )
            )
            weighted_sum += flicker_score * temporal_weight
            weight_total += temporal_weight
        elif temporal:
            evidence.append(
                EvidenceItem(
                    "temporal_artifact",
                    False,
                    None,
                    temporal,
                    "Not assessed: " + temporal.get("reason", "unknown"),
                )
            )

        base_weight_sum = sum(self.WEIGHTS.values())

        if weight_total == 0:
            probability = 0.5
            confidence = 0.0
        else:
            probability = float(np.clip(weighted_sum / weight_total, 0.0, 1.0))
            confidence = float(np.clip(weight_total / base_weight_sum, 0.0, 1.0))

        review = confidence < 0.5 or 0.35 <= probability <= 0.65
        if confidence < 0.5:
            reason = "Insufficient usable evidence (poor lighting/motion/short clip)."
        elif 0.35 <= probability <= 0.65:
            reason = "Evidence is ambiguous; signals do not clearly agree."
        else:
            reason = (
                "Evidence sufficiently one-sided; routine review still advised for consequential decisions."
            )

        return AuthenticityAssessment(
            manipulation_probability=probability,
            confidence=confidence,
            evidence=evidence,
            content_sha256="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            human_review_recommended=review,
            review_reason=reason,
        )


class AssessmentDocumentor:
    @staticmethod
    def generate_report(assessment: AuthenticityAssessment) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("FACE VIDEO AUTHENTICITY ASSESSMENT")
        lines.append("=" * 60)
        lines.append(f"Generated: {assessment.timestamp}")
        lines.append(f"Confidence in estimate: {assessment.confidence:.1%}")
        lines.append("")

        lines.append("EVIDENCE SUMMARY:")
        lines.append("-" * 40)

        for evidence in assessment.evidence:
            status = "Available" if evidence.available else "Not assessed"
            lines.append(f"{evidence.name}: {status}")
            if evidence.value is not None:
                if evidence.value > 0.5:
                    interpretation = "Suggests manipulation"
                elif evidence.value < 0.3:
                    interpretation = "Suggests authenticity"
                else:
                    interpretation = "Ambiguous"
                lines.append(f"  Score: {evidence.value:.3f} ({interpretation})")
            lines.append(f"  Note: {evidence.note}")
            lines.append("")

        lines.append("OVERALL ASSESSMENT:")
        lines.append(f"  Probability of manipulation: {assessment.manipulation_probability:.1%}")
        lines.append(f"  Review recommended: {'Yes' if assessment.human_review_recommended else 'No'}")
        lines.append(f"  Reason: {assessment.review_reason}")
        lines.append("")
        lines.append("LIMITATIONS:")
        lines.append("  - This is a statistical estimate, not a forensic determination")
        lines.append("  - Individual signals are weak-to-moderate evidence only")
        lines.append("  - Calibration requires validation against labeled benchmarks")
        lines.append("  - Should not be used as sole grounds for consequential decisions")
        lines.append("=" * 60)

        return "\n".join(lines)


class FaceVideoAuthenticityPipeline:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.rppg = POSRppg(fps=fps)
        self.blink = BlinkRateMonitor(fps=fps)
        self.jitter = LandmarkTemporalConsistency()
        self.spectral = SpectralArtifactDetector()
        self.temporal = TemporalArtifactDetector()
        self.uncertainty = UncertaintyTracker()
        self.estimator = AuthenticityEstimator()
        self._frame_hashes: List[str] = []
        self._face_crops: List[np.ndarray] = []
        self._frame_count = 0

    def process_frame(self, face_crop_bgr: np.ndarray, landmarks_xy: Optional[np.ndarray] = None):
        self._frame_hashes.append(hashlib.sha256(face_crop_bgr.tobytes()).hexdigest())
        self._face_crops.append(face_crop_bgr.copy())
        self.rppg.add_frame_roi(face_crop_bgr)

        if landmarks_xy is not None:
            self.blink.update(landmarks_xy)
            self.jitter.update(landmarks_xy)

        lighting = CaptureQuality.assess_lighting(face_crop_bgr)
        motion_frames = self._face_crops[-5:] if len(self._face_crops) >= 3 else []
        motion = (
            CaptureQuality.assess_motion(motion_frames)
            if len(motion_frames) >= 3
            else {"laplacian_variance": 0.0}
        )

        self.uncertainty.add_frame(
            FrameAssessment(
                frame_idx=self._frame_count,
                spectral_score=None,
                landmark_count=landmarks_xy.shape[0] if landmarks_xy is not None else 0,
                lighting_quality=lighting.get("score", 0.0),
                motion_blur=motion.get("laplacian_variance", 0.0),
            )
        )

        self._frame_count += 1

    def finalize(self) -> AuthenticityAssessment:
        last_crop = self._face_crops[-1] if self._face_crops else None
        spectral_result = self.spectral.analyze(last_crop)
        jitter_result = self.jitter.assess()
        blink_result = self.blink.estimate_blink_rate()
        rppg_result = self.rppg.estimate()

        temporal_result = None
        if len(self._face_crops) >= 30:
            sampled_crops = self._face_crops[::3][:100]
            temporal_result = self.temporal.analyze(sampled_crops, self.fps)

        quality_report = self.uncertainty.get_evidence_quality_report()

        assessment = self.estimator.estimate(
            spectral_result, jitter_result, blink_result, rppg_result, temporal_result
        )
        content_hash = hashlib.sha256("".join(self._frame_hashes).encode()).hexdigest()
        assessment.content_sha256 = content_hash

        assessment.evidence.append(
            EvidenceItem(
                "capture_quality",
                quality_report.get("available", True),
                quality_report.get("overall_quality", 0.0),
                quality_report,
                "Overall quality of capture conditions affecting evidence reliability.",
            )
        )

        return assessment


def test_edge_cases():
    rng = np.random.default_rng(42)

    pipeline_good = FaceVideoAuthenticityPipeline(fps=30)
    for i in range(300):
        base_face = np.ones((96, 96, 3), dtype=np.uint8) * 128
        noise = rng.normal(0, 5, base_face.shape).astype(np.int16)
        crop = np.clip(base_face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ppg_variation = np.sin(2 * np.pi * 1.2 * i / 30) * 2
        crop = np.clip(crop.astype(np.float32) + ppg_variation, 0, 255).astype(np.uint8)
        landmarks = rng.normal(0.5, 0.005, (478, 2))
        pipeline_good.process_frame(crop, landmarks)

    result_good = pipeline_good.finalize()
    assert result_good.confidence > 0.7, f"Good conditions should yield high confidence, got {result_good.confidence:.3f}"
    print(f"Good conditions test passed. Confidence: {result_good.confidence:.3f}")

    pipeline_dark = FaceVideoAuthenticityPipeline(fps=30)
    for i in range(300):
        dark_crop = (rng.normal(30, 10, (96, 96, 3)).clip(0, 255)).astype(np.uint8)
        pipeline_dark.process_frame(dark_crop, rng.normal(0.5, 0.02, (478, 2)))

    result_dark = pipeline_dark.finalize()
    assert result_dark.confidence < 0.6, f"Poor lighting should reduce confidence, got {result_dark.confidence:.3f}"
    print(f"Poor lighting test passed. Confidence: {result_dark.confidence:.3f}")

    pipeline_empty = FaceVideoAuthenticityPipeline()
    result_empty = pipeline_empty.finalize()
    assert result_empty.confidence == 0.0, f"No data should yield zero confidence, got {result_empty.confidence:.3f}"
    assert result_empty.human_review_recommended, "Should recommend review with no data"
    print(f"Empty pipeline test passed. Confidence: {result_empty.confidence:.3f}")

    print("\nAll edge case tests passed!")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pipeline = FaceVideoAuthenticityPipeline(fps=30.0)

    for i in range(150):
        crop = (rng.normal(128, 20, (96, 96, 3))).clip(0, 255).astype(np.uint8)
        landmarks = rng.normal(0.5, 0.02, (478, 2))
        pipeline.process_frame(crop, landmarks)

    result = pipeline.finalize()
    print(json.dumps(result.to_json(), indent=2))
    print("\n" + AssessmentDocumentor.generate_report(result))
    print("\nRunning edge case tests...\n")
    test_edge_cases()