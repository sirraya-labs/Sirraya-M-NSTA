# Face Video Authenticity Pipeline

A face-video analysis pipeline that combines a small set of published, reproducible signals into a manipulation-probability estimate with an explicit confidence score. It does not produce a verdict and is not a forensic tool.

## What it does

Given a sequence of face-cropped video frames (and optionally per-frame facial landmarks), the pipeline computes four independent signals and fuses them into a single probability with an associated confidence level. Each signal is computed only when the input data supports it; signals that cannot be computed are marked unavailable and excluded from the fusion rather than defaulted to a neutral value.

## Components

### Capture quality gating (`CaptureQuality`)
Assesses lighting and motion blur from raw frames. Used to determine whether downstream signal extraction (particularly pulse estimation) is likely to be reliable.

- `assess_lighting`: brightness and contrast from grayscale frame statistics.
- `assess_motion`: Laplacian variance across recent frames as a blur indicator.

### Remote photoplethysmography (`POSRppg`)
Extracts a pulse signal from facial skin color variation using the POS (Plane-Orthogonal-to-Skin) algorithm (Wang, den Brinker, Stuijk, de Haan, IEEE TBME 2017). Reports heart rate, signal-to-noise ratio, and whether the estimated rate falls in a physiologically plausible range (40-180 bpm).

This is necessary-but-not-sufficient evidence of a real, well-lit face. A clean pulse signal does not by itself confirm authenticity, and absence of one (commonly due to poor lighting or motion) is not treated as evidence of manipulation.

### Spectral artifact detection (`SpectralArtifactDetector`)
Computes the 2D power spectrum of a face crop, azimuthally averages it into radial frequency bins, and measures the ratio of high-frequency to low-frequency energy. Generative upsampling architectures tend to leave a measurable bump in this ratio (Durall, Keuper, Keuper, CVPR 2020; Zhang, Karaman, Chang, WIFS 2019). Returns a continuous anomaly score rather than a binary classification.

Compression and upscaling of genuine footage can also elevate this ratio. The signal is reported as supportive, not conclusive.

### Blink rate monitoring (`BlinkRateMonitor`)
Tracks eye aspect ratio (EAR) across frames (Soukupova & Cech, CVWW 2016) and counts blink events. Compares the resulting blink rate to a typical resting range (12-20 per minute), motivated by early findings that face-swap deepfakes under-blinked relative to natural human rates (Li, Chang, Lyu, WIFS 2018).

This signal has degraded in discriminative value over time as generative pipelines have specifically addressed it. A normal blink rate is not evidence of authenticity; an abnormal rate is weak corroborating evidence at most.

### Landmark temporal consistency (`LandmarkTemporalConsistency`)
Tracks frame-to-frame displacement of a fixed set of stable facial landmarks (nose bridge, forehead, chin region) and computes normalized jitter relative to the clip's own running statistics. This is self-referential: it does not compare against an external reference frame or assume any single frame represents ground-truth geometry.

Both unusually high jitter (erratic motion) and unusually low jitter (suspiciously smooth motion) are scored as mild evidence.

### Fusion (`AuthenticityEstimator`)
Combines all available signals into a single `manipulation_probability` (0-1) using fixed, documented weights:

| Signal | Weight |
|---|---|
| Spectral artifact | 0.40 |
| Landmark jitter | 0.25 |
| Blink rate | 0.15 |
| rPPG plausibility | 0.20 |

Unavailable signals are excluded from both the weighted sum and the weight denominator, so the probability is always computed only from evidence that actually exists. A separate `confidence` score reflects how much of the total possible evidence weight was actually available, independent of the probability value itself.

These weights are documented starting points based on the cited literature, not coefficients fit to labeled data. They should be validated and recalibrated against a labeled benchmark (e.g. FaceForensics++, DFDC, Celeb-DF) before any production use.

### Output (`AuthenticityAssessment`)
JSON-serializable result containing:

- `manipulation_probability`: 0-1 estimate, higher means more evidence of manipulation.
- `confidence`: 0-1, how much usable evidence was available.
- `evidence`: per-signal breakdown with availability, score, raw detail, and caveats.
- `content_sha256`: a content hash of the processed frames, for audit and reproducibility purposes. This is a plain digest, not a blockchain record.
- `human_review_recommended`: true when confidence is low or the probability falls in an ambiguous band (0.35-0.65).
- `disclaimer`: explicit statement that this is a statistical estimate from weak-to-moderate signals, not a forensic determination, and should not be sole grounds for action against a person.

## Orchestration (`FaceVideoAuthenticityPipeline`)

Minimal wiring class. Call `process_frame(face_crop_bgr, landmarks_xy)` per frame, then `finalize()` to get the `AuthenticityAssessment`. The pipeline does not include a face detector or landmarker; callers must supply face-cropped frames and, where available, landmark arrays (expected shape `(478, 2)`, compatible with MediaPipe FaceMesh indexing).

## Requirements

- numpy
- opencv-python (`cv2`)
- scipy

## Usage

```python
from face_video_authenticity import FaceVideoAuthenticityPipeline

pipeline = FaceVideoAuthenticityPipeline(fps=30.0)

for face_crop_bgr, landmarks_xy in frames:
    pipeline.process_frame(face_crop_bgr, landmarks_xy)

result = pipeline.finalize()
print(result.to_json())
```

Running the module directly (`python face_video_authenticity.py`) executes a smoke test against synthetic frames and prints the resulting JSON.

## Limitations

- Not validated against any labeled deepfake dataset. Fusion weights are not calibrated coefficients.
- Blink-rate and rPPG signals are weak and can be defeated or rendered uninformative by current generative methods.
- The spectral artifact detector responds to any high-frequency anomaly, including compression and upscaling artifacts unrelated to manipulation.
- No face detection, landmark extraction, or temporal alignment is provided; output quality depends entirely on the quality of inputs supplied by the caller.
- Designed for triage and prioritization, not as a sole basis for decisions affecting individuals.