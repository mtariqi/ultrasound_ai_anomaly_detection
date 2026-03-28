"""
pipeline.py
═══════════════════════════════════════════════════════════════════════
Ultrasound Image Quality Assessment & Anomaly Detection Pipeline
FDA-Aligned AI/ML Medical Imaging Analysis System

Author : Md Tariqul Islam
Purpose: End-to-end pipeline for ultrasound image analysis with
         MITRE-style threat mapping replaced by FDA SaMD classification,
         confidence scoring, uncertainty quantification, and structured
         clinical reporting aligned with FDA AI/ML Action Plan guidance.

References:
  - FDA Artificial Intelligence/Machine Learning (AI/ML)-Based Software
    as a Medical Device (SaMD) Action Plan, Jan 2021
  - FDA Guidance: Marketing Submission Recommendations for a Predetermined
    Change Control Plan for AI/ML-Enabled Devices, 2023
  - IEC 62304: Medical device software lifecycle processes
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import sys
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── FDA SaMD Classification Reference ────────────────────────────────────────
# Software as a Medical Device risk classification per FDA guidance
SAMD_CLASSES = {
    "I":   "Non-serious situation, non-critical decision support",
    "II":  "Serious situation, non-critical OR non-serious, critical decision support",
    "III": "Serious situation, critical decision support",
}

# Intended Use Statement (IUS) — required for FDA 510(k) / De Novo submissions
INTENDED_USE = {
    "device_name":      "UltraAnomalyNet",
    "version":          "1.0.0",
    "samd_class":       "II",
    "intended_use":     (
        "The device is intended to assist qualified clinicians in identifying "
        "image quality anomalies and potential pathological features in 2D "
        "B-mode ultrasound images. It is not intended to replace clinical "
        "judgement or serve as a standalone diagnostic device."
    ),
    "indications":      "Adults undergoing abdominal or cardiac ultrasound examination",
    "contraindications":"Paediatric imaging; 3D/4D ultrasound modes; elastography",
    "output_type":      "Computer-Aided Detection (CADe) — decision support only",
}

# ── Performance Thresholds (FDA-relevant metrics) ─────────────────────────────
PERFORMANCE_TARGETS = {
    "sensitivity":    0.90,   # True positive rate (recall)
    "specificity":    0.85,   # True negative rate
    "ppv":            0.80,   # Positive predictive value (precision)
    "auc_roc":        0.92,   # Area under ROC curve
    "uncertainty_cap":0.35,   # Max acceptable epistemic uncertainty
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ImageQualityMetrics:
    """Signal-to-noise ratio, contrast, and artifact scores for one frame."""
    snr_db:          float
    contrast_ratio:  float
    artifact_score:  float   # 0 = clean, 1 = severe artifact
    dynamic_range:   float
    sharpness:       float
    quality_grade:   str     # A / B / C / REJECT

@dataclass
class SignalFeatures:
    """Frequency-domain and statistical features extracted from a waveform."""
    mean_amplitude:   float
    std_amplitude:    float
    peak_frequency_hz:float
    spectral_entropy: float
    kurtosis:         float
    dominant_band:    str

@dataclass
class AnomalyResult:
    """Output of the anomaly detection module for one sample."""
    sample_id:           str
    timestamp:           str
    anomaly_detected:    bool
    confidence:          float   # 0–1
    uncertainty:         float   # epistemic uncertainty (MC-Dropout estimate)
    severity:            str     # CRITICAL / HIGH / MEDIUM / LOW / NORMAL
    findings:            list[str]
    quality_metrics:     ImageQualityMetrics
    signal_features:     Optional[SignalFeatures]
    fda_flag:            str     # REFER / REVIEW / PASS
    audit_id:            str     = field(default_factory=lambda: str(uuid.uuid4())[:8])


# ── Module 1: Image Simulation & Quality Assessment ───────────────────────────

def simulate_ultrasound_frame(
    size: int = 128,
    add_artifact: bool = False,
    add_lesion: bool = False,
    noise_level: float = 0.15,
) -> np.ndarray:
    """
    Simulate a 2D B-mode ultrasound frame with speckle noise.
    In a production system this would be replaced by DICOM ingestion
    via pydicom: ds = pydicom.dcmread(filepath); img = ds.pixel_array
    """
    rng = np.random.default_rng(seed=42)
    # Tissue background with speckle
    tissue   = rng.rayleigh(scale=0.4, size=(size, size))
    speckle  = rng.normal(0, noise_level, (size, size))
    frame    = tissue + speckle

    # Simulate acoustic shadow (dark stripe — common artifact)
    if add_artifact:
        frame[:, size//2 - 4 : size//2 + 4] *= 0.05

    # Simulate hypoechoic lesion (darker circular region)
    if add_lesion:
        cx, cy, r = size // 3, size // 3, size // 8
        Y, X = np.ogrid[:size, :size]
        mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        frame[mask] *= 0.25

    return np.clip(frame, 0, 1)


def compute_image_quality(img: np.ndarray) -> ImageQualityMetrics:
    """
    Compute standard ultrasound image quality metrics.

    SNR   : signal power relative to noise floor
    CR    : contrast ratio between foreground and background regions
    Sharpness: Laplacian variance (higher = sharper edges)
    """
    # Estimate signal & noise via local mean / std
    local_mean = uniform_filter(img, size=8)
    noise_est  = img - local_mean
    signal_pow = np.mean(local_mean**2)
    noise_pow  = np.var(noise_est) + 1e-10
    snr_db     = 10 * np.log10(signal_pow / noise_pow)

    # Contrast ratio: top quartile vs bottom quartile intensities
    p75, p25   = np.percentile(img, 75), np.percentile(img, 25)
    cr         = (p75 - p25) / (p75 + p25 + 1e-10)

    # Artifact score: fraction of near-zero pixels (acoustic shadow proxy)
    artifact   = np.mean(img < 0.05)

    # Dynamic range
    dyn_range  = float(img.max() - img.min())

    # Sharpness via Laplacian variance
    from scipy.ndimage import laplace
    sharpness  = float(np.var(laplace(img)))

    # Grade assignment
    if snr_db >= 12 and artifact < 0.05:
        grade = "A"
    elif snr_db >= 8 and artifact < 0.15:
        grade = "B"
    elif snr_db >= 4 and artifact < 0.30:
        grade = "C"
    else:
        grade = "REJECT"

    return ImageQualityMetrics(
        snr_db         = round(snr_db, 2),
        contrast_ratio = round(float(cr), 3),
        artifact_score = round(float(artifact), 3),
        dynamic_range  = round(dyn_range, 3),
        sharpness      = round(sharpness, 5),
        quality_grade  = grade,
    )


# ── Module 2: Signal Processing (ECG / EEG) ──────────────────────────────────

def simulate_ecg_signal(
    duration_s: float = 5.0,
    fs: float = 500.0,
    add_noise: bool = True,
    add_artifact: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a synthetic ECG waveform at given sampling rate."""
    t = np.linspace(0, duration_s, int(fs * duration_s))
    # QRS complex approximation via sum of Gaussians
    ecg = np.zeros_like(t)
    hr  = 75  # bpm
    beat_interval = fs * 60 / hr
    for beat_idx in range(int(len(t) / beat_interval)):
        c = int(beat_idx * beat_interval)
        if c >= len(t): break
        window = np.arange(max(0, c-20), min(len(t), c+20))
        # P wave
        ecg[max(0,c-15):max(0,c-10)] += 0.15 * np.exp(
            -0.5*((np.arange(max(0,c-15),max(0,c-10))-c+12)/2)**2)
        # QRS
        ecg[window] += np.exp(-0.5*((window - c)/2.5)**2)
        # T wave
        ecg[min(len(t)-1,c+10):min(len(t),c+25)] += 0.25 * np.exp(
            -0.5*((np.arange(min(len(t)-1,c+10),min(len(t),c+25))-c-17)/5)**2)

    if add_noise:
        ecg += np.random.normal(0, 0.03, len(t))
    if add_artifact:
        # Baseline wander
        ecg += 0.3 * np.sin(2 * np.pi * 0.3 * t)

    return t, ecg


def extract_signal_features(t: np.ndarray, sig: np.ndarray, fs: float = 500.0) -> SignalFeatures:
    """
    Extract frequency-domain and statistical features from a biosignal.
    Relevant for arrhythmia detection, seizure detection, and waveform QC.
    """
    # Bandpass filter: 0.5–40 Hz (ECG diagnostic range per AHA guidelines)
    b, a   = scipy_signal.butter(4, [0.5, 40.0], btype="band", fs=fs)
    filt   = scipy_signal.filtfilt(b, a, sig)

    # Power spectral density
    freqs, psd = scipy_signal.welch(filt, fs=fs, nperseg=min(256, len(filt)))
    peak_freq  = float(freqs[np.argmax(psd)])
    total_pow  = np.sum(psd) + 1e-10

    # Spectral entropy (Shannon)
    psd_norm    = psd / total_pow
    spec_entropy= float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))

    # Statistical features
    from scipy.stats import kurtosis as sp_kurtosis
    kurt = float(sp_kurtosis(filt))

    # Dominant frequency band
    if peak_freq < 0.5:
        band = "sub-Hz (baseline wander)"
    elif peak_freq < 4:
        band = "0.5–4 Hz (low)"
    elif peak_freq < 15:
        band = "4–15 Hz (QRS dominant)"
    else:
        band = "15–40 Hz (high frequency)"

    return SignalFeatures(
        mean_amplitude   = round(float(np.mean(np.abs(filt))), 4),
        std_amplitude    = round(float(np.std(filt)), 4),
        peak_frequency_hz= round(peak_freq, 2),
        spectral_entropy = round(spec_entropy, 3),
        kurtosis         = round(kurt, 3),
        dominant_band    = band,
    )


# ── Module 3: ML Anomaly Detection (Isolation Forest + MC-Dropout proxy) ─────

def build_feature_vector(qm: ImageQualityMetrics, sf: Optional[SignalFeatures] = None) -> np.ndarray:
    """Assemble a flat feature vector from quality and signal metrics."""
    img_feats = [
        qm.snr_db,
        qm.contrast_ratio,
        qm.artifact_score,
        qm.dynamic_range,
        qm.sharpness,
    ]
    if sf:
        sig_feats = [
            sf.mean_amplitude,
            sf.std_amplitude,
            sf.peak_frequency_hz,
            sf.spectral_entropy,
            sf.kurtosis,
        ]
        return np.array(img_feats + sig_feats)
    return np.array(img_feats)


def train_anomaly_model(n_samples: int = 200) -> tuple[IsolationForest, StandardScaler]:
    """
    Train an Isolation Forest on synthetic normal-distribution samples.
    In production: replace with retrospectively labelled clinical data,
    with training/validation/test splits stratified by site and device.
    """
    rng   = np.random.default_rng(0)
    # Normal: high SNR, low artifact, moderate sharpness
    X_normal = rng.normal(
        loc  =[12.0, 0.45, 0.02, 0.80, 0.0004],
        scale=[2.0,  0.08, 0.01, 0.08, 0.0001],
        size =(n_samples, 5)
    )
    scaler = StandardScaler().fit(X_normal)
    model  = IsolationForest(
        n_estimators   = 200,
        contamination  = 0.08,
        random_state   = 42,
    ).fit(scaler.transform(X_normal))
    return model, scaler


def monte_carlo_uncertainty(
    model: IsolationForest,
    scaler: StandardScaler,
    x: np.ndarray,
    n_passes: int = 30,
) -> tuple[float, float]:
    """
    Estimate epistemic uncertainty via score variance across perturbed inputs
    (MC-Dropout proxy — in a deep learning model this would be actual
    dropout inference passes as per Gal & Ghahramani, 2016).
    Returns (mean_score_normalised, uncertainty_std).
    """
    rng    = np.random.default_rng(1)
    scores = []
    for _ in range(n_passes):
        x_perturbed = x + rng.normal(0, 0.02, x.shape)
        score       = model.decision_function(scaler.transform(x_perturbed.reshape(1, -1)))[0]
        scores.append(score)
    mean_score  = float(np.mean(scores))
    uncertainty = float(np.std(scores))
    # Normalise to [0, 1] confidence
    confidence  = float(np.clip(1.0 / (1.0 + np.exp(-mean_score * 3)), 0, 1))
    return confidence, uncertainty


# ── Module 4: Severity & FDA Flag Assignment ──────────────────────────────────

def assign_severity(confidence: float, uncertainty: float, qm: ImageQualityMetrics) -> tuple[str, str, list[str]]:
    """
    Map detection results to severity tier and FDA action flag.

    FDA flag logic:
      REFER  — confident anomaly, low uncertainty → escalate to radiologist
      REVIEW — uncertain or borderline → human review required
      PASS   — confident normal, acceptable quality
    """
    findings = []
    anomaly  = confidence > 0.60

    if qm.quality_grade == "REJECT":
        findings.append("Image quality below minimum threshold — rescan required")
    if qm.artifact_score > 0.15:
        findings.append(f"Acoustic artifact detected (score={qm.artifact_score:.2f})")
    if qm.snr_db < 6:
        findings.append(f"Low SNR ({qm.snr_db:.1f} dB) — possible probe contact issue")
    if qm.contrast_ratio < 0.20:
        findings.append("Reduced tissue contrast — verify gain settings")

    # Severity
    if anomaly and uncertainty < PERFORMANCE_TARGETS["uncertainty_cap"]:
        if confidence > 0.90:
            severity = "CRITICAL"
            findings.append("High-confidence anomaly detected — immediate clinical review")
        elif confidence > 0.75:
            severity = "HIGH"
            findings.append("Probable anomaly — clinical correlation recommended")
        else:
            severity = "MEDIUM"
            findings.append("Possible anomaly — supplementary imaging advised")
    elif uncertainty >= PERFORMANCE_TARGETS["uncertainty_cap"]:
        severity = "LOW"
        findings.append(f"High model uncertainty ({uncertainty:.3f}) — result unreliable")
    else:
        severity = "NORMAL"
        findings.append("No anomaly detected within detection threshold")

    # FDA flag
    if severity in ("CRITICAL", "HIGH") and uncertainty < 0.25:
        fda_flag = "REFER"
    elif severity in ("MEDIUM", "LOW") or uncertainty >= 0.25:
        fda_flag = "REVIEW"
    else:
        fda_flag = "PASS"

    return severity, fda_flag, findings


# ── Module 5: Full Pipeline ────────────────────────────────────────────────────

def run_pipeline(
    n_cases: int = 6,
    model: Optional[IsolationForest] = None,
    scaler: Optional[StandardScaler] = None,
) -> list[AnomalyResult]:
    """
    Run the end-to-end pipeline on a batch of simulated cases.
    Case definitions simulate a representative clinical mix:
      - Normal studies
      - Artifact-only cases (quality failure)
      - True pathological anomalies
      - Borderline / uncertain cases
    """
    if model is None or scaler is None:
        model, scaler = train_anomaly_model()

    cases = [
        {"label": "Normal study",              "artifact": False, "lesion": False, "noise": 0.10, "ecg_artifact": False},
        {"label": "Acoustic shadow artifact",  "artifact": True,  "lesion": False, "noise": 0.12, "ecg_artifact": False},
        {"label": "Hypoechoic lesion",         "artifact": False, "lesion": True,  "noise": 0.10, "ecg_artifact": False},
        {"label": "Artifact + lesion",         "artifact": True,  "lesion": True,  "noise": 0.20, "ecg_artifact": True},
        {"label": "High-noise / poor contact", "artifact": False, "lesion": False, "noise": 0.45, "ecg_artifact": True},
        {"label": "Subtle lesion (borderline)","artifact": False, "lesion": True,  "noise": 0.08, "ecg_artifact": False},
    ]

    results = []
    for i, case in enumerate(cases[:n_cases]):
        sample_id = f"CASE-{i+1:03d}"
        ts        = datetime.now().isoformat(timespec="seconds")

        # Image branch
        img = simulate_ultrasound_frame(
            size=128,
            add_artifact=case["artifact"],
            add_lesion  =case["lesion"],
            noise_level =case["noise"],
        )
        qm  = compute_image_quality(img)

        # Signal branch
        _, ecg = simulate_ecg_signal(add_artifact=case["ecg_artifact"])
        sf     = extract_signal_features(
            np.linspace(0, 5, len(ecg)), ecg, fs=500.0
        )

        # Feature extraction & anomaly scoring
        x            = build_feature_vector(qm, sf)
        confidence, uncertainty = monte_carlo_uncertainty(model, scaler, x[:5])

        # Severity & FDA flag
        severity, fda_flag, findings = assign_severity(confidence, uncertainty, qm)

        results.append(AnomalyResult(
            sample_id        = sample_id,
            timestamp        = ts,
            anomaly_detected = confidence > 0.60,
            confidence       = round(confidence, 3),
            uncertainty      = round(uncertainty, 3),
            severity         = severity,
            findings         = findings,
            quality_metrics  = qm,
            signal_features  = sf,
            fda_flag         = fda_flag,
        ))

    return results


# ── Module 6: Clinical Report & Audit Trail ───────────────────────────────────

def print_clinical_report(results: list[AnomalyResult]) -> None:
    """Print a structured, FDA-aligned clinical summary report."""
    sep = "═" * 66
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{sep}")
    print("  ULTRASOUND AI ANALYSIS — CLINICAL SUMMARY REPORT")
    print(f"  Device       : {INTENDED_USE['device_name']} v{INTENDED_USE['version']}")
    print(f"  SaMD Class   : {INTENDED_USE['samd_class']} — {SAMD_CLASSES[INTENDED_USE['samd_class']]}")
    print(f"  Generated    : {now}")
    print(f"  Cases        : {len(results)}")
    print(sep)

    # Case results
    print("\n📋  CASE FINDINGS")
    print("─" * 66)
    flag_icons = {"REFER": "🔴 REFER", "REVIEW": "🟡 REVIEW", "PASS": "🟢 PASS"}
    for r in results:
        qm = r.quality_metrics
        print(f"\n  ┌─ {r.sample_id}  [{flag_icons.get(r.fda_flag, r.fda_flag)}]")
        print(f"  │  Severity     : {r.severity}")
        print(f"  │  Confidence   : {r.confidence:.1%}  ±uncertainty {r.uncertainty:.3f}")
        print(f"  │  Image grade  : {qm.quality_grade}  (SNR={qm.snr_db:.1f} dB, Artifact={qm.artifact_score:.2f})")
        for finding in r.findings:
            print(f"  │  Finding      : {finding}")
        print(f"  │  Audit ID     : {r.audit_id}")
        print(f"  └─{'─' * 58}")

    # Summary statistics
    print("\n\n📊  COHORT SUMMARY")
    print("─" * 66)
    flags   = [r.fda_flag  for r in results]
    sevs    = [r.severity  for r in results]
    grades  = [r.quality_metrics.quality_grade for r in results]
    mean_c  = np.mean([r.confidence  for r in results])
    mean_u  = np.mean([r.uncertainty for r in results])

    for flag in ("REFER", "REVIEW", "PASS"):
        count = flags.count(flag)
        bar   = "█" * count
        print(f"  {flag_icons[flag]:<18} {bar} ({count})")

    print(f"\n  Mean confidence  : {mean_c:.1%}")
    print(f"  Mean uncertainty : {mean_u:.3f}")
    print(f"  Quality grades   : {', '.join(grades)}")

    # FDA compliance note
    print("\n\n⚖️   FDA COMPLIANCE NOTICE")
    print("─" * 66)
    print(f"  Intended use   : {INTENDED_USE['output_type']}")
    print(f"  Output type    : Decision support — not a standalone diagnosis")
    print(f"  Indications    : {INTENDED_USE['indications']}")
    print(f"  Contraindics   : {INTENDED_USE['contraindications']}")
    print(f"  Performance tgt: Sensitivity≥{PERFORMANCE_TARGETS['sensitivity']:.0%}, "
          f"Specificity≥{PERFORMANCE_TARGETS['specificity']:.0%}, "
          f"AUC≥{PERFORMANCE_TARGETS['auc_roc']}")
    print(f"\n  ⚠  All REFER and REVIEW findings must be verified by a qualified")
    print(f"     clinician before any clinical action is taken.")

    print(f"\n{sep}")
    print("  END OF REPORT")
    print(f"{sep}\n")


def export_audit_trail(results: list[AnomalyResult], path: str = "audit_trail.json") -> None:
    """Export full audit trail to JSON for regulatory traceability."""
    trail = {
        "generated":    datetime.now().isoformat(),
        "device":       INTENDED_USE,
        "performance":  PERFORMANCE_TARGETS,
        "samd_classes": SAMD_CLASSES,
        "cases":        [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(trail, f, indent=2)
    print(f"[+] Audit trail exported → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("[*] Training anomaly detection model...")
    model, scaler = train_anomaly_model()

    print("[*] Running pipeline on simulated clinical cases...")
    results = run_pipeline(model=model, scaler=scaler)

    print_clinical_report(results)
    export_audit_trail(results, path="audit_trail.json")


if __name__ == "__main__":
    main()
