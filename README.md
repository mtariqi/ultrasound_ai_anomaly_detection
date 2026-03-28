<div align="center">

```
██╗   ██╗██╗  ████████╗██████╗  █████╗      █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗███╗   ██╗███████╗████████╗
██║   ██║██║  ╚══██╔══╝██╔══██╗██╔══██╗    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝████╗  ██║██╔════╝╚══██╔══╝
██║   ██║██║     ██║   ██████╔╝███████║    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ ██╔██╗ ██║█████╗     ██║
██║   ██║██║     ██║   ██╔══██╗██╔══██║    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  ██║╚██╗██║██╔══╝     ██║
╚██████╔╝███████╗██║   ██║  ██║██║  ██║    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   ██║ ╚████║███████╗   ██║
 ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝  ╚═══╝╚══════╝   ╚═╝
```

### 🏥 Ultrasound Image Quality Assessment & Anomaly Detection Pipeline

*Ingest → Preprocess → Detect → Quantify Uncertainty → Report → Comply*

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![FDA SaMD](https://img.shields.io/badge/FDA-SaMD%20Class%20II-E8000B?style=for-the-badge)](https://www.fda.gov/medical-devices/software-medical-device-samd)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)](https://github.com/mtariqi)

</div>

---

## Overview

**UltraAnomalyNet** is a Python-based end-to-end medical imaging AI pipeline designed to demonstrate graduate-level competency in biostatistics, medical image analysis, signal processing, machine learning, and FDA AI/ML regulatory alignment.

The pipeline ingests simulated ultrasound frames and biosignals, extracts clinically relevant features, applies anomaly detection with uncertainty quantification, and produces structured clinical reports aligned with FDA Software as a Medical Device (SaMD) guidance — including intended use statements, model cards, and audit trails.

> *Designed to reflect real-world medical AI workflows: from raw imaging data to a regulatory-compliant clinical decision support output.*

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                              │
│   Ultrasound frames (DICOM)  │  ECG/EEG waveforms  │  Metadata    │
└──────────────┬───────────────┴────────────┬─────────┴──────────────┘
               ▼                            ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   IMAGE PREPROCESSING   │   │   SIGNAL PREPROCESSING  │
│  Denoise, norm, crop    │   │  Bandpass, FFT, segment │
└──────────┬──────────────┘   └──────────────┬──────────┘
           │                                 │
           ▼                                 ▼
┌────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                  │
│  SNR · Contrast · Artifact score · Spectral entropy   │
│  Sharpness · Peak frequency · Kurtosis · Amplitude    │
└────────────────────────────┬───────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│               ML ANOMALY DETECTION                     │
│  Isolation Forest  +  MC-Dropout Uncertainty (30 pass)│
│  Confidence score · Epistemic uncertainty estimate    │
└────────────────────────────┬───────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│              FDA SAMD COMPLIANCE LAYER                 │
│  Intended Use Statement · Severity scoring (5-tier)   │
│  FDA flag: REFER / REVIEW / PASS · Performance targets│
└────────────────────────────┬───────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│                      OUTPUTS                            │
│  Clinical report · JSON audit trail · Model card       │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Module | Capability | Clinical Relevance |
|---|---|---|
| 🖼️ **Image QA** | SNR, contrast ratio, artifact scoring, dynamic range, sharpness | Detects low-quality scans before AI inference — mirrors clinical QA workflows |
| 📡 **Signal Processing** | Bandpass filtering, Welch PSD, spectral entropy, kurtosis | ECG/EEG feature extraction for arrhythmia and seizure detection workflows |
| 🧠 **Anomaly Detection** | Isolation Forest with configurable contamination rate | Unsupervised detection handles class imbalance common in medical datasets |
| 🎲 **Uncertainty Quantification** | MC-Dropout proxy (30-pass perturbation) | Flags unreliable predictions — critical for FDA safety thresholds |
| ⚖️ **FDA Alignment** | SaMD Class II classification, intended use, performance targets | Reflects real 510(k)/De Novo submission requirements |
| 📄 **Audit Trail** | Full JSON trace of every inference decision | Supports post-market surveillance and regulatory traceability |

---

## Project Structure

```text
ultrasound-anomaly-detection/
│
├── 📄 README.md                ← This document
├── 🐍 pipeline.py              ← End-to-end analysis pipeline
├── 📋 MODEL_CARD.md            ← FDA-aligned model documentation
├── 🔬 signal_analysis.py       ← Standalone ECG/EEG signal module
├── 📊 evaluation.py            ← Performance benchmarking & ROC analysis
├── 🗃️ audit_trail.json         ← Sample regulatory audit log
├── 📦 requirements.txt         ← Python dependencies
├── 🚫 .gitignore
├── ⚖️  LICENSE
└── 📁 notebooks/
    ├── 01_image_quality_analysis.ipynb
    ├── 02_signal_processing_demo.ipynb
    └── 03_model_evaluation.ipynb
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core pipeline |
| Image Processing | NumPy, SciPy.ndimage | SNR, contrast, artifact estimation |
| Signal Processing | SciPy.signal | Butterworth filter, Welch PSD, spectral entropy |
| Machine Learning | scikit-learn | Isolation Forest anomaly detector, StandardScaler |
| Uncertainty | MC-Dropout proxy | Epistemic uncertainty quantification |
| Imaging I/O | pydicom | DICOM file ingestion (production) |
| Regulatory | Custom SaMD layer | FDA alignment, intended use, audit trail |

---

##  Getting Started

### Installation

```bash
git clone https://github.com/mtariqi/ultrasound-anomaly-detection.git
cd ultrasound-anomaly-detection
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Run on built-in simulated cases
python pipeline.py

# Export audit trail (auto-generated alongside report)
# → audit_trail.json
```

### Requirements

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
pydicom>=2.4
```

---

## 📋 Sample Output

```
══════════════════════════════════════════════════════════════════
  ULTRASOUND AI ANALYSIS — CLINICAL SUMMARY REPORT
  Device       : UltraAnomalyNet v1.0.0
  SaMD Class   : II — Serious situation, non-critical decision support
  Generated    : 2024-01-01 09:00:00
  Cases        : 6
══════════════════════════════════════════════════════════════════

📋  CASE FINDINGS

  ┌─ CASE-004  [🔴 REFER]
  │  Severity     : CRITICAL
  │  Confidence   : 94.2%  ±uncertainty 0.018
  │  Image grade  : C  (SNR=4.5 dB, Artifact=0.14)
  │  Finding      : High-confidence anomaly detected — immediate clinical review
  │  Finding      : Acoustic artifact detected (score=0.14)
  │  Audit ID     : 8d849293
  └──────────────────────────────────────────────────────────────

📊  COHORT SUMMARY
  🔴 REFER    ██ (2)
  🟡 REVIEW   █ (1)
  🟢 PASS     ███ (3)

⚖️   FDA COMPLIANCE NOTICE
  Intended use   : Computer-Aided Detection (CADe) — decision support only
  Performance tgt: Sensitivity≥90%, Specificity≥85%, AUC≥0.92
  ⚠  All REFER and REVIEW findings must be verified by a qualified clinician.
```

---

## 🗺️ FDA Regulatory Alignment

This project is structured to reflect the FDA's AI/ML-Based SaMD Action Plan (January 2021) and the 2023 Predetermined Change Control Plan guidance.

| Regulatory Requirement | Implementation |
|---|---|
| **Intended Use Statement** | `INTENDED_USE` dict — device name, SaMD class, indications, contraindications |
| **SaMD Classification** | Class II (serious situation, non-critical decision support) |
| **Performance Targets** | Sensitivity ≥90%, Specificity ≥85%, AUC ≥0.92, PPV ≥80% |
| **Uncertainty Quantification** | MC-Dropout proxy; predictions flagged when uncertainty ≥0.35 |
| **Audit Trail** | Full JSON log — sample ID, timestamp, confidence, findings, audit UUID |
| **Output Type** | CADe (Computer-Aided Detection) — decision support, not standalone diagnosis |
| **Human Oversight** | All REFER/REVIEW outputs require qualified clinician verification |
| **Model Card** | Separate `MODEL_CARD.md` — performance, bias, limitations, OOD risks |

---

## 🧪 Clinical Use Cases

- **Image Quality Gating** — Automatically reject scans below minimum SNR before AI inference, reducing false positives from poor-quality inputs
- **Hypoechoic Lesion Detection** — Flag potential cysts, nodules, or mass lesions for radiologist review
- **Acoustic Artifact Classification** — Distinguish pathology from imaging artifacts (acoustic shadow, reverberation)
- **ECG/EEG Quality Assessment** — Assess biosignal quality and extract diagnostic features for downstream analysis
- **CADe Workflow Integration** — Slot into existing PACS/RIS workflows as a pre-read screening tool

---

##  Learning Outcomes & Skills Demonstrated

- Applied **biostatistical feature engineering** to medical imaging data (SNR, contrast ratio, spectral entropy)
- Implemented **signal processing** pipelines: Butterworth bandpass filtering, Welch power spectral density, kurtosis analysis
- Designed an **ML anomaly detection** system with configurable sensitivity appropriate for clinical imbalanced datasets
- Implemented **uncertainty quantification** (MC-Dropout proxy) — a key FDA safety requirement for AI/ML medical devices
- Structured outputs to reflect **FDA SaMD regulatory requirements**: intended use, performance targets, audit trail, and model cards
- Demonstrated understanding of **clinical decision support** design principles: human-in-the-loop, output tiering, contraindication handling

---

## 🔮 Future Roadmap

- [ ] **Real DICOM ingestion** — pydicom integration with full DICOM tag extraction
- [ ] **Deep learning backbone** — Replace Isolation Forest with CNN (EfficientNet-B0) for image-level feature extraction
- [ ] **True MC-Dropout** — Implement dropout-enabled PyTorch/TensorFlow model for principled uncertainty estimation
- [ ] **SHAP explainability** — Generate feature importance maps for regulatory-grade model explanations
- [ ] **Prospective validation** — Benchmarking against public ultrasound datasets (BUSI, CLUST, US-4)
- [ ] **FHIR/HL7 output** — Structure clinical reports as FHIR DiagnosticReport resources for EHR integration
- [ ] **Federated learning** — Multi-site training without data sharing, per FDA guidance on real-world performance monitoring

---

## 📄 Citation

```bibtex
@software{islam_ultraanomalynet_2024,
  author    = {Islam, Md Tariqul},
  title     = {UltraAnomalyNet: Ultrasound Image Quality Assessment and Anomaly Detection Pipeline},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/mtariqi/ultrasound-anomaly-detection}
}
```

---

## 👤 Author

<div align="center">

**Md Tariqul Islam**

[![GitHub](https://img.shields.io/badge/GitHub-mtariqi-181717?style=for-the-badge&logo=github)](https://github.com/mtariqi)

*Biostatistics · Medical Imaging AI · Signal Processing · FDA Regulatory Science*

</div>

---

<div align="center">

*Advancing clinical decision support through principled, regulatory-aware AI engineering.*

⭐ Star this repo if it was useful — it supports continued open-source medical AI work.

</div>
