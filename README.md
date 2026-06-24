# CEI-Net: Conflict-Aware Evidential Inference for Multi-View Mammography

Official implementation of **CEI-Net**, introduced in:

> **When Views Disagree: Conflict-Aware Evidential Inference for Multi-View Mammography**


## Overview

Mammography screening commonly uses craniocaudal (CC) and mediolateral oblique (MLO) projections because they provide complementary views of the breast. However, the two views can differ in anatomical coverage, positioning, image quality, and lesion visibility. Consequently, they may provide conflicting diagnostic evidence.

Most multi-view learning methods combine features or probabilities deterministically. This can produce overly confident predictions when one view is ambiguous, degraded, or contradicts the other.

CEI-Net treats each mammographic view as an uncertain evidential source. Instead of forcing agreement during feature fusion, it represents each view with a Dirichlet belief distribution and explicitly models class-specific disagreement before fusion. Conflicting evidence is attenuated, while reliable evidence from an available view is retained.

The model outputs both class probabilities and epistemic uncertainty.

## Method

For each paired CC and MLO examination, CEI-Net performs:

```text
CC image and MLO image
        ↓
Shared ConvNeXt encoder
        ↓
Residual feature refinement
        ↓
Foreground-aware diverse Top-M token selection
        ↓
Token-feedback refinement
        ↓
Token-level evidential aggregation
        ↓
Per-view Dirichlet distributions
        ↓
Reliability-weighted, class-wise conflict-gated fusion
        ↓
Fused probabilities and epistemic uncertainty
```

### Core components

**Foreground-aware tokenization**

CEI-Net selects salient regional tokens from each refined feature map. A breast foreground mask suppresses background-dominated attention, while diversity-aware Top-M selection prevents tokens from collapsing into a single spatial region.

**Token-feedback refinement**

Selected tokens are projected back into the feature map through a gated residual update. This reinforces informative regional evidence before final token aggregation. The default implementation uses eight tokens and two feedback steps.

**Evidential aggregation**

Each token is mapped to nonnegative class evidence. Evidence is summed across tokens to form a view-specific Dirichlet distribution:

```text
alpha = evidence + 1
```

The Dirichlet mean represents class probability, while its total strength represents evidential confidence.

**Uncertainty-derived reliability**

For each view, epistemic uncertainty is computed from Dirichlet strength. Views with stronger committed evidence receive greater weight during fusion.

**Class-wise conflict gating**

CEI-Net computes Jensen–Shannon disagreement separately for each class hypothesis across CC and MLO predictions. Reliability-weighted evidence is then attenuated using a class-wise conflict gate. This reduces evidence only for disputed classes rather than forcing a confident benign or malignant decision.

When only one view is available, cross-view disagreement is set to zero and the model reduces to single-view evidential inference.

## Key Features

* Shared ConvNeXt-Tiny or ConvNeXt-Small encoder
* Foreground-aware diverse Top-M regional token selection
* Token-feedback refinement with retained token locations
* Dirichlet evidential learning with auxiliary per-view supervision
* Reliability weighting derived directly from predictive uncertainty
* Class-conditional Jensen–Shannon conflict gating
* Validation-only temperature scaling
* Patient or study grouped splitting to reduce identity leakage
* CC-only and MLO-only robustness evaluation

## Reported Results

The following values are reported in the manuscript as mean ± standard deviation across four independent runs.

| Model            |   CBIS-DDSM AUROC |     CBIS-DDSM ECE | VinDr-Mammo AUROC |   VinDr-Mammo ECE |
| ---------------- | ----------------: | ----------------: | ----------------: | ----------------: |
| Mammo-Clustering |     0.803 ± 0.002 |     0.304 ± 0.004 |     0.806 ± 0.003 |     0.234 ± 0.018 |
| CEI-Net-T        |     0.822 ± 0.003 |     0.118 ± 0.003 |     0.813 ± 0.004 |     0.079 ± 0.003 |
| **CEI-Net-S**    | **0.855 ± 0.005** | **0.109 ± 0.002** | **0.829 ± 0.005** | **0.073 ± 0.004** |

CEI-Net improves both discrimination and calibration by combining uncertainty-derived reliability weighting with class-specific conflict suppression.

## Repository Contents

```text
CEI-Net/
├── CEI-Net-S_CBSDDSM_VinDr_public.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

The notebook is self-contained and supports both CBIS-DDSM and VinDr-Mammo through a configuration option.

## Installation

Create an environment with PyTorch appropriate for your CUDA setup, then install the remaining dependencies:

```bash
pip install torch torchvision numpy pandas opencv-python pydicom scikit-learn tqdm pillow jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
CEI-Net-S_CBSDDSM_VinDr_public.ipynb
```

## Dataset Preparation

Users are responsible for obtaining the datasets from their official sources and complying with all access restrictions, licenses, and data-use agreements.

### CBIS-DDSM

The CBIS-DDSM pipeline uses mass cases and constructs paired CC and MLO samples by grouping images according to:

```text
patient ID + laterality + abnormality ID
```

Binary labels are derived from pathology:

```text
benign = 0
malignant = 1
```

The expected default structure is:

```text
data/
└── CBIS-DDSM/
    ├── csv/
    │   ├── mass_case_description_train_set.csv
    │   ├── mass_case_description_test_set.csv
    │   └── dicom_info.csv
    └── ...
```

Configure the notebook as follows:

```python
cfg.dataset_name = "cbis_ddsm"
cfg.cbis_root = Path("data/CBIS-DDSM")
cfg.cbis_image_source = "dicom"
```

### VinDr-Mammo

The VinDr-Mammo pipeline constructs breast-level CC and MLO pairs using:

```text
study ID + laterality
```

The binary label is derived from BI-RADS:

```text
BI-RADS < 4 = 0
BI-RADS ≥ 4 = 1
```

When multiple images exist for the same view, the implementation retains the image with the largest foreground breast area after preprocessing.

Configure the notebook as follows:

```python
cfg.dataset_name = "vindr"
cfg.vindr_image_root = Path("data/VinDr-Mammo")
cfg.vindr_annotations_csv = Path("data/vindr_detection_v1_folds.csv")
cfg.split_level = "study"
```

Use patient-level splitting instead when reliable patient identifiers are available:

```python
cfg.split_level = "patient"
```

## Training and Evaluation Protocol

The notebook applies grouped data splitting to reduce patient or study overlap across partitions.

```text
80% train and development partition
20% held-out test partition
15% of the train and development partition used for validation
```

The validation partition is used for:

* early stopping
* checkpoint selection based on AUROC
* temperature fitting for post-hoc calibration

The held-out test partition is evaluated only after the best validation checkpoint has been selected.

Default settings include:

```text
Backbone: ConvNeXt-Small
Image size: 224 × 224
Batch size: 8
Maximum epochs: 100
Optimizer: Adam
Learning rate schedule: cosine annealing
Top-M tokens: 8
Token-feedback steps: 2
```

To reproduce multi-run results, repeat training with independent seeds and aggregate AUROC, ECE, and NLL across runs.

## Outputs

Each run saves its outputs under:

```text
outputs/ceinet_s_<dataset_name>/
```

The output directory contains:

```text
best_ceinet_s.pt
config.json
train_pairs.csv
validation_pairs.csv
test_pairs.csv
history.json
results.json
test_predictions.npz
```

## Missing-View Evaluation

The notebook optionally evaluates missing-view robustness after training:

```python
cfg.evaluate_missing_views = True
```

This reports performance for:

```text
CC-only inference
MLO-only inference
Full CC + MLO inference
```

## Citation

Use the following anonymized citation while the manuscript is under review:

```bibtex
@misc{rayhan2026ceinet,
  title={When Views Disagree: Conflict-Aware Evidential Inference for Multi-View Mammography},
  author={Md Rayhan Ahmed, Patricia Lasserre},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2026}
}
```

Update the author list, publication venue, and DOI after acceptance.

## Disclaimer

This repository is provided for research and educational purposes only. CEI-Net is not a clinical diagnostic system and must not be used for patient care, clinical triage, or treatment decisions without prospective validation, clinical oversight, and appropriate regulatory approval.
