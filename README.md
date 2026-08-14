# Resilient Deep Ensemble with Conformal Safety Shield & Test-Time Adaptation (TTA)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end implementation of a **probabilistic deep ensemble** integrated with **split conformal calibration**, **runtime safety verification**, and **unsupervised Test-Time Adaptation (TTA)** under simulated input sensor drift.

Developed as part of computational research into safe AI architectures, runtime verification, uncertainty quantification, and out-of-distribution (OOD) robustness.

---

## Key Features

### Probabilistic Deep Ensemble

An (M)-member neural-network ensemble estimates a predictive mean and predictive variance for each input:

[
\left{
\mu_m(x),\sigma_m^2(x)
\right}_{m=1}^{M}.
]

Each ensemble member is trained using a Gaussian Negative Log-Likelihood (NLL) objective, allowing the model to represent both predictive uncertainty and observation noise.

### Uncertainty Decomposition

The ensemble separates predictive uncertainty into:

* **Epistemic uncertainty** — uncertainty associated with model disagreement.
* **Aleatoric uncertainty** — uncertainty associated with inherent observation noise.

### Split Conformal Calibration

A calibration dataset is used to estimate a non-conformity threshold for the epistemic-uncertainty score.

For the reported experiment, the target miscoverage level is:

[
\alpha = 0.05,
]

corresponding to a nominal calibration level of:

[
1-\alpha = 0.95.
]

The resulting calibrated epistemic threshold is:

[
\hat{q}_{\text{epistemic}} = 0.005192.
]

> **Important:** This threshold is a calibrated bound on the selected uncertainty score under the assumptions of the conformal calibration procedure. It should not be interpreted as an unconditional 95% prediction-coverage guarantee under arbitrary distribution shift.

### Online Test-Time Adaptation (TTA)

The system performs unsupervised feature-normalization re-centering during inference using dynamic BatchNorm statistics.

No target-domain ground-truth labels are required during test-time adaptation.

### Runtime Safety Projection

A runtime safety layer monitors output-range invariants:

[
y \in [y_{\min},y_{\max}]
]

together with the calibrated epistemic-uncertainty threshold.

Predictions that violate the configured output constraints can be projected back into the permitted safety range.

---

## Empirical Benchmark Results

The system was evaluated under simulated physical sensor calibration drift with:

[
\delta = +1.5.
]

The experiment used 400 test samples.

| Metric                          | Pre-Adaptation (Fixed Training Statistics) | Post-Adaptation (Dynamic TTA) |      Performance Delta |
| :------------------------------ | -----------------------------------------: | ----------------------------: | ---------------------: |
| **Raw Test MSE**                |                                   `0.7578` |                  **`0.0137`** |   **98.19% reduction** |
| **Safety Intercepts Triggered** |                                  `0 / 400` |                     `0 / 400` | No violations observed |
| **Calibrated Epistemic Bound**  |                                 `0.005192` |                    `0.005192` |              Unchanged |

### Benchmark Interpretation

Under the simulated sensor calibration drift, dynamic TTA reduced the raw test MSE from:

[
0.7578
]

to:

[
0.0137.
]

This corresponds to a relative reduction of approximately:

[
98.19%.
]

At the same time, the runtime safety layer reported:

* **0/400 safety intercepts** before adaptation.
* **0/400 safety intercepts** after adaptation.
* A calibrated epistemic-uncertainty threshold of **0.005192**.
* A nominal conformal miscoverage level of:

[
\alpha = 0.05.
]

These results demonstrate the intended interaction between **distribution-shift adaptation**, **uncertainty estimation**, **conformal calibration**, and **runtime safety verification**.

---

## Mathematical Formulation

### 1. Ensemble Predictive Mean

For an (M)-member ensemble, let the (m)-th model produce:

[
\mu_m(x)
]

and

[
\sigma_m^2(x).
]

The ensemble predictive mean is:

[
\boxed{
\mu_{\text{ensemble}}(x)
========================

\frac{1}{M}
\sum_{m=1}^{M}
\mu_m(x)
}
]

where (M) is the number of ensemble members.

---

### 2. Epistemic Uncertainty

Epistemic uncertainty is estimated from disagreement between ensemble members.

Using the sample variance of the ensemble means:

[
\boxed{
\sigma_{\text{epistemic}}^2(x)
==============================

\frac{1}{M-1}
\sum_{m=1}^{M}
\left(
\mu_m(x)
--------

\mu_{\text{ensemble}}(x)
\right)^2
}
]

A larger value indicates greater disagreement among the ensemble members.

---

### 3. Aleatoric Uncertainty

Each ensemble member predicts an observation-noise variance:

[
\sigma_m^2(x).
]

The ensemble-average aleatoric variance is:

[
\boxed{
\sigma_{\text{aleatoric}}^2(x)
==============================

\frac{1}{M}
\sum_{m=1}^{M}
\sigma_m^2(x)
}
]

This represents the average data-dependent uncertainty estimated by the ensemble.

---

### 4. Total Predictive Variance

The total predictive variance can be decomposed into epistemic and aleatoric components:

[
\boxed{
\sigma_{\text{total}}^2(x)
==========================

\sigma_{\text{epistemic}}^2(x)
+
\sigma_{\text{aleatoric}}^2(x)
}
]

or equivalently:

[
\boxed{
\sigma_{\text{total}}^2(x)
==========================

\frac{1}{M-1}
\sum_{m=1}^{M}
\left(
\mu_m(x)
--------

\mu_{\text{ensemble}}(x)
\right)^2
+
\frac{1}{M}
\sum_{m=1}^{M}
\sigma_m^2(x)
}
]

where:

* (\sigma_{\text{epistemic}}^2(x)) represents **model-disagreement uncertainty**.
* (\sigma_{\text{aleatoric}}^2(x)) represents **data-dependent observation noise**.
* (\sigma_{\text{total}}^2(x)) represents the combined predictive variance.

---

## Conformal Calibration

Let the calibration dataset be:

[
D_{\text{calib}}
================

\left{
x_i
\right}*{i=1}^{N*{\text{calib}}}.
]

For each calibration sample, define the epistemic uncertainty score:

[
s_i
===

\sigma_{\text{epistemic}}^2(x_i).
]

The target miscoverage level is:

[
\alpha = 0.05.
]

The finite-sample conformal quantile index is:

[
k
=

\left\lceil
(N_{\text{calib}}+1)(1-\alpha)
\right\rceil.
]

The calibrated threshold is then obtained from the corresponding empirical order statistic:

[
\boxed{
\hat{q}_{\text{epistemic}}
==========================

s_{(k)}
}
]

where (s_{(k)}) denotes the (k)-th smallest calibration score, subject to the usual finite-sample conformal quantile convention.

For the reported experiment:

[
\boxed{
\hat{q}_{\text{epistemic}}
==========================

0.005192
}
]

with:

[
\alpha = 0.05.
]

Therefore, the nominal calibration level is:

[
\boxed{
1-\alpha = 0.95
}
]

### Interpretation

The calibrated threshold provides a reference boundary for the epistemic-uncertainty score during runtime monitoring.

The conformal interpretation depends on the assumptions of the calibration procedure, including the relevant exchangeability assumptions. The nominal (95%) level should therefore **not** be interpreted as an unconditional guarantee under arbitrary distribution shift.

---

## Runtime Uncertainty Check

During inference, the epistemic uncertainty is compared with the calibrated threshold:

[
\boxed{
\sigma_{\text{epistemic}}^2(x)
\leq
\hat{q}_{\text{epistemic}}
}
]

If:

[
\sigma_{\text{epistemic}}^2(x)

>

\hat{q}_{\text{epistemic}},
]

the sample is considered to exhibit elevated model disagreement and can be flagged by the runtime safety mechanism according to the implementation.

---

## System Architecture

The overall pipeline is:

```text
                 ┌──────────────────────────┐
                 │       Input Process x    │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Sensor / Input Drift   │
                 │        δ = +1.5          │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │  Test-Time Adaptation    │
                 │          (TTA)           │
                 │  Dynamic BatchNorm Stats │
                 └────────────┬─────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │          Deep Ensemble         │
              │                                │
              │  Model 1 → μ₁, σ₁²             │
              │  Model 2 → μ₂, σ₂²             │
              │  ...                           │
              │  Model M → μM, σM²             │
              └───────────────┬────────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Uncertainty Decomposition│
                 │                          │
                 │ Epistemic + Aleatoric    │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Conformal Calibration  │
                 │                          │
                 │ σ²_epistemic ≤ q̂         │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Runtime Safety Guard   │
                 │                          │
                 │ y ∈ [ymin, ymax]         │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Safe Prediction      │
                 └──────────────────────────┘
```

---

## Repository Structure

```text
Resilient-Deep-Ensemble-TTA/
│
├── main.py              # Main execution script
├── README.md            # Repository documentation
└── .gitignore           # Python temporary files and cache exclusion
```

---

## Quick Start

### Prerequisites

Python **3.8 or later** is required.

Install the dependencies:

```bash
pip install torch numpy matplotlib
```

### Run the Experiment

Navigate to the repository directory and execute:

```bash
python main.py
```

The program will:

1. Train/evaluate the ensemble.
2. Estimate predictive uncertainty.
3. Perform conformal calibration.
4. Introduce the simulated sensor drift.
5. Evaluate the pre-TTA model.
6. Perform unsupervised TTA.
7. Evaluate the post-TTA model.
8. Apply the runtime safety mechanism.
9. Generate diagnostic visualizations.

---

## Running in Python IDLE

1. Open **Python IDLE**.
2. Select **File → Open**.
3. Open `main.py`.
4. Press **F5**, or select **Run → Run Module**.

---

## Visualizations Generated

Upon execution, the script generates a **two-panel diagnostic plot**.

### Left Panel — Runtime Safety Shield & Online TTA

The left panel compares:

* Pre-adaptation predictions using fixed training statistics.
* Post-adaptation predictions using dynamic TTA.
* Ground-truth process values under simulated sensor drift.
* Configured runtime safety boundaries.

This visualization illustrates how TTA compensates for the simulated input-distribution shift.

### Right Panel — Epistemic Uncertainty & Calibration Threshold

The right panel displays:

* Epistemic uncertainty values.
* The calibrated epistemic threshold.
* The relationship between runtime uncertainty and the calibration boundary.

The reported threshold is:

[
\boxed{
\hat{q}_{\text{epistemic}}
==========================

0.005192
}
]

---

## Safety Mechanism

The runtime safety layer combines two principal checks.

### 1. Output Range Invariant

The predicted output must satisfy:

[
\boxed{
y \in [y_{\min},y_{\max}]
}
]

If a prediction falls outside the configured interval, it can be projected back into the permitted range.

Conceptually:

[
\boxed{
y_{\text{safe}}
===============

\min
\left(
\max(y,y_{\min}),
y_{\max}
\right)
}
]

### 2. Epistemic Uncertainty Constraint

The estimated epistemic uncertainty is compared against the calibrated threshold:

[
\boxed{
\sigma_{\text{epistemic}}^2(x)
\leq
\hat{q}_{\text{epistemic}}
}
]

If the threshold is exceeded, the runtime system can flag the prediction or apply the configured safety response.

This creates an additional monitoring layer for situations in which the ensemble exhibits unusually high model disagreement.

---

## Test-Time Adaptation

The TTA mechanism operates without requiring target labels during inference.

The core principle is to adapt feature-normalization statistics to accommodate the shifted input distribution.

Conceptually:

```text
Training Distribution
        │
        ▼
Fixed BatchNorm Statistics
        │
        ├──────────────► Pre-TTA Prediction
        │
        ▼
Shifted Test Distribution
        │
        ▼
Dynamic Feature Re-Centering
        │
        ▼
Post-TTA Prediction
```

The adaptation process is unsupervised with respect to the target labels and is intended to compensate for changes in the input distribution caused by sensor calibration drift.

---

## Experimental Configuration

The reported experiment uses:

| Configuration               |      Value      |
| :-------------------------- | :-------------: |
| Sensor drift                | (\delta = +1.5) |
| Target conformal level      |      (95%)      |
| Conformal miscoverage level | (\alpha = 0.05) |
| Calibrated epistemic bound  |    `0.005192`   |
| Test samples                |      `400`      |
| Pre-TTA MSE                 |     `0.7578`    |
| Post-TTA MSE                |     `0.0137`    |
| MSE reduction               |     `98.19%`    |
| Pre-TTA safety intercepts   |    `0 / 400`    |
| Post-TTA safety intercepts  |    `0 / 400`    |

---

## Reproducibility

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ikechukwukamalu8/Resilient-Deep-Ensemble-TTA.git
cd Resilient-Deep-Ensemble-TTA

pip install torch numpy matplotlib

python main.py
```

The implementation is designed to run in standard Python environments, including:

* Python Terminal
* Windows Command Prompt
* PowerShell
* Python IDLE
* Compatible Python IDEs

---

## Research Context

This implementation explores the combination of three complementary approaches to reliable machine learning.

### Probabilistic Deep Ensembles

Deep ensembles provide multiple predictive models whose disagreement can be used as an empirical estimate of epistemic uncertainty.

### Conformal Calibration

Split conformal calibration provides a distribution-free framework for calibrating non-conformity scores under the assumptions required by the selected conformal procedure.

### Test-Time Adaptation

TTA enables models to respond to distributional changes encountered during deployment without requiring labeled target-domain observations.

Together, these components form a layered architecture:

[
\boxed{
\text{Adaptation}
+
\text{Uncertainty Estimation}
+
\text{Conformal Calibration}
+
\text{Runtime Safety}
}
]

The objective is not simply to improve predictive accuracy, but to combine **adaptation, uncertainty estimation, statistical calibration, and explicit runtime safety mechanisms**.

---

## Important Statistical Notes

### Conformal Calibration

The reported (95%) level should be interpreted as a **nominal conformal calibration level** under the assumptions of the specific conformal procedure.

In particular, conformal guarantees rely on appropriate calibration and exchangeability assumptions. They should not be interpreted as unconditional guarantees under arbitrary distribution shift.

### Sensor Drift

The reported experiment uses a simulated sensor calibration drift:

[
\delta = +1.5.
]

Consequently, the observed improvement should be interpreted specifically within the experimental setting represented by the implementation.

### MSE Improvement

The reduction from:

[
0.7578
]

to:

[
0.0137
]

demonstrates substantial improvement under the reported simulated drift condition.

However, this result alone does not establish robustness against every possible real-world sensor failure, adversarial perturbation, or unseen distribution shift.

### Safety Intercepts

The reported:

[
0/400
]

safety intercept count means that no runtime safety intervention was recorded for the evaluated test samples under the configured conditions.

It does **not** imply that the safety mechanism would never trigger under other operating conditions.

---

## License

This project is released under the **MIT License**.

See the repository license file for the complete license terms.

---

## Author & Citation

**Ikechukwu Okechi Kamalu**

MSc Machine Learning Researcher & Biostatistician

**Email:** `ikechukwukamalu8@gmail.com`

**GitHub:** `github.com/ikechukwukamalu8`

**ORCID:** `0009-0008-3922-6310`

### Citation

If you use this implementation in academic or research work, please cite the project and acknowledge the author:

```text
Ikechukwu Okechi Kamalu
Resilient Deep Ensemble with Conformal Safety Shield
& Test-Time Adaptation (TTA)
```

---

## Project Summary

This project provides an end-to-end computational framework for investigating **safe and adaptive probabilistic machine learning under sensor distribution shift**.

The architecture combines:

1. **Probabilistic deep ensembles** for predictive uncertainty.
2. **Epistemic/aleatoric uncertainty decomposition**.
3. **Split conformal calibration** for uncertainty-score thresholding.
4. **Unsupervised test-time adaptation** for simulated sensor drift.
5. **Runtime safety projection** for enforcing output invariants.
6. **Diagnostic visualization** for evaluating adaptation and uncertainty.

Under the reported simulated sensor drift of:

[
\delta = +1.5,
]

the system reduced test MSE from:

[
0.7578
]

to:

[
0.0137,
]

corresponding to a **98.19% reduction in MSE**.

The runtime safety monitor recorded:

[
0/400
]

safety intercepts across the evaluated test samples.

The calibrated epistemic threshold was:

[
\hat{q}_{\text{epistemic}}
==========================

0.005192.
]

Overall, the project provides a computational framework for studying the intersection of:

**deep ensembles → uncertainty quantification → conformal calibration → test-time adaptation → runtime verification → safe AI.**
