# Resilient Deep Ensemble with Conformal Safety Shield & Test-Time Adaptation (TTA)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end implementation of a **probabilistic deep ensemble** integrated with **Split Conformal Prediction** runtime verification and **unsupervised Test-Time Adaptation (TTA)** under severe input sensor drift.

Developed as part of computational research into safe AI architectures, runtime verification, and out-of-distribution (OOD) robustness.

---

## Key Features

* **Probabilistic Deep Ensemble:**
  An $M$-member neural network ensemble estimating predictive mean $\mu(x)$ and log-variance $\log\sigma^2(x)$ under Gaussian Negative Log-Likelihood (NLL) loss.

* **Uncertainty Decomposition:**
  Separates predictive variance into **epistemic uncertainty** (model disagreement across ensemble members) and **aleatoric uncertainty** (inherent data noise).

* **Split Conformal Prediction Shield:**
  Calibrates a certified non-conformity threshold $\hat{q}_{\text{epistemic}}$ on clean calibration data at a target coverage guarantee of:

  $$1 - \alpha = 0.95$$

* **Online Test-Time Adaptation (TTA):**
  Performs unsupervised feature-normalization re-centering during inference using dynamic BatchNorm statistics without requiring true target $y$ labels.

* **Runtime Safety Projection:**
  A safety guard monitors range invariants:

  $$y \in [y_{\min}, y_{\max}]$$

  together with epistemic uncertainty limits, projecting out-of-bound predictions back into certified safety boundaries.

---

## Empirical Benchmark Results

The system was evaluated under active physical sensor calibration drift with:

$$\delta = +1.5$$

applied to the input process data.

| Metric                          | Pre-Adaptation (Fixed Training Stats) | Post-Adaptation (Dynamic TTA) |      Performance Delta      |
| :------------------------------ | :-----------------------------------: | :---------------------------: | :-------------------------: |
| **Raw Test MSE**                |                `0.7578`               |          **`0.0137`**         |     **98.19% Reduction**    |
| **Safety Intercepts Triggered** |               `0 / 400`               |           `0 / 400`           |     Satisfied Invariants    |
| **Calibrated Epistemic Bound**  |               `0.005192`              |           `0.005192`          | Certified ($\alpha = 0.05$) |

### Benchmark Interpretation

Under the simulated sensor calibration drift, dynamic TTA substantially reduced the raw test MSE from `0.7578` to `0.0137`, representing a **98.19% reduction in error**.

At the same time, the runtime safety layer reported:

* **0/400 safety intercepts** before adaptation.
* **0/400 safety intercepts** after adaptation.
* A stable calibrated epistemic uncertainty bound of **0.005192**.
* A target conformal error rate of $\alpha = 0.05$.

These results demonstrate the intended interaction between **distribution-shift adaptation**, **uncertainty estimation**, and **runtime safety verification**.

---

## Mathematical Formulation

### 1. Epistemic & Aleatoric Variance Decomposition

For ensemble predictions:

$$
{\mu_m(x), \sigma_m^2(x)}_{m=1}^{M}
$$

the ensemble predictive mean is:

$$
\mu_{\text{ensemble}}(x)
========================

\frac{1}{M}
\sum_{m=1}^{M}
\mu_m(x)
$$

The epistemic variance, representing model disagreement, is:

$$
\sigma^2_{\text{epistemic}}(x)
==============================

\frac{1}{M-1}
\sum_{m=1}^{M}
\left(
\mu_m(x)
--------

\mu_{\text{ensemble}}(x)
\right)^2
$$

The total predictive variance is decomposed as:

$$
\sigma^2_{\text{total}}(x)
==========================

\sigma^2_{\text{epistemic}}(x)
+
\frac{1}{M}
\sum_{m=1}^{M}
\sigma_m^2(x)
$$

where:

* $\sigma^2_{\text{epistemic}}$ represents **epistemic uncertainty**.
* $\sigma_m^2$ represents **aleatoric uncertainty** for ensemble member $m$.
* $\sigma^2_{\text{total}}$ represents the combined predictive uncertainty.

---

### 2. Conformal Prediction Bound

Given a calibration dataset:

$$
D_{\text{calib}}
$$

and an error rate:

$$
\alpha = 0.05
$$

the non-conformity threshold is computed from the empirical distribution of epistemic uncertainty:

$$
\hat{q}
=======

\text{Quantile}
\left(
\left{
\sigma^2_{\text{epistemic}}(x_i)
\right}*{i=1}^{N*{\text{calib}}},
\frac{
\left\lceil
(N_{\text{calib}}+1)(1-\alpha)
\right\rceil
}{
N_{\text{calib}}
}
\right)
$$

The resulting threshold provides the runtime verification boundary used by the conformal safety shield.

For the reported experiment:

$$
\hat{q}_{\text{epistemic}} = 0.005192
$$

with:

$$
\alpha = 0.05
$$

and therefore a nominal target coverage of:

$$
1-\alpha = 0.95
$$

---

## System Architecture

The overall pipeline can be summarized as:

```text
                 ┌──────────────────────────┐
                 │      Input Process x     │
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
                 │     Test-Time Adaptation │
                 │          (TTA)           │
                 │ Dynamic BatchNorm Stats  │
                 └────────────┬─────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │       Deep Ensemble            │
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
                 │ Conformal Safety Shield  │
                 │                          │
                 │ σ²_epistemic ≤ q̂         │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Runtime Safety Projection│
                 │                          │
                 │ y ∈ [ymin, ymax]          │
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
├── README.md            # Comprehensive repository documentation
└── .gitignore           # Python temporary files and cache exclusion
```

---

## Quick Start & Local Execution

### Prerequisites

Ensure that **Python 3.8 or later** is installed.

Install the required dependencies:

```bash
pip install torch numpy matplotlib
```

---

## Running in Terminal

Navigate to the repository directory and execute:

```bash
python main.py
```

The program will train/evaluate the ensemble, perform calibration, execute TTA under sensor drift, apply the runtime safety shield, and generate the diagnostic visualization.

---

## Running in Python IDLE

1. Open **Python IDLE**.
2. Select **File → Open**.
3. Open `main.py`.
4. Press **F5**, or select **Run → Run Module**.

---

## Visualizations Generated

Upon execution, the script generates a **two-panel diagnostic plot**.

### Left Plot — Runtime Safety Shield & Online TTA

The left panel compares:

* Pre-adaptation predictions using fixed training statistics.
* Post-adaptation predictions using dynamic TTA.
* Ground-truth process values under sensor drift.
* Runtime safety boundaries.

This visualization demonstrates the ability of TTA to compensate for distributional changes in the input sensor data.

### Right Plot — Epistemic Uncertainty & Conformal Threshold

The right panel displays:

* The epistemic uncertainty distribution.
* The calibrated conformal verification threshold.
* The relationship between runtime uncertainty and the certified uncertainty boundary.

The threshold is:

$$
\hat{q}_{\text{epistemic}} = 0.005192
$$

---

## Safety Mechanism

The runtime safety layer combines two primary checks.

### 1. Output Range Invariant

Predictions must satisfy:

$$
y \in [y_{\min}, y_{\max}]
$$

Predictions outside this interval are projected back into the certified output range.

### 2. Epistemic Uncertainty Constraint

The epistemic uncertainty is compared against the calibrated conformal threshold:

$$
\sigma^2_{\text{epistemic}}(x)
\leq
\hat{q}_{\text{epistemic}}
$$

If the uncertainty exceeds the calibrated limit, the runtime safety mechanism can flag or constrain the prediction according to the implementation.

This provides an additional layer of protection against predictions made under conditions where the ensemble exhibits excessive model disagreement.

---

## Test-Time Adaptation

The TTA mechanism operates without requiring target labels during inference.

The core principle is to update feature-normalization statistics to accommodate the shifted input distribution.

Conceptually:

```text
Training Distribution
        │
        ▼
Fixed BatchNorm Statistics
        │
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

This enables the model to adapt to sensor calibration drift while preserving the unsupervised nature of test-time inference.

---

## Experimental Configuration

The reported experiment uses:

| Configuration              |      Value      |
| :------------------------- | :-------------: |
| Sensor drift               | $\delta = +1.5$ |
| Target conformal coverage  |      $95%$      |
| Conformal error rate       | $\alpha = 0.05$ |
| Calibrated epistemic bound |    `0.005192`   |
| Test samples               |      `400`      |
| Pre-TTA MSE                |     `0.7578`    |
| Post-TTA MSE               |     `0.0137`    |
| MSE reduction              |     `98.19%`    |
| Pre-TTA safety intercepts  |    `0 / 400`    |
| Post-TTA safety intercepts |    `0 / 400`    |

---

## Reproducibility

To reproduce the reported experiment:

```bash
git clone <repository-url>
cd Resilient-Deep-Ensemble-TTA

pip install torch numpy matplotlib

python main.py
```

The implementation is designed to run locally using standard Python execution environments, including:

* Python Terminal
* Windows Command Prompt
* PowerShell
* Python IDLE
* Compatible IDEs

---

## Research Context

This implementation explores the combination of three complementary approaches to reliable machine learning:

### Probabilistic Deep Ensembles

Deep ensembles provide multiple predictive models whose disagreement can be used as an empirical estimate of epistemic uncertainty.

### Conformal Prediction

Split conformal calibration provides a distribution-free mechanism for constructing uncertainty or verification thresholds under exchangeability assumptions.

### Test-Time Adaptation

TTA enables models to respond to distributional changes encountered during deployment without requiring labeled target-domain observations.

Together, these components form a layered architecture:

$$
\boxed{
\text{Adaptation}
+
\text{Uncertainty Estimation}
+
\text{Conformal Verification}
+
\text{Runtime Safety}
}
$$

The objective is not simply to improve predictive accuracy, but to combine **adaptation and prediction with explicit runtime safety mechanisms**.

---

## Important Statistical Note

The conformal guarantee should be interpreted under the assumptions required by the specific conformal procedure, particularly the relevant exchangeability or calibration assumptions.

The reported `95%` coverage is therefore a **nominal conformal target**, rather than an unconditional guarantee under arbitrary distribution shift.

Likewise, improved MSE under the simulated drift condition does not by itself establish robustness to every possible real-world sensor failure or distribution shift.

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

If you use this implementation in academic or research work, please cite the project and acknowledge the author.

```text
Ikechukwu Okechi Kamalu
Resilient Deep Ensemble with Conformal Safety Shield
& Test-Time Adaptation (TTA)
```

---

## Project Summary

This project provides an end-to-end framework for investigating **safe and adaptive probabilistic machine learning under sensor distribution shift**.

The architecture combines:

1. **Probabilistic deep ensembles** for predictive uncertainty.
2. **Epistemic/aleatoric uncertainty decomposition**.
3. **Split conformal calibration** for uncertainty-based runtime verification.
4. **Unsupervised test-time adaptation** for sensor drift.
5. **Runtime safety projection** for enforcing output invariants.
6. **Diagnostic visualization** for evaluating adaptation and uncertainty.

Under the reported simulated sensor drift of $\delta = +1.5$, the system reduced test MSE from `0.7578` to `0.0137`, corresponding to a **98.19% reduction**, while the runtime safety monitor recorded **0 safety intercepts across 400 test samples**.

The project therefore serves as a computational research framework for studying the intersection of **deep ensembles, uncertainty quantification, conformal prediction, test-time adaptation, runtime verification, and safe AI**.
