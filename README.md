# Resilient Deep Ensemble with Conformal Safety Shield & Test-Time Adaptation (TTA)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end implementation of a **probabilistic deep ensemble** integrated with **conformal calibration**, **runtime safety verification**, and **unsupervised Test-Time Adaptation (TTA)** under severe input sensor drift.

Developed as part of computational research into safe AI architectures, runtime verification, uncertainty quantification, and out-of-distribution (OOD) robustness.

---

## Key Features

### Probabilistic Deep Ensemble

An $M$-member neural network ensemble estimates a predictive mean $\mu_m(x)$ and aleatoric variance $\sigma_m^2(x)$ for each ensemble member using Gaussian Negative Log-Likelihood (NLL) loss.

### Uncertainty Decomposition

The ensemble separates predictive uncertainty into:

- **Epistemic uncertainty** — uncertainty arising from disagreement between ensemble members.
- **Aleatoric uncertainty** — uncertainty associated with inherent observation or process noise.

### Conformal Calibration

A split-conformal calibration procedure is used to obtain an empirical threshold for the selected epistemic-uncertainty non-conformity score.

The reported experiment uses:

$$\alpha = 0.05$$

corresponding to a nominal calibration level of:

$$1-\alpha = 0.95$$

### Online Test-Time Adaptation (TTA)

The system performs unsupervised feature-normalization re-centering during inference using dynamic BatchNorm statistics, without requiring target-domain labels.

### Runtime Safety Projection

A runtime safety layer monitors output-range invariants:

$$y \in [y_{\min}, y_{\max}]$$

together with an epistemic-uncertainty threshold. Predictions that violate the configured output range can be projected back into the permitted safety interval.

---

## Empirical Benchmark Results

The system was evaluated under simulated physical sensor calibration drift with:

$$\delta = +1.5$$

applied to the input process data.

| Metric | Pre-Adaptation (Fixed Training Stats) | Post-Adaptation (Dynamic TTA) | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Raw Test MSE** | `0.7578` | **`0.0137`** | **98.19% reduction** |
| **Safety Intercepts Triggered** | `0 / 400` | `0 / 400` | Satisfied invariants |
| **Calibrated Epistemic Bound** | `0.005192` | `0.005192` | $\alpha = 0.05$ |

### Benchmark Interpretation

Under the simulated sensor calibration drift, dynamic TTA reduced the raw test MSE from `0.7578` to `0.0137`.

The relative reduction in MSE is:

$$\frac{0.7578 - 0.0137}{0.7578} \times 100 \approx 98.19\%$$

At the same time, the runtime safety layer reported:

- **0/400 safety intercepts** before adaptation.
- **0/400 safety intercepts** after adaptation.
- A calibrated epistemic uncertainty threshold of **0.005192**.
- A nominal conformal calibration level of **95%**.

These results demonstrate the intended interaction between **distribution-shift adaptation**, **uncertainty estimation**, and **runtime safety verification**.

> **Important:** The reported MSE improvement is specific to the simulated drift experiment and should not be interpreted as evidence of robustness to arbitrary real-world sensor failures or distribution shifts.

---

## Mathematical Formulation

### 1. Ensemble Prediction

For an ensemble containing $M$ probabilistic neural networks, member $m$ produces:

$$\mu_m(x), \qquad \sigma_m^2(x), \qquad m = 1,\ldots,M$$

where:

- $\mu_m(x)$ is the predictive mean of ensemble member $m$.
- $\sigma_m^2(x)$ is its predicted aleatoric variance.

The ensemble predictive mean is:

$$\mu_{\text{ensemble}}(x) = \frac{1}{M} \sum_{m=1}^{M} \mu_m(x)$$

### 2. Epistemic Uncertainty

The epistemic variance represents disagreement between ensemble members:

$$\sigma^2_{\text{epistemic}}(x) = \frac{1}{M-1} \sum_{m=1}^{M} \left( \mu_m(x) - \mu_{\text{ensemble}}(x) \right)^2$$

A larger value indicates greater disagreement between the ensemble members.

### 3. Aleatoric Uncertainty

The ensemble-averaged aleatoric variance is:

$$\sigma^2_{\text{aleatoric}}(x) = \frac{1}{M} \sum_{m=1}^{M} \sigma_m^2(x)$$

### 4. Total Predictive Variance

The total predictive variance is decomposed as:

$$\sigma^2_{\text{total}}(x) = \sigma^2_{\text{epistemic}}(x) + \sigma^2_{\text{aleatoric}}(x)$$

Equivalently,

$$\sigma^2_{\text{total}}(x) = \frac{1}{M-1} \sum_{m=1}^{M} \left( \mu_m(x) - \mu_{\text{ensemble}}(x) \right)^2 + \frac{1}{M} \sum_{m=1}^{M} \sigma_m^2(x)$$

where:

- $\sigma^2_{\text{epistemic}}(x)$ represents model uncertainty.
- $\sigma^2_{\text{aleatoric}}(x)$ represents data or observation uncertainty.
- $\sigma^2_{\text{total}}(x)$ represents the combined predictive uncertainty.

---

## Conformal Calibration

Let the calibration dataset be:

$$D_{\text{calib}} = \{ (x_i,y_i) \}_{i=1}^{N_{\text{calib}}}$$

and let the target error rate be:

$$\alpha = 0.05$$

For the implementation's epistemic-uncertainty non-conformity score, define:

$$s_i = \sigma^2_{\text{epistemic}}(x_i)$$

The calibrated threshold is obtained from the empirical calibration distribution:

$$\hat{q} = \text{Quantile}\left( \{ s_i \}_{i=1}^{N_{\text{calib}}}, \frac{\lceil (N_{\text{calib}}+1)(1-\alpha) \rceil}{N_{\text{calib}}} \right)$$

For the reported experiment:

$$\hat{q}_{\text{epistemic}} = 0.005192$$

with:

$$\alpha = 0.05$$

and therefore:

$$1-\alpha = 0.95$$

The resulting value is used as the runtime epistemic-uncertainty threshold by the safety shield.

> **Statistical qualification:** The conformal calibration level is associated with the chosen non-conformity score under the assumptions of the conformal procedure, particularly the relevant calibration/exchangeability assumptions. Calibrating epistemic uncertainty alone should not be interpreted as automatically providing a 95% prediction-interval coverage guarantee for $y$.

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
                  │  Conformal Calibration   │
                  │                          │
                  │ σ²_epistemic ≤ q̂          │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Runtime Safety Projection│
                  │                          │
                  │ y ∈ [ymin, ymax]         │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │      Safe Prediction     │
                  └──────────────────────────┘

**## Repository Structure**

```text
Resilient-Deep-Ensemble-TTA/
│
├── main.py              # Main execution script
├── README.md            # Repository documentation
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

### Run the Experiment

Navigate to the repository directory and execute:

```bash
python main.py
```

The program will:

1. Train/evaluate the probabilistic ensemble.
2. Estimate predictive uncertainty.
3. Perform conformal calibration.
4. Apply simulated sensor drift.
5. Evaluate pre-TTA predictions.
6. Perform unsupervised TTA.
7. Evaluate post-TTA predictions.
8. Apply the runtime safety layer.
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
* Ground-truth process values under sensor drift.
* Runtime safety boundaries.

This visualization demonstrates the effect of TTA on predictions under the simulated input distribution shift.

### Right Panel — Epistemic Uncertainty & Conformal Threshold

The right panel displays:

* Epistemic uncertainty.
* The calibrated conformal threshold.
* The relationship between runtime uncertainty and the configured verification boundary.

The reported threshold is:

$$
\hat{q}_{\text{epistemic}}
==========================

0.005192
$$

---

## Safety Mechanism

The runtime safety layer combines two primary checks.

### 1. Output Range Invariant

Predictions are required to satisfy:

$$
y \in [y_{\min}, y_{\max}]
$$

Predictions outside this configured interval can be projected back into the permitted output range.

### 2. Epistemic Uncertainty Constraint

The ensemble epistemic uncertainty is compared against the calibrated threshold:

$$
\sigma^2_{\text{epistemic}}(x)
\leq
\hat{q}_{\text{epistemic}}
$$

If the uncertainty exceeds the configured threshold, the runtime safety mechanism can flag or constrain the prediction according to the implementation.

This provides an additional runtime protection mechanism for conditions in which the ensemble exhibits excessive model disagreement.

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

This enables the model to adapt to simulated sensor calibration drift while preserving the unsupervised nature of test-time inference.

---

## Experimental Configuration

The reported experiment uses:

| Configuration              |      Value      |
| :------------------------- | :-------------: |
| Sensor drift               | $\delta = +1.5$ |
| Target conformal level     |      $95%$      |
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

Clone the repository:

```bash
git clone https://github.com/ikechukwuKamalu8/Resilient-Deep-Ensemble-TTA.git
cd Resilient-Deep-Ensemble-TTA
```

Install the dependencies:

```bash
pip install torch numpy matplotlib
```

Run the experiment:

```bash
python main.py
```

The implementation is designed to run locally using standard Python execution environments, including:

* Python Terminal
* Windows Command Prompt
* PowerShell
* Python IDLE
* Compatible Python IDEs

---

## Research Context

This implementation explores the combination of several complementary approaches to reliable machine learning.

### Probabilistic Deep Ensembles

Deep ensembles provide multiple predictive models whose disagreement can be used as an empirical estimate of epistemic uncertainty.

### Uncertainty Quantification

The probabilistic ensemble separates uncertainty into epistemic and aleatoric components, providing information about both model disagreement and inherent observation noise.

### Conformal Calibration

Split conformal calibration provides a distribution-free calibration framework for the selected non-conformity score under the assumptions required by the conformal procedure.

### Test-Time Adaptation

TTA enables models to respond to distributional changes encountered during deployment without requiring labeled target-domain observations.

### Runtime Safety Verification

A runtime safety layer monitors configured uncertainty and output constraints and can intervene when those constraints are violated.

Together, these components form a layered architecture:

$$
\boxed{
\text{Adaptation}
+
\text{Uncertainty Estimation}
+
\text{Conformal Calibration}
+
\text{Runtime Safety}
}
$$

The objective is not simply to improve predictive accuracy, but to combine **adaptation, probabilistic prediction, uncertainty estimation, calibration, and explicit runtime safety mechanisms**.

---

## Important Statistical Note

The reported `95%` value should be interpreted as a **nominal conformal calibration level** for the selected non-conformity score.

The validity of the corresponding conformal calibration depends on the assumptions of the specific conformal procedure, including the relevant exchangeability or calibration assumptions.

In particular:

* The calibration threshold is estimated from the calibration dataset.
* The threshold is applied to the selected epistemic-uncertainty score.
* The `95%` level should not automatically be interpreted as unconditional 95% coverage for the response variable $y$.
* Test-time adaptation introduces distributional changes that may affect the assumptions underlying calibration.
* Improved MSE under the simulated drift condition does not establish robustness to arbitrary real-world sensor failures or unseen distribution shifts.

Therefore, the reported results should be interpreted as an **empirical demonstration of the proposed architecture under the specified experimental conditions**.

---

## Limitations

The current implementation should be understood as a research prototype rather than a complete production safety certification system.

Important limitations include:

1. **Simulated sensor drift:** The reported $\delta = +1.5$ shift represents a controlled experimental condition.
2. **Finite calibration data:** The conformal threshold depends on the available calibration sample.
3. **Distribution shift:** Standard conformal validity assumptions may not hold under arbitrary deployment-time distribution shift.
4. **Epistemic thresholding:** An epistemic-uncertainty threshold is a runtime diagnostic/verification criterion and is not, by itself, equivalent to a calibrated prediction interval.
5. **Safety projection:** Output projection enforces the configured numerical range but does not prove that the underlying physical system is safe in every operating condition.
6. **TTA stability:** Test-time adaptation can itself introduce failure modes if adaptation statistics become unreliable or contaminated by atypical inputs.

These limitations provide opportunities for future research into stronger distribution-shift guarantees, adaptive conformal methods, robust TTA, and formal runtime verification.

---

## Future Research Directions

Potential extensions include:

* Adaptive conformal prediction under distribution shift.
* Conformal risk control for deployment-time safety constraints.
* Robust test-time adaptation under adversarial or corrupted inputs.
* Online uncertainty recalibration.
* Distribution-shift detection before TTA activation.
* Formal verification of the runtime safety projection.
* Multi-sensor fusion and heterogeneous sensor drift.
* Probabilistic forecasting for multivariate process systems.
* Sequential conformal methods for time-series deployment.
* Certified adaptation and safety monitoring for autonomous systems.

---

## License

This project is released under the **MIT License**.

See the repository license file for the complete license terms.

---

## Author & Citation

**Ikechukwu Okechi Kamalu**

MSc Machine Learning Researcher & Biostatistician

**Email:** `ikechukwukamalu8@gmail.com`

**GitHub:** `github.com/ikechukwuKamalu8`

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

This project provides an end-to-end computational framework for investigating **safe and adaptive probabilistic machine learning under sensor distribution shift**.

The architecture combines:

1. **Probabilistic deep ensembles** for predictive uncertainty.
2. **Epistemic/aleatoric uncertainty decomposition**.
3. **Conformal calibration** for uncertainty-based runtime verification.
4. **Unsupervised test-time adaptation** for sensor drift.
5. **Runtime safety projection** for enforcing output invariants.
6. **Diagnostic visualization** for evaluating adaptation and uncertainty.

Under the reported simulated sensor drift of:

$$
\delta = +1.5
$$

the system reduced test MSE from `0.7578` to `0.0137`, corresponding to a **98.19% reduction**, while the runtime safety monitor recorded **0 safety intercepts across 400 test samples**.

The project therefore serves as a computational research framework for studying the intersection of:

**deep ensembles, uncertainty quantification, conformal calibration, test-time adaptation, runtime verification, and safe AI.**
