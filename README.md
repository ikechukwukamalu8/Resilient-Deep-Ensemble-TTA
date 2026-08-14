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

$$D_{\text{calib}} = \left\{ (x_i,y_i) \right\}_{i=1}^{N_{\text{calib}}}$$

and let the target error rate be:

$$\alpha = 0.05$$

For the implementation's epistemic-uncertainty non-conformity score, define:

$$s_i = \sigma^2_{\text{epistemic}}(x_i)$$

The calibrated threshold is obtained from the empirical calibration distribution:

$$\hat{q} = \operatorname{Quantile} \left( \left\{ s_i \right\}_{i=1}^{N_{\text{calib}}}, \frac{\lceil (N_{\text{calib}}+1)(1-\alpha) \rceil}{N_{\text{calib}}} \right)$$

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
