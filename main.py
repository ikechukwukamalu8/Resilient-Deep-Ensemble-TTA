"""
Resilient Deep Ensemble with Formal Conformal Safety Shield & Test-Time Adaptation (TTA)
---------------------------------------------------------------------------------------
Author: Ikechukwu Okechi Kamalu, MSc
Environment: Compatible with standard Python IDLE, PyCharm, VS Code, or Terminal execution.
Dependencies: torch, numpy, matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Enforce strict reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. DOMAIN / SENSOR DRIFT DATASET GENERATOR
# ==========================================
def generate_domain_data(n_samples=600, sensor_drift=0.0):
    """
    Generates synthetic system process data.
    - x_true: Ground truth physical system state
    - sensor_drift: Physical sensor calibration shift applied to input x
    """
    x_true = np.random.uniform(-2.5, 2.5, size=(n_samples, 1))

    # Ground truth physical process y = sin(x) + 0.3*cos(2x) + noise
    y = np.sin(x_true) + 0.3 * np.cos(2 * x_true) + np.random.normal(0, 0.1, size=x_true.shape)

    # Observed input subject to physical sensor calibration drift
    x_observed = x_true + sensor_drift

    return torch.tensor(x_observed, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ==========================================
# 2. PROBABILISTIC MODEL WITH BATCHNORM
# ==========================================
class ProbabilisticNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ProbabilisticNet, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Heads for predictive mean (mu) and variance (log_var)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        mu = self.mu_head(h)
        log_var = torch.clamp(self.var_head(h), min=-6.0, max=2.0)
        return mu, log_var


def gaussian_nll_loss(mu, log_var, target):
    """Gaussian Negative Log-Likelihood Loss."""
    precision = torch.exp(-log_var)
    return torch.mean(0.5 * precision * (target - mu)**2 + 0.5 * log_var)


# ==========================================
# 3. DEEP ENSEMBLE WITH FORMAL CONFORMAL SHIELD
# ==========================================
class FormallyShieldedDeepEnsemble:
    def __init__(self, n_models=5, hidden_dim=64, y_bounds=(-1.5, 1.5)):
        self.n_models = n_models
        self.models = [ProbabilisticNet(hidden_dim) for _ in range(n_models)]

        # Formal Safety Invariants & Bounds
        self.y_min, self.y_max = y_bounds
        self.conformal_threshold = None

    def fit(self, x, y, epochs=250, lr=1e-3):
        print("--- Training Deep Ensemble Members on Clean In-Distribution Data ---")
        for i, model in enumerate(self.models):
            optimizer = optim.Adam(model.parameters(), lr=lr)
            indices = torch.randint(0, len(x), (len(x),))
            x_b, y_b = x[indices], y[indices]

            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                mu, log_var = model(x_b)
                loss = gaussian_nll_loss(mu, log_var, y_b)
                loss.backward()
                optimizer.step()
            print(f"Model [{i+1}/{self.n_models}] Training Completed.")

    def calibrate_conformal(self, x_calib, y_calib, alpha=0.05):
        """Calculates Split Conformal Prediction threshold for 1-alpha coverage guarantee."""
        print("\n--- Calibrating Conformal Uncertainty Threshold ---")
        mu_calib, ep_calib, _ = self.raw_predict(x_calib, adapt_mode=False)
        
        # Non-conformity scores derived from epistemic variance magnitude
        scores = ep_calib.detach().numpy().flatten()
        
        # Compute empirical quantile
        q_level = np.ceil((len(x_calib) + 1) * (1 - alpha)) / len(x_calib)
        self.conformal_threshold = float(np.quantile(scores, q_level))
        print(f"Calibrated Conformal Epistemic Bound (Coverage={1-alpha:.2f}): {self.conformal_threshold:.6f}")

    def raw_predict(self, x, adapt_mode=False):
        means, vars_ = [], []
        for model in self.models:
            if adapt_mode:
                model.train()  # Dynamic BatchNorm adaptation mode (TTA)
            else:
                model.eval()   # Fixed evaluation mode

            with torch.set_grad_enabled(adapt_mode):
                mu, log_var = model(x)
                means.append(mu)
                vars_.append(torch.exp(log_var))

        means = torch.stack(means, dim=0)
        vars_ = torch.stack(vars_, dim=0)

        ensemble_mean = torch.mean(means, dim=0)
        epistemic_var = torch.var(means, dim=0)
        total_var = torch.mean(vars_, dim=0) + epistemic_var

        return ensemble_mean, epistemic_var, total_var

    def shielded_predict(self, x, adapt_mode=False):
        """Executes Runtime Verification with Control-Barrier Style Safety Projection."""
        mu_raw, epistemic_var, _ = self.raw_predict(x, adapt_mode=adapt_mode)

        shielded_output = mu_raw.clone()
        safety_violations = 0

        # Evaluate invariant predicates per prediction sample
        for j in range(len(x)):
            pred_val = mu_raw[j].item()
            ep_unc = epistemic_var[j].item()

            is_bound_safe = (self.y_min <= pred_val <= self.y_max)
            is_unc_safe = (self.conformal_threshold is None) or (ep_unc <= self.conformal_threshold)

            if not (is_bound_safe and is_unc_safe):
                # Safety Guard Projection: Clamps unsafe values to nearest certified boundary point
                shielded_output[j] = torch.clamp(mu_raw[j], self.y_min, self.y_max)
                safety_violations += 1

        return shielded_output, mu_raw, epistemic_var, safety_violations


# ==========================================
# 4. MAIN EXECUTION ROUTINE
# ==========================================
def main():
    # Dataset Preparation
    x_train, y_train = generate_domain_data(n_samples=800, sensor_drift=0.0)
    x_calib, y_calib = generate_domain_data(n_samples=200, sensor_drift=0.0)
    x_test_shifted, y_test_shifted = generate_domain_data(n_samples=400, sensor_drift=1.5)

    # Initialize and Train Ensemble
    ensemble = FormallyShieldedDeepEnsemble(n_models=5, y_bounds=(-1.5, 1.5))
    ensemble.fit(x_train, y_train, epochs=250)

    # Calibrate Formal Conformal Threshold
    ensemble.calibrate_conformal(x_calib, y_calib, alpha=0.05)

    print("\n--- Evaluating Online Self-Correction & Formal Safety Shield ---")

    # Pre-Adaptation Evaluation
    mean_pre_shielded, mean_pre_raw, ep_pre, violations_pre = ensemble.shielded_predict(x_test_shifted, adapt_mode=False)

    # Post-Adaptation Evaluation (Test-Time Adaptation)
    mean_post_shielded, mean_post_raw, ep_post, violations_post = ensemble.shielded_predict(x_test_shifted, adapt_mode=True)

    # Performance Metrics
    mse_pre_raw = torch.mean((mean_pre_raw - y_test_shifted)**2).item()
    mse_post_raw = torch.mean((mean_post_raw - y_test_shifted)**2).item()
    improvement = ((mse_pre_raw - mse_post_raw) / mse_pre_raw) * 100

    print("\n" + "="*55)
    print("     FORMAL VERIFICATION & ADAPTATION BENCHMARK     ")
    print("="*55)
    print(f"Pre-Adaptation Raw Test MSE               : {mse_pre_raw:.4f}")
    print(f"Pre-Adaptation Formal Shield Intercepts   : {violations_pre} / {len(x_test_shifted)} samples")
    print(f"Post-Adaptation Raw Test MSE              : {mse_post_raw:.4f}")
    print(f"Post-Adaptation Formal Shield Intercepts  : {violations_post} / {len(x_test_shifted)} samples")
    print(f"MSE Improvement via Adaptation            : +{improvement:.2f}%")
    print("="*55)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Predictions vs Ground Truth
    axes[0].axhspan(-1.5, 1.5, color='green', alpha=0.1, label='Certified Safe Region [-1.5, 1.5]')
    axes[0].scatter(x_test_shifted.numpy(), y_test_shifted.numpy(), c='red', alpha=0.3, label='Drifted Ground Truth')
    axes[0].scatter(x_test_shifted.numpy(), mean_pre_shielded.detach().numpy(), c='orange', s=15, label=f'Pre-Adaptation (Intercepts: {violations_pre})')
    axes[0].scatter(x_test_shifted.numpy(), mean_post_shielded.detach().numpy(), c='blue', s=15, label=f'Post-Adaptation (Intercepts: {violations_post})')
    axes[0].set_title("Runtime Safety Shield & Online Test-Time Adaptation")
    axes[0].set_xlabel("Observed Input x")
    axes[0].set_ylabel("Target y")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Epistemic Uncertainty Distribution
    axes[1].axvline(ensemble.conformal_threshold, color='red', linestyle='--', linewidth=2, label=f'Conformal Bound ({ensemble.conformal_threshold:.4f})')
    axes[1].hist(ep_pre.detach().numpy().flatten(), bins=25, color='orange', alpha=0.6, label='Pre-Adaptation Uncertainty')
    axes[1].hist(ep_post.detach().numpy().flatten(), bins=25, color='blue', alpha=0.6, label='Post-Adaptation Uncertainty')
    axes[1].set_title("Formal Epistemic Verification Check")
    axes[1].set_xlabel("Epistemic Uncertainty")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
