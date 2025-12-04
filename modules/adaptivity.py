import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


plt.rcParams['figure.dpi'] = 300

class QuantileRegressor(nn.Module):
    """
    A Neural Network that predicts conditional quantiles.
    Outputs 3 values corresponding to quantiles: [alpha/2, 0.5, 1 - alpha/2].
    """
    def __init__(self, hidden_dim=64):
        super(QuantileRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # Outputs: Lower, Median, Upper
        )

    def forward(self, x):
        return self.net(x)

class AdaptivityExperiment:
    """
    Experiment 2: Simulated Heteroscedastic Data.
    Demonstrates 'Adaptivity': How CP handles varying noise levels vs Fixed Width methods.
    """

    def __init__(self, n_samples=20000, alpha=0.1, save_dir='./results/adaptivity', gpu_id=0):
        """
        :param n_samples: Total number of data points.
        :param alpha: Target error rate (0.1 means 90% coverage).
        """
        self.n_samples = n_samples
        self.alpha = alpha
        self.save_dir = save_dir
        
        # --- GPU Setup ---
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{gpu_id}")
            print(f"[Init] Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            print("[Init] GPU not available. Using CPU.")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Set seeds
        self._set_seeds()
        self.model = None

    def _set_seeds(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)

    def generate_data(self):
        """
        Generates synthetic data: y = sin(x) + epsilon
        epsilon ~ N(0, |x| * scale) -> Noise increases with x (Heteroscedasticity).
        """
        print("[1/5] Generating synthetic heteroscedastic data...")
        # Generate x in range [-4, 4]
        X = np.random.uniform(-4, 4, self.n_samples)
        
        # Heteroscedastic noise: Noise std deviation depends on |x|
        # Adding a small constant 0.1 so noise is never exactly 0
        noise_std = np.abs(X) * 0.5 + 0.2
        noise = np.random.normal(0, noise_std, self.n_samples)
        
        y = np.sin(X) + noise
        
        # Sort X for better plotting later
        sort_idx = np.argsort(X)
        X, y = X[sort_idx], y[sort_idx]

        # Split: Train (40%), Calibration (30%), Test (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
        X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Convert to Tensors
        self.X_train = torch.FloatTensor(X_train).unsqueeze(1)
        self.y_train = torch.FloatTensor(y_train).unsqueeze(1)
        self.X_cal = torch.FloatTensor(X_cal).unsqueeze(1)
        self.y_cal = torch.FloatTensor(y_cal).unsqueeze(1)
        self.X_test = torch.FloatTensor(X_test).unsqueeze(1)
        self.y_test = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Keep numpy versions for plotting
        self.X_test_np = X_test
        self.y_test_np = y_test

        print(f"      Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")

    def pinball_loss(self, preds, targets):
        """
        Calculates Pinball Loss (Quantile Loss) for multi-quantile prediction.
        Quantiles used: [alpha/2, 0.5, 1 - alpha/2]
        """
        quantiles = torch.tensor([self.alpha/2, 0.5, 1 - self.alpha/2], device=self.device)
        
        # preds shape: [batch, 3], targets shape: [batch, 1]
        # Broadcast targets to match preds
        errors = targets - preds
        
        # Pinball loss logic: max(q * error, (q - 1) * error)
        loss = torch.max((quantiles - 1) * errors, quantiles * errors)
        return torch.mean(loss)

    def train_quantile_model(self):
        """Trains the Quantile Regressor using Pinball Loss."""
        print(f"[2/5] Training Quantile Regression Model on {self.device}...")
        self.model = QuantileRegressor().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        X_train_dev = self.X_train.to(self.device)
        y_train_dev = self.y_train.to(self.device)
        
        epochs = 1000
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X_train_dev)
            loss = self.pinball_loss(preds, y_train_dev)
            loss.backward()
            optimizer.step()
            
        print("      Training complete.")

    def run_methods(self):
        """
        Implements:
        1. Fixed Width Baseline (using median prediction + global constant width)
        2. CQR (Conformalized Quantile Regression)
        """
        print("[3/5] Computing prediction intervals...")
        self.model.eval()
        
        # --- Inference on Calibration and Test ---
        with torch.no_grad():
            # Output: [Lower, Median, Upper]
            preds_cal = self.model(self.X_cal.to(self.device)).cpu()
            preds_test = self.model(self.X_test.to(self.device)).cpu()

        # Extract specific quantiles
        # 0: lower (alpha/2), 1: median, 2: upper (1-alpha/2)
        cal_low, cal_med, cal_high = preds_cal[:, 0], preds_cal[:, 1], preds_cal[:, 2]
        test_low, test_med, test_high = preds_test[:, 0], preds_test[:, 1], preds_test[:, 2]
        
        y_cal = self.y_cal.squeeze()
        y_test = self.y_test.squeeze()

        # ==========================================
        # Method A: Fixed Width Baseline
        # ==========================================
        # Logic: Use the Median prediction, but assume constant variance.
        # We calculate the 95th percentile of absolute residuals on calibration set
        # to construct a "Globally Valid" fixed width.
        abs_residuals = torch.abs(y_cal - cal_med)
        
        # Find the constant C such that Median +/- C covers (1-alpha)
        # We use the empirical quantile of residuals to be fair
        n = len(y_cal)
        q_level = np.ceil((n+1)*(1-self.alpha))/n
        fixed_width = torch.quantile(abs_residuals, q_level).item()
        
        self.base_lower = test_med - fixed_width
        self.base_upper = test_med + fixed_width
        
        # ==========================================
        # Method B: CQR (Adaptive)
        # ==========================================
        # Logic: Use learned quantiles (low, high) and calibrate them.
        # Score s_i = max(q_low - y, y - q_high)
        scores = torch.max(cal_low - y_cal, y_cal - cal_high)
        
        # Compute correction factor q_hat
        q_hat = torch.quantile(scores, q_level).item()
        
        # Construct Adaptive Intervals
        self.cqr_lower = test_low - q_hat
        self.cqr_upper = test_high + q_hat
        
        # --- Metrics Calculation ---
        cov_base = ((y_test >= self.base_lower) & (y_test <= self.base_upper)).float().mean().item()
        cov_cqr = ((y_test >= self.cqr_lower) & (y_test <= self.cqr_upper)).float().mean().item()
        
        width_base = (self.base_upper - self.base_lower).mean().item()
        width_cqr = (self.cqr_upper - self.cqr_lower).mean().item()

        print("="*40)
        print("          RESULTS SUMMARY")
        print("="*40)
        print(f"Target Coverage: {1-self.alpha:.1%}")
        print(f"[Method A: Fixed]    Coverage: {cov_base:.1%} | Avg Width: {width_base:.4f}")
        print(f"[Method B: Adaptive] Coverage: {cov_cqr:.1%} | Avg Width: {width_cqr:.4f}")
        print("="*40)
        
        self.results = {
            'base_cov': cov_base, 'cqr_cov': cov_cqr,
            'base_width': width_base, 'cqr_width': width_cqr
        }

    def plot_comparison(self):
        """
        Visualizes the 'Tube' vs 'Trumpet' shape.
        """
        print("[4/5] Generating comparison plots...")
        
        # Sort X for proper fill_between plotting
        sort_idx = np.argsort(self.X_test_np)
        X_sorted = self.X_test_np[sort_idx]
        y_sorted = self.y_test_np[sort_idx]
        
        # Helper to convert tensor to sorted numpy
        def to_sorted_np(tensor):
            return tensor.numpy()[sort_idx]

        base_low = to_sorted_np(self.base_lower)
        base_high = to_sorted_np(self.base_upper)
        cqr_low = to_sorted_np(self.cqr_lower)
        cqr_high = to_sorted_np(self.cqr_upper)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        # --- Plot 1: Fixed Width (Baseline) ---
        ax = axes[0]
        # Plot data points
        ax.scatter(X_sorted, y_sorted, s=10, alpha=0.3, color='gray', label='Test Data')
        # Plot Bands
        ax.fill_between(X_sorted, base_low, base_high, color='#e74c3c', alpha=0.3, label='Prediction Band')
        # Plot Mean Prediction (approx)
        # ax.plot(X_sorted, (base_low+base_high)/2, color='#c0392b', linestyle='--', linewidth=2)
        
        ax.set_title(f"Fixed Width (Baseline)\nCoverage: {self.results['base_cov']:.1%}, Avg Width: {self.results['base_width']:.2f}", fontsize=14)
        ax.set_xlabel("Input X")
        ax.set_ylabel("Target Y")
        ax.legend(loc='upper left')

        # --- Plot 2: Adaptive CQR ---
        ax = axes[1]
        # Plot data points
        ax.scatter(X_sorted, y_sorted, s=10, alpha=0.3, color='gray', label='Test Data')
        # Plot Bands
        ax.fill_between(X_sorted, cqr_low, cqr_high, color='#2ecc71', alpha=0.3, label='Prediction Band (Adaptive)')
        
        ax.set_title(f"Conformalized Quantile Regression (CQR)\nCoverage: {self.results['cqr_cov']:.1%}, Avg Width: {self.results['cqr_width']:.2f}", fontsize=14)
        ax.set_xlabel("Input X")
        ax.legend(loc='upper left')
        
        save_path = os.path.join(self.save_dir, 'adaptivity_viz.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"      Comparison plot saved to {save_path}")
        plt.close()

    def run(self):
        print("="*50)
        print("Starting Experiment: Visualizing Adaptivity")
        print("="*50)
        self.generate_data()
        self.train_quantile_model()
        self.run_methods()
        self.plot_comparison()
        print("\nExperiment Completed.")

if __name__ == "__main__":
    exp = AdaptivityExperiment(alpha=0.1, gpu_id=0)
    exp.run()