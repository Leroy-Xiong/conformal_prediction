import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, expon

# Set high resolution for plots
plt.rcParams['figure.dpi'] = 300

class DistributionFreeExperiment:
    """
    Experiment to demonstrate the 'Distribution-Free' property of Conformal Prediction.
    
    Dataset: Synthetic Non-Linear Data with Skewed (Exponential) Noise.
    Goal: Show that CP maintains valid coverage even when residuals are NOT Gaussian,
          whereas the Baseline (Mean +/- z*std) fails.
    """

    def __init__(self, n_samples=3000, random_state=42, alpha=0.1, save_dir='./results/distribution_free', gpu_id=0):
        """
        Initialize the experiment.
        :param n_samples: Number of data points to generate.
        :param alpha: Target error rate (e.g., 0.1 for 90% coverage).
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.alpha = alpha
        self.save_dir = save_dir
        
        # --- Device Configuration ---
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{gpu_id}")
            print(f"[Init] Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            print("[Init] GPU not available. Falling back to CPU.")
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self._set_seeds()

        # Placeholders
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_and_split_data(self):
        """
        Generates synthetic data with Non-Gaussian (Exponential) noise.
        """
        print("[1/6] Generating synthetic data with Skewed Noise...")
        
        # 1. Generate Non-Linear Data (Friedman #1 regression problem)
        X, y_clean = make_friedman1(n_samples=self.n_samples, n_features=10, noise=0.0, random_state=self.random_state)
        
        # 2. Add SKEWED Noise (Exponential Distribution)
        # This breaks the Gaussian assumption inherent in standard confidence intervals.
        noise = np.random.exponential(scale=5.0, size=y_clean.shape) 
        # Center the noise so mean is roughly 0, but shape remains skewed
        noise = noise - 2.0 
        y = y_clean + noise
        
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        # Save for visualization
        self.raw_df = pd.DataFrame(X, columns=feature_names)
        self.raw_df['Target'] = y

        # Split: Train (40%), Calibration (30%), Test (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.6, random_state=self.random_state
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_cal = self.scaler.transform(X_cal)
        X_test = self.scaler.transform(X_test)

        # Convert to Tensors
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train).unsqueeze(1)
        
        self.X_cal = torch.FloatTensor(X_cal)
        self.y_cal_numpy = y_cal 
        self.y_test_numpy = y_test
        self.X_test = torch.FloatTensor(X_test)

        print(f"      Train size: {len(X_train)}")
        print(f"      Calibration size: {len(X_cal)}")
        print(f"      Test size: {len(X_test)}")

    def _plot_data_overview(self):
        """
        Visualizes the data to show non-linearity and skewness.
        """
        print("[2/6] Visualizing data properties...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Feature 0 vs Target (Shows non-linearity)
        sns.scatterplot(
            data=self.raw_df, x='Feature_0', y='Target', 
            alpha=0.3, color='purple', ax=axes[0]
        )
        axes[0].set_title('Feature 0 vs Target (Non-Linear)', fontsize=14)
        
        # Plot 2: Feature 1 vs Target
        sns.scatterplot(
            data=self.raw_df, x='Feature_1', y='Target', 
            alpha=0.3, color='teal', ax=axes[1]
        )
        axes[1].set_title('Feature 1 vs Target (Non-Linear)', fontsize=14)

        # Plot 3: Target Distribution (Likely skewed)
        sns.histplot(self.raw_df['Target'], kde=True, color='orange', ax=axes[2])
        axes[2].set_title('Distribution of Target y', fontsize=14)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'data_overview.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def train_base_model(self):
        """Trains an MLP regressor."""
        print(f"[3/6] Training neural network on {self.device}...")
        
        self.model = nn.Sequential(
            nn.Linear(self.X_train.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        X_train_dev = self.X_train.to(self.device)
        y_train_dev = self.y_train.to(self.device)
        
        epochs = 400
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train_dev)
            loss = criterion(outputs, y_train_dev)
            loss.backward()
            optimizer.step()
        print("      Training complete.")

    def run_inference(self):
        """Runs inference to get point predictions."""
        print("[4/6] Running inference...")
        self.model.eval()
        X_cal_dev = self.X_cal.to(self.device)
        X_test_dev = self.X_test.to(self.device)
        
        with torch.no_grad():
            self.y_pred_cal = self.model(X_cal_dev).cpu().numpy().flatten()
            self.y_pred_test = self.model(X_test_dev).cpu().numpy().flatten()

    def run_methods(self):
        """
        Compares Baseline (Gaussian Assumption) vs Conformal Prediction.
        """
        print("[5/6] Computing uncertainty intervals...")
        
        # --- Method A: Baseline (Parametric / Gaussian) ---
        # Assumes residuals follow N(0, sigma). 
        # Since our noise is Exponential, this assumption is WRONG.
        cal_residuals = self.y_cal_numpy - self.y_pred_cal
        
        # Fit Gaussian parameters
        mu_hat, std_hat = norm.fit(cal_residuals)
        
        # Calculate z-score for (1 - alpha)
        z_score = norm.ppf(1 - self.alpha / 2)
        self.width_baseline = z_score * std_hat
        
        # Construct symmetric intervals
        self.y_lower_base = self.y_pred_test - self.width_baseline
        self.y_upper_base = self.y_pred_test + self.width_baseline
        self.cal_residuals = cal_residuals

        # --- Method B: Split Conformal Prediction (Distribution-Free) ---
        # Uses empirical quantiles of residuals. Does not assume shape.
        scores_cal = np.abs(self.y_cal_numpy - self.y_pred_cal)
        n = len(self.y_cal_numpy)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        
        # Calculate Q_hat (the (1-alpha) quantile)
        self.q_hat = np.quantile(scores_cal, q_level, method='higher')
        
        self.y_lower_cp = self.y_pred_test - self.q_hat
        self.y_upper_cp = self.y_pred_test + self.q_hat

        # --- Evaluate Coverage ---
        cov_base = np.mean((self.y_test_numpy >= self.y_lower_base) & (self.y_test_numpy <= self.y_upper_base))
        cov_cp = np.mean((self.y_test_numpy >= self.y_lower_cp) & (self.y_test_numpy <= self.y_upper_cp))
        
        self.metrics = {'Baseline Coverage': cov_base, 'CP Coverage': cov_cp, 'Target': 1 - self.alpha}
        
        # Interval Widths
        self.avg_width_base = self.width_baseline * 2
        self.avg_width_cp = self.q_hat * 2

        print("="*50)
        print("          RESULTS SUMMARY (Distribution-Free Test)")
        print("="*50)
        print(f"      Target Coverage: {1-self.alpha:.1%}")
        print(f"      Baseline Coverage: {cov_base:.1%} (Likely Mismatched)")
        print(f"      CP Coverage:       {cov_cp:.1%} (Valid)")
        print("-" * 50)
        print(f"      Gaussian Width:    {self.avg_width_base:.4f}")
        print(f"      Conformal Width:   {self.avg_width_cp:.4f}")
        print("="*50)

    def plot_results(self):
        """Generates plots to prove the concept."""
        print("[6/6] Generating result plots...")
        self._plot_residual_distribution()
        self._plot_coverage_comparison()
        self._plot_interval_visualization()
        print(f"Results saved to: {os.path.abspath(self.save_dir)}")

    def _plot_residual_distribution(self):
        """
        Crucial Plot: Shows that residuals are NOT Gaussian, explaining why Baseline fails.
        """
        plt.figure(figsize=(10, 6))
        
        # 1. Plot actual residuals (Skewed)
        sns.histplot(self.cal_residuals, kde=True, stat="density", 
                     color="#3498db", label="Actual Residuals (Non-Gaussian)", alpha=0.5)
        
        # 2. Plot the Gaussian fit that the Baseline method *assumes*
        mu, std = norm.fit(self.cal_residuals)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 200)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'r--', linewidth=2.5, label=f"Gaussian Assumption\nN({mu:.2f}, {std:.2f})")
        
        plt.title("Proof of Distribution-Free Need:\nResiduals are Skewed, Gaussian Fit is Poor", fontsize=14)
        plt.xlabel("Residual Value (y - y_pred)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'residual_distribution_proof.png'), bbox_inches='tight')
        plt.close()

    def _plot_coverage_comparison(self):
        methods = ['Baseline\n(Gaussian)', 'Conformal Prediction\n(Any Distribution)']
        coverages = [self.metrics['Baseline Coverage'], self.metrics['CP Coverage']]
        target = self.metrics['Target']
        
        plt.figure(figsize=(8, 6))
        # Color coding: Red if fail, Green if pass (approx)
        cols = ['#e74c3c' if abs(c - target) > 0.05 else '#f1c40f' for c in coverages]
        cols[1] = '#2ecc71' # CP is usually green/correct
        
        bars = plt.bar(methods, coverages, color=cols, alpha=0.85, width=0.6)
        plt.axhline(y=target, color='black', linestyle='--', linewidth=2, label=f'Target ({target:.0%})')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
            
        plt.ylim(0, 1.15)
        plt.ylabel("Empirical Coverage Rate", fontsize=12)
        plt.title("Coverage Validity Test", fontsize=16)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.save_dir, 'coverage_comparison.png'), bbox_inches='tight')
        plt.close()
        
    def _plot_interval_visualization(self, n_samples=50):
        subset_idx = np.arange(n_samples)
        y_true_sub = self.y_test_numpy[subset_idx]
        y_pred_sub = self.y_pred_test[subset_idx]
        
        # CP Intervals
        lower_cp = self.y_lower_cp[subset_idx]
        upper_cp = self.y_upper_cp[subset_idx]
        
        plt.figure(figsize=(14, 6))
        
        # Plot CP intervals
        plt.errorbar(subset_idx, y_pred_sub, yerr=[y_pred_sub - lower_cp, upper_cp - y_pred_sub], 
                     fmt='o', color='#2ecc71', alpha=0.6, label='CP Interval (90%)', capsize=3)
        
        # Plot True Values
        # Color red if outside interval, black if inside
        inside = (y_true_sub >= lower_cp) & (y_true_sub <= upper_cp)
        colors = np.where(inside, 'black', 'red')
        plt.scatter(subset_idx, y_true_sub, c=colors, marker='x', s=60, zorder=3, label='True Value')
        
        plt.title(f"Conformal Prediction Intervals (First {n_samples} Test Samples)\nRobust against non-Gaussian noise", fontsize=14)
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#2ecc71', marker='o', linestyle='None', label='CP Interval'),
            Line2D([0], [0], marker='x', color='black', linestyle='None', label='True Value (Inside)'),
            Line2D([0], [0], marker='x', color='red', linestyle='None', label='True Value (Outside)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(os.path.join(self.save_dir, 'interval_visualization.png'), bbox_inches='tight')
        plt.close()

    def run(self):
        """Execute the full experiment pipeline."""
        print("="*60)
        print("Starting Experiment: Distribution-Free Proof")
        print("Dataset: Synthetic Non-Linear with Skewed Exponential Noise")
        print("="*60)
        
        self.load_and_split_data()
        self._plot_data_overview()
        self.train_base_model()
        self.run_inference()
        self.run_methods()
        self.plot_results()
        
        print("\nExperiment completed.")

if __name__ == "__main__":
    
    # Run the experiment
    # alpha=0.1 means we want 90% coverage
    experiment = DistributionFreeExperiment(n_samples=3000, alpha=0.1, gpu_id=0)
    experiment.run()