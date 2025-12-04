import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


plt.rcParams['figure.dpi'] = 300


class CaliforniaHousing:
    """
    Real Data Regression (California Housing).
    Comparison between Baseline (Gaussian Assumption) and Conformal Prediction.
    
    Includes Data Visualization and GPU Support.
    """

    def __init__(self, random_state=42, alpha=0.1, save_dir='./results/california_housing', gpu_id=0):
        """
        Initialize the experiment settings.
        :param random_state: Seed for reproducibility.
        :param alpha: Error rate (e.g., 0.1 for 90% coverage).
        :param save_dir: Directory to save the output plots.
        :param gpu_id: The ID of the GPU to use (e.g., 0 or 1). Falls back to CPU if unavailable.
        """
        self.random_state = random_state
        self.alpha = alpha
        self.save_dir = save_dir
        
        # --- Robust Device Configuration ---
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                print(f"[Warning] Requested gpu_id={gpu_id} but only {num_gpus} available. Fallback to 0.")
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
        """Sets random seeds for CPU and GPU to ensure reproducibility."""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_and_split_data(self):
        """
        Loads California Housing dataset, prepares raw DataFrame for viz, and Tensors for training.
        """
        print("[1/6] Loading and splitting data...")
        data = fetch_california_housing()
        X, y = data.data, data.target
        feature_names = data.feature_names

        # Save raw dataframe for visualization purposes
        self.raw_df = pd.DataFrame(X, columns=feature_names)
        self.raw_df['MedHouseVal'] = y

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
        Visualizes the raw data distribution to justify the complexity.
        Saved as 'data_overview.png'.
        """
        print("[2/6] Visualizing raw data...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Geographic Distribution (The "Map")
        sns.scatterplot(
            data=self.raw_df, x='Longitude', y='Latitude', hue='MedHouseVal', 
            palette='viridis', alpha=0.4, s=20, ax=axes[0]
        )
        axes[0].set_title('Geographic Distribution of Prices', fontsize=14)
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')

        # Subplot 2: Target Distribution (Histogram)
        sns.histplot(self.raw_df['MedHouseVal'], kde=True, color='orange', ax=axes[1])
        axes[1].set_title('Distribution of House Prices (Target)', fontsize=14)
        axes[1].set_xlabel('Median House Value ($100k)')
        
        # Subplot 3: Feature vs Target (MedInc is the strongest predictor)
        sns.scatterplot(
            data=self.raw_df, x='MedInc', y='MedHouseVal', 
            alpha=0.2, color='steelblue', ax=axes[2]
        )
        axes[2].set_title('Median Income vs. House Price', fontsize=14)
        axes[2].set_xlabel('Median Income')
        axes[2].set_ylabel('House Price')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'data_overview.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"      Data overview saved to {save_path}")
        plt.close()

    def train_base_model(self):
        """Trains a simple MLP on the specified device."""
        print(f"[3/6] Training base neural network on {self.device}...")
        
        self.model = nn.Sequential(
            nn.Linear(self.X_train.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        X_train_dev = self.X_train.to(self.device)
        y_train_dev = self.y_train.to(self.device)
        
        epochs = 300
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train_dev)
            loss = criterion(outputs, y_train_dev)
            loss.backward()
            optimizer.step()
        print("      Training complete.")

    def run_inference(self):
        """Runs inference on GPU and moves results back to CPU."""
        print("[4/6] Running inference...")
        self.model.eval()
        X_cal_dev = self.X_cal.to(self.device)
        X_test_dev = self.X_test.to(self.device)
        
        with torch.no_grad():
            self.y_pred_cal = self.model(X_cal_dev).cpu().numpy().flatten()
            self.y_pred_test = self.model(X_test_dev).cpu().numpy().flatten()

    def run_methods(self):
        """Implements Baseline vs CP logic."""
        print("[5/6] Computing uncertainty intervals...")
        
        # Method A: Baseline (Gaussian)
        cal_residuals = self.y_cal_numpy - self.y_pred_cal
        sigma_hat = np.sqrt(np.mean(cal_residuals**2))
        z_score = norm.ppf(1 - self.alpha / 2)
        
        # Calculate Half-Width
        self.width_baseline = z_score * sigma_hat
        
        self.y_lower_base = self.y_pred_test - self.width_baseline
        self.y_upper_base = self.y_pred_test + self.width_baseline
        self.cal_residuals = cal_residuals

        # Method B: Split Conformal Prediction
        scores_cal = np.abs(self.y_cal_numpy - self.y_pred_cal)
        n = len(self.y_cal_numpy)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        
        # Calculate Half-Width (q_hat)
        self.q_hat = np.quantile(scores_cal, q_level, method='higher')
        
        self.y_lower_cp = self.y_pred_test - self.q_hat
        self.y_upper_cp = self.y_pred_test + self.q_hat

        # Metrics
        cov_base = np.mean((self.y_test_numpy >= self.y_lower_base) & (self.y_test_numpy <= self.y_upper_base))
        cov_cp = np.mean((self.y_test_numpy >= self.y_lower_cp) & (self.y_test_numpy <= self.y_upper_cp))
        
        self.metrics = {'Baseline Coverage': cov_base, 'CP Coverage': cov_cp, 'Target': 1 - self.alpha}
        
        print("="*40)
        print("          RESULTS SUMMARY")
        print("="*40)
        print(f"      Target Coverage: {1-self.alpha:.1%}")
        print(f"      Baseline Coverage: {cov_base:.1%} (FAIL)")
        print(f"      CP Coverage:       {cov_cp:.1%} (PASS)")
        print("-" * 40)
        
        # --- NEW: Calculate and Print Interval Widths ---
        # Note: In this method (Split Conformal with absolute residuals), the width is constant for all points.
        # Width = Upper - Lower
        self.avg_width_base = self.width_baseline * 2
        self.avg_width_cp = self.q_hat * 2
        
        print(f"      Gaussian Interval Width:  {self.avg_width_base:.4f}")
        print(f"      Conformal Interval Width: {self.avg_width_cp:.4f}")
        
        # Add interpretation
        if self.avg_width_cp > self.avg_width_base:
            diff_pct = (self.avg_width_cp - self.avg_width_base) / self.avg_width_base * 100
            print(f"      [Analysis] CP interval is {diff_pct:.1f}% wider to ensure safety.")
        print("="*40)

    def plot_results(self):
        """Generates all results plots."""
        print("[6/6] Generating result plots...")
        self._plot_residual_distribution()
        self._plot_coverage_comparison()
        self._plot_interval_visualization()
        print(f"Results saved to: {os.path.abspath(self.save_dir)}")

    def _plot_residual_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.cal_residuals, kde=True, stat="density", 
                     color="#3498db", label="Actual Residuals (Calibration)", alpha=0.6)
        
        mu, std = norm.fit(self.cal_residuals)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'r--', linewidth=2.5, label=f"Gaussian Assumption\nN(0, {std:.2f})")
        
        plt.title("Why Baseline Fails: Non-Gaussian Residuals", fontsize=16)
        plt.xlabel("Residual Value", fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(self.save_dir, 'residual_distribution.png'), bbox_inches='tight')
        plt.close()

    def _plot_coverage_comparison(self):
        methods = ['Baseline\n(Gaussian)', 'Conformal Prediction\n(Distribution-Free)']
        coverages = [self.metrics['Baseline Coverage'], self.metrics['CP Coverage']]
        target = self.metrics['Target']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(methods, coverages, color=['#e74c3c', '#2ecc71'], alpha=0.85, width=0.6)
        plt.axhline(y=target, color='black', linestyle='--', linewidth=2, label=f'Target ({target:.0%})')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.1%}', ha='center', va='bottom', fontsize=14)
            
        plt.ylim(0, 1.1)
        plt.ylabel("Empirical Coverage Rate", fontsize=12)
        plt.title("Coverage Comparison", fontsize=16)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.save_dir, 'coverage_comparison.png'), bbox_inches='tight')
        plt.close()
        
    def _plot_interval_visualization(self, n_samples=50):
        subset_idx = np.arange(n_samples)
        y_true_sub = self.y_test_numpy[subset_idx]
        y_pred_sub = self.y_pred_test[subset_idx]
        
        lower_base = self.y_lower_base[subset_idx]
        upper_base = self.y_upper_base[subset_idx]
        lower_cp = self.y_lower_cp[subset_idx]
        upper_cp = self.y_upper_cp[subset_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Plot Baseline
        axes[0].errorbar(subset_idx, y_pred_sub, yerr=[y_pred_sub - lower_base, upper_base - y_pred_sub], 
                         fmt='o', color='#e74c3c', alpha=0.6, label=f'Width: {self.avg_width_base:.2f}')
        axes[0].scatter(subset_idx, y_true_sub, color='black', marker='x', s=40, label='True Value')
        axes[0].set_title(f"Baseline: {self.metrics['Baseline Coverage']:.1%} Coverage", fontsize=14)
        axes[0].legend(loc='upper right')
        
        # Plot CP
        axes[1].errorbar(subset_idx, y_pred_sub, yerr=[y_pred_sub - lower_cp, upper_cp - y_pred_sub], 
                         fmt='o', color='#2ecc71', alpha=0.6, label=f'Width: {self.avg_width_cp:.2f}')
        axes[1].scatter(subset_idx, y_true_sub, color='black', marker='x', s=40, label='True Value')
        axes[1].set_title(f"CP: {self.metrics['CP Coverage']:.1%} Coverage", fontsize=14)
        axes[1].legend(loc='upper right')
        
        plt.suptitle("Prediction Intervals (First 50 Test Samples)", fontsize=16)
        plt.savefig(os.path.join(self.save_dir, 'interval_visualization.png'), bbox_inches='tight')
        plt.close()

    def run(self):
        """Execute the full experiment pipeline."""
        print("="*50)
        print("Starting Experiment: Real Data Regression")
        print("="*50)
        
        self.load_and_split_data()
        self._plot_data_overview()
        self.train_base_model()
        self.run_inference()
        self.run_methods()
        self.plot_results()
        
        print("\nExperiment completed successfully.")

if __name__ == "__main__":
    
    experiment = CaliforniaHousing(alpha=0.1, gpu_id=0)
    experiment.run()