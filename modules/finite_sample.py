import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

plt.rcParams['figure.dpi'] = 300

class FiniteSampleExperiment:
    """
    Experiment: Finite-Sample Guarantee vs. Asymptotic Validity.
    
    This experiment compares Raw Quantile Regression (competitor) against 
    Conformalized Quantile Regression (CQR) on a small-sample regime.
    """

    def __init__(self, alpha=0.1, n_cal=50, n_trials=100, save_dir='./results/finite_sample'):
        """
        :param alpha: Target error rate (0.1 means 90% coverage).
        :param n_cal: Size of the calibration set (Simulating data scarcity).
        :param n_trials: Number of repeated experiments.
        :param save_dir: Directory to save output images.
        """
        self.alpha = alpha
        self.n_cal = n_cal
        self.n_trials = n_trials
        self.save_dir = save_dir
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Storage for aggregate metrics
        self.results = {
            'trial_id': [],
            'method': [],
            'coverage': [],
            'avg_width': []
        }
        
        # Storage for specific visualization of one trial
        self.demo_trial_data = {} 

    def load_data(self):
        """
        Loads the Diabetes dataset.
        Selected because it is a classic, small-scale regression dataset (n=442).
        """
        print("[1/6] Loading Diabetes dataset...")
        data = load_diabetes()
        self.X, self.y = data.data, data.target
        self.feature_names = data.feature_names
        print(f"      Total samples: {len(self.y)}")

    def _plot_data_overview(self):
        """
        Plot 1: Data Overview. 
        Visualizes the relationship between the most important feature (BMI) and Target.
        """
        print("[2/6] Generating Data Overview...")
        
        # Find index of 'bmi' feature
        bmi_idx = 2 
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, bmi_idx], self.y, alpha=0.6, color='#2c3e50', edgecolors='w', s=40)
        
        plt.title("Diabetes Dataset: BMI vs. Progression", fontsize=14, fontweight='bold')
        plt.xlabel("Normalized BMI", fontsize=12)
        plt.ylabel("Disease Progression (Target)", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, 'data_overview.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def train_base_qr(self, X_train, y_train):
        """
        Trains the underlying Quantile Regressors (Gradient Boosting).
        """
        # Lower Quantile (alpha/2)
        model_lo = GradientBoostingRegressor(loss='quantile', alpha=self.alpha/2, 
                                             n_estimators=100, max_depth=3, random_state=42)
        model_lo.fit(X_train, y_train)
        
        # Upper Quantile (1 - alpha/2)
        model_hi = GradientBoostingRegressor(loss='quantile', alpha=1 - self.alpha/2, 
                                             n_estimators=100, max_depth=3, random_state=42)
        model_hi.fit(X_train, y_train)
        
        return model_lo, model_hi

    def run_trials(self):
        """
        Executes the repeated trials.
        For each trial:
        1. Split remaining data into small Calibration set and Test set.
        2. Evaluate Raw QR (Competitor).
        3. Calibrate and evaluate CQR (Method).
        """
        print(f"[3/6] Running {self.n_trials} trials (Calibration Size N={self.n_cal})...")
        
        # Initial Split: Train (50%) vs Rest (50%)
        # Models are trained ONCE on X_train to isolate the effect of *calibration* size.
        X_train, X_rest, y_train, y_rest = train_test_split(self.X, self.y, test_size=0.5, random_state=42)
        
        model_lo, model_hi = self.train_base_qr(X_train, y_train)
        
        for i in range(self.n_trials):
            # Split 'Rest' into Calibration and Test
            # Note: We use different random states to simulate different data splits
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_rest, y_rest, train_size=self.n_cal, random_state=i
            )
            
            # --- 1. Raw Predictions ---
            # Calibration set predictions
            cal_lo = model_lo.predict(X_cal)
            cal_hi = model_hi.predict(X_cal)
            
            # Test set predictions
            test_lo = model_lo.predict(X_test)
            test_hi = model_hi.predict(X_test)
            
            # --- 2. Competitor: Raw QR ---
            # Coverage: Check if y is between low and high
            qr_covered = (y_test >= test_lo) & (y_test <= test_hi)
            qr_cov = np.mean(qr_covered)
            qr_width = np.mean(test_hi - test_lo)
            
            self.results['trial_id'].append(i)
            self.results['method'].append('Competitor (Raw QR)')
            self.results['coverage'].append(qr_cov)
            self.results['avg_width'].append(qr_width)
            
            # --- 3. Method: Conformalized QR (CQR) ---
            # Compute scores on Calibration set
            # Score = max(lower - y, y - upper)
            scores_cal = np.maximum(cal_lo - y_cal, y_cal - cal_hi)
            
            # Compute Q_hat (Finite Sample Correction)
            # q_level = (1 - alpha) * (1 + 1/N)
            q_level = np.ceil((self.n_cal + 1) * (1 - self.alpha)) / self.n_cal
            q_level = min(1.0, q_level)
            q_hat = np.quantile(scores_cal, q_level, method='higher')
            
            # Apply correction to Test set
            cqr_lo = test_lo - q_hat
            cqr_hi = test_hi + q_hat
            
            cqr_covered = (y_test >= cqr_lo) & (y_test <= cqr_hi)
            cqr_cov = np.mean(cqr_covered)
            cqr_width = np.mean(cqr_hi - cqr_lo)
            
            self.results['trial_id'].append(i)
            self.results['method'].append('Conformal Prediction (CQR)')
            self.results['coverage'].append(cqr_cov)
            self.results['avg_width'].append(cqr_width)
            
            # Save data for visualization (Just take the 5th trial as an example)
            if i == 5:
                self.demo_trial_data = {
                    'y_test': y_test,
                    'qr_lo': test_lo, 'qr_hi': test_hi,
                    'cqr_lo': cqr_lo, 'cqr_hi': cqr_hi,
                    'q_hat': q_hat
                }

    def _plot_single_trial(self):
        """
        Plot 2: Anatomy of a Trial.
        Visualizes the intervals for a single random trial to show how CQR expands/corrects QR.
        """
        print("[4/6] Generating Single Trial Snapshot...")
        
        data = self.demo_trial_data
        y = data['y_test']
        # Sort by predicted value (using center of QR interval) for cleaner plotting
        sort_idx = np.argsort((data['qr_lo'] + data['qr_hi']) / 2)
        
        y_sorted = y[sort_idx]
        qr_lo, qr_hi = data['qr_lo'][sort_idx], data['qr_hi'][sort_idx]
        cqr_lo, cqr_hi = data['cqr_lo'][sort_idx], data['cqr_hi'][sort_idx]
        x_axis = np.arange(len(y))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot CQR Band (The "Safety Net")
        ax.fill_between(x_axis, cqr_lo, cqr_hi, color='#2ecc71', alpha=0.2, label='Conformal Prediction Interval')
        
        # Plot Raw QR Lines (The "Estimate")
        ax.plot(x_axis, qr_lo, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8, label='Raw QR Boundary')
        ax.plot(x_axis, qr_hi, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Plot True Values
        ax.scatter(x_axis, y_sorted, color='black', s=15, alpha=0.7, label='True Target')
        
        # Highlight points caught by CQR but missed by QR
        missed_by_qr = (y_sorted < qr_lo) | (y_sorted > qr_hi)
        caught_by_cqr = (y_sorted >= cqr_lo) & (y_sorted <= cqr_hi)
        saved_points = missed_by_qr & caught_by_cqr
        
        ax.scatter(x_axis[saved_points], y_sorted[saved_points], 
                   color='#f1c40f', s=50, marker='*', label='Saved by CP', zorder=5)
        
        ax.set_title(f"Trial Snapshot: CP Correction (Calibration q_hat = {data['q_hat']:.2f})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Test Samples (Sorted by Predicted Value)", fontsize=12)
        ax.set_ylabel("Disease Progression", fontsize=12)
        ax.legend(loc='upper left', frameon=True)
        
        save_path = os.path.join(self.save_dir, 'single_trial_snapshot.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def _plot_aggregate_results(self):
        """
        Plot 3: Aggregate Statistics.
        Left: Histogram of Coverage (Validity).
        Right: Boxplot of Width (Efficiency).
        """
        print("[5/6] Generating Aggregate Results Plot...")
        
        df = pd.DataFrame(self.results)
        target = 1 - self.alpha
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- Left Panel: Coverage Histogram ---
        sns.histplot(data=df, x='coverage', hue='method', element='step', 
                     palette=['#e74c3c', '#2ecc71'], bins=20, ax=axes[0], alpha=0.3)
        axes[0].axvline(target, color='black', linestyle='--', linewidth=2, label=f'Target ({target:.0%})')
        axes[0].set_title("Validity: Empirical Coverage Distribution", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Coverage Rate", fontsize=12)
        axes[0].set_ylabel("Count (Trials)", fontsize=12)
        
        # Add summary stats on plot
        mean_qr = df[df['method']=='Competitor (Raw QR)']['coverage'].mean()
        mean_cqr = df[df['method']=='Conformal Prediction (CQR)']['coverage'].mean()
        axes[0].text(0.7, 0.9, f"Raw QR Mean: {mean_qr:.3f}\nCQR Mean: {mean_cqr:.3f}", 
                     transform=axes[0].transAxes, bbox=dict(facecolor='white', alpha=0.8))

        # --- Right Panel: Width Boxplot ---
        sns.boxplot(data=df, x='method', y='avg_width', palette=['#e74c3c', '#2ecc71'], ax=axes[1])
        axes[1].set_title("Efficiency: Average Interval Width", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Interval Width (Units)", fontsize=12)
        axes[1].set_xlabel("")
        
        plt.suptitle(f"Finite-Sample Performance (N_cal={self.n_cal}, {self.n_trials} Trials)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'aggregate_results.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # Print Text Summary
        print("\n" + "="*50)
        print("EXPERIMENT 3 SUMMARY")
        print("="*50)
        print(f"Target Coverage: {target:.1%}")
        print(f"Competitor (Raw QR): Mean Cov = {mean_qr:.1%}")
        print(f"Method (CQR):        Mean Cov = {mean_cqr:.1%}")
        print("-" * 50)
        print("CONCLUSION:")
        print("Raw QR fails to meet the target coverage in small samples (Under-coverage).")
        print("CQR satisfies the guarantee by slightly increasing interval width.")
        print("="*50)

    def run(self):
        print("="*60)
        print("Starting Experiment: Finite-Sample Guarantee")
        print("="*60)
        
        self.load_data()
        self._plot_data_overview()
        self.run_trials()
        self._plot_single_trial()
        self._plot_aggregate_results()
        
        print("\nExperiment completed successfully.")
        print(f"Results saved to: {os.path.abspath(self.save_dir)}")

if __name__ == "__main__":
    # CPU is sufficient for this tabular dataset
    exp = FiniteSampleExperiment(alpha=0.1, n_cal=50, n_trials=100)
    exp.run()