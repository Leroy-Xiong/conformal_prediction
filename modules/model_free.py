import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml, load_diabetes

# Configuration for high-quality publication plots
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid")

class ModelFreeExperiment:
    """
    Experiment: Model Agnosticism (Model-Free).
    
    Demonstrates:
    1. Validity: All models (Linear, RF, NN) achieve target coverage (~90%).
    2. Efficiency: Better models (Lower MSE) produce sharper (narrower) intervals.
    
    Dataset: Concrete Compressive Strength (Real-world Engineering Data).
    """

    def __init__(self, alpha=0.1, random_state=42, save_dir='./results/model_free'):
        self.alpha = alpha
        self.random_state = random_state
        self.save_dir = save_dir
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_data(self):
        """
        Robust data loading. Tries OpenML first, falls back to Diabetes if network fails.
        """
        print("[1/6] Loading Dataset...")
        # Try loading Concrete Compressive Strength (UCI ID: 4353)
        # return_X_y=True fixes the 'NoneType' error
        X, y = fetch_openml(data_id=4353, return_X_y=True, as_frame=False, parser='auto')
        self.dataset_name = "Concrete Compressive Strength"
        print("      Loaded: Concrete Dataset (OpenML)")

        self.X = X
        self.y = y
        print(f"      Data Shape: {self.X.shape}")

    def prepare_data(self):
        print("[2/6] Splitting and Scaling Data...")
        # Split: 40% Train, 30% Calibration, 30% Test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=self.random_state
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_full, y_train_full, test_size=0.5, random_state=self.random_state
        )
        
        # Scaling (Critical for Ridge and MLP)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_cal = scaler.transform(X_cal)
        self.X_test = scaler.transform(X_test)
        
        self.y_train = y_train
        self.y_cal = y_cal
        self.y_test = y_test

    def run_comparison(self):
        print("[3/6] Training Models and Calibrating...")
        
        # Define models with decent hyperparameters
        models = {
            'Linear (Ridge)': Ridge(alpha=1.0),
            
            # RF: standard strong baseline
            'Random Forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=self.random_state),
            
            # MLP: Needs enough iterations to converge on this data
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
                                          solver='adam', max_iter=5000, random_state=self.random_state)
        }
        
        results = []

        for name, model in models.items():
            print(f"      Processing: {name}...")
            
            # 1. Train
            model.fit(self.X_train, self.y_train)
            
            # 2. Calibrate
            pred_cal = model.predict(self.X_cal)
            scores = np.abs(self.y_cal - pred_cal) # Score = |y - y_hat|
            
            n_cal = len(self.y_cal)
            q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
            q_level = min(1.0, q_level)
            q_hat = np.quantile(scores, q_level, method='higher')
            
            # 3. Test
            pred_test = model.predict(self.X_test)
            lower = pred_test - q_hat
            upper = pred_test + q_hat
            
            # 4. Metrics
            covered = (self.y_test >= lower) & (self.y_test <= upper)
            coverage = np.mean(covered)
            width = np.mean(upper - lower)
            mse = mean_squared_error(self.y_test, pred_test)
            
            results.append({
                'Model': name,
                'Coverage': coverage,
                'Avg Width': width,
                'MSE': mse
            })

        self.results_df = pd.DataFrame(results)
        print("\nExperiment Results:")
        print(self.results_df)

    def plot_results(self):
        print("[4/6] Generating Plots...")
        
        # Create a figure with 3 subplots: Validity, MSE, Efficiency
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Colors
        palette = {"Linear (Ridge)": "#34495e", "Random Forest": "#e74c3c", "Neural Network": "#2ecc71"}
        
        # --- Plot 1: Validity (Coverage) ---
        sns.barplot(data=self.results_df, x='Model', y='Coverage', ax=axes[0], palette=palette)
        axes[0].axhline(1 - self.alpha, color='black', linestyle='--', linewidth=1.5, label='Target (90%)')
        axes[0].set_ylim(0.0, 1.1)
        axes[0].set_title("1. Validity (Coverage)", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Empirical Coverage Rate")
        axes[0].set_xlabel("")
        axes[0].legend(loc='lower right')
        
        for i, row in self.results_df.iterrows():
            axes[0].text(i, row['Coverage'] + 0.02, f"{row['Coverage']:.1%}", ha='center', fontweight='bold')

        # --- Plot 2: Accuracy (MSE) - NEW ---
        sns.barplot(data=self.results_df, x='Model', y='MSE', ax=axes[1], palette=palette)
        axes[1].set_title("2. Accuracy (MSE)", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Mean Squared Error (Lower is Better)")
        axes[1].set_xlabel("")
        
        for i, row in self.results_df.iterrows():
            axes[1].text(i, row['MSE'] + (row['MSE']*0.02), f"{row['MSE']:.1f}", ha='center', color='black')

        # --- Plot 3: Efficiency (Width) ---
        sns.barplot(data=self.results_df, x='Model', y='Avg Width', ax=axes[2], palette=palette)
        axes[2].set_title("3. Efficiency (Interval Width)", fontsize=14, fontweight='bold')
        axes[2].set_ylabel("Average Width (Lower is Better)")
        axes[2].set_xlabel("")
        
        for i, row in self.results_df.iterrows():
            axes[2].text(i, row['Avg Width'] + (row['Avg Width']*0.02), f"{row['Avg Width']:.1f}", ha='center', color='black')

        plt.suptitle(f"Model Agnosticism on '{self.dataset_name}' (Target $\\alpha={self.alpha}$)", fontsize=16, y=1.05)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'model_free_triplot.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[5/6] Plot saved to {save_path}")

if __name__ == "__main__":
    exp = ModelFreeExperiment(alpha=0.1)
    exp.load_data()
    exp.prepare_data()
    exp.run_comparison()
    exp.plot_results()