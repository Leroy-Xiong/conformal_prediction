import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Import All 7 Model Types ---
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Configuration for high-quality publication plots
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid")

class ModelFreeExperiment:
    """
    Experiment: Model Agnosticism (Model-Free).
    Sorted by Accuracy (MSE) and exported as separate images.
    """

    def __init__(self, alpha=0.1, random_state=42, save_dir='./results/model_free'):
        self.alpha = alpha
        self.random_state = random_state
        self.save_dir = save_dir
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_data(self):
        print("[1/6] Downloading Dataset via KaggleHub...")
        try:
            path = kagglehub.dataset_download("elikplim/concrete-compressive-strength-data-set")
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV file found.")
            
            df = pd.read_csv(csv_files[0])
            self.dataset_name = "Concrete Strength"
            self.X = df.iloc[:, :-1].values
            self.y = df.iloc[:, -1].values
            print(f"      Loaded. Shape: {self.X.shape}")
            
        except Exception as e:
            print(f"      Error: {e}")
            raise e

    def prepare_data(self):
        print("[2/6] Splitting and Scaling Data...")
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=self.random_state
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_full, y_train_full, test_size=0.4, random_state=self.random_state
        )
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_cal = scaler.transform(X_cal)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_cal = y_cal
        self.y_test = y_test

    def run_comparison(self):
        print("[3/6] Training 7 Models and Calibrating...")
        
        models = {
            'Linear (Ridge)': Ridge(alpha=1.0),
            'KNN (Distance)': KNeighborsRegressor(n_neighbors=5),
            'SVR (Kernel)': SVR(kernel='rbf', C=10.0, epsilon=0.1),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=self.random_state),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=2000, random_state=self.random_state)
        }
        
        results = []

        for name, model in models.items():
            print(f"      Processing: {name}...")
            model.fit(self.X_train, self.y_train)
            
            pred_cal = model.predict(self.X_cal)
            scores = np.abs(self.y_cal - pred_cal)
            
            n_cal = len(self.y_cal)
            q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
            q_level = min(1.0, q_level)
            q_hat = np.quantile(scores, q_level, method='higher')
            
            pred_test = model.predict(self.X_test)
            lower = pred_test - q_hat
            upper = pred_test + q_hat
            
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

        # SORTING STEP: Sort by MSE Descending (High Error -> Low Error)
        # This makes the plots show the progression from "Weak" to "Strong" models
        self.results_df = pd.DataFrame(results).sort_values(by='MSE', ascending=False)
        print("\nExperiment Results (Sorted by MSE):")
        print(self.results_df)

    def plot_results(self):
        print("[4/6] Generating Separate Plots...")
        
        # Define a consistent color palette
        # Use 'Spectral' so the transition corresponds to the model sorting
        palette = sns.color_palette("Spectral", n_colors=7)

        # --- Plot 1: Validity (Coverage) ---
        plt.figure(figsize=(8, 6))
        ax1 = sns.barplot(data=self.results_df, x='Model', y='Coverage', hue='Model', palette=palette, legend=False)
        plt.axhline(1 - self.alpha, color='black', linestyle='--', linewidth=2, label='Target (90%)')
        plt.ylim(0.0, 1.15)
        plt.title("Validity: Coverage is Model-Invariant", fontsize=16)
        plt.ylabel("Empirical Coverage Rate", fontsize=14)
        plt.xlabel("")
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        
        for i, row in enumerate(self.results_df.itertuples()):
            ax1.text(i, row.Coverage + 0.02, f"{row.Coverage:.1%}", ha='center', fontweight='bold', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'model_free_validity.png'))
        print("      Saved: model_free_validity.png")
        plt.close()

        # --- Plot 2: Accuracy (MSE) ---
        plt.figure(figsize=(8, 6))
        ax2 = sns.barplot(data=self.results_df, x='Model', y='MSE', hue='Model', palette=palette, legend=False)
        plt.title("Accuracy: MSE (Sorted Worst to Best)", fontsize=16)
        plt.ylabel("Mean Squared Error (Lower is Better)", fontsize=14)
        plt.xlabel("")
        plt.xticks(rotation=45, ha='right')
        
        for i, row in enumerate(self.results_df.itertuples()):
            ax2.text(i, row.MSE + 5, f"{row.MSE:.0f}", ha='center', color='black', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'model_free_accuracy.png'))
        print("      Saved: model_free_accuracy.png")
        plt.close()

        # --- Plot 3: Efficiency (Width) ---
        plt.figure(figsize=(8, 6))
        ax3 = sns.barplot(data=self.results_df, x='Model', y='Avg Width', hue='Model', palette=palette, legend=False)
        plt.title("Efficiency: Interval Width", fontsize=16)
        plt.ylabel("Average Width (Narrower is Better)", fontsize=14)
        plt.xlabel("")
        plt.xticks(rotation=45, ha='right')
        
        for i, row in enumerate(self.results_df.itertuples()):
            ax3.text(i, row._3 + 1, f"{row._3:.1f}", ha='center', color='black', fontsize=10) # row._3 is Avg Width
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'model_free_efficiency.png'))
        print("      Saved: model_free_efficiency.png")
        plt.close()
        
    def run(self):
        self.load_data()
        self.prepare_data()
        self.run_comparison()
        self.plot_results()

if __name__ == "__main__":
    exp = ModelFreeExperiment(alpha=0.1)
    exp.load_data()
    exp.prepare_data()
    exp.run_comparison()
    exp.plot_results()