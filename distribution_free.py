import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from typing import Tuple, List

def generate_data(n_samples: int = 1000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for regression experiment.
    
    Parameters:
    n_samples (int): Number of data points to generate
    random_state (int): Random seed for reproducibility
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: X (features) and y (target)
    """
    np.random.seed(random_state)
    X = np.random.uniform(-4, 4, size=(n_samples, 2))
    # Non-linear relationship with noise
    y = 2 * X[:, 0] + np.sin(X[:, 1]) + 0.5 * np.random.randn(n_samples)
    return X, y

def split_conformal_prediction(X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> Tuple[dict, np.ndarray]:
    """
    Implement split conformal prediction for regression.
    
    Parameters:
    X (np.ndarray): Feature matrix
    y (np.ndarray): Target values
    alpha (float): Significance level (1 - coverage)
    
    Returns:
    Tuple[dict, np.ndarray]: Results dictionary and calibration scores
    """
    # Split data into proper training set and calibration set
    X_proper_train, X_calib, y_proper_train, y_calib = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Split calibration set into calibration and test sets
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_calib, y_calib, test_size=0.5, random_state=42
    )
    
    # Train regression model on proper training set
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_proper_train, y_proper_train)
    
    # Calculate residuals on calibration set
    y_calib_pred = model.predict(X_calib)
    residuals_calib = np.abs(y_calib - y_calib_pred)
    
    # Calculate conformity scores and quantile
    quantile_level = np.ceil((len(residuals_calib) + 1) * (1 - alpha)) / len(residuals_calib)
    quantile = np.quantile(residuals_calib, quantile_level, method='higher')
    
    # Generate predictions and prediction intervals for test set
    y_test_pred = model.predict(X_test)
    lower_bounds = y_test_pred - quantile
    upper_bounds = y_test_pred + quantile
    
    # Calculate coverage and average interval length
    coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
    avg_interval_length = np.mean(upper_bounds - lower_bounds)
    
    results = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'quantile': quantile,
        'coverage': coverage,
        'avg_interval_length': avg_interval_length,
        'alpha': alpha
    }
    
    return results, residuals_calib

def plot_results(results: dict, residuals: np.ndarray, save_path: str = None) -> None:
    """
    Plot conformal prediction results and diagnostics.
    
    Parameters:
    results (dict): Results dictionary from split_conformal_prediction
    residuals (np.ndarray): Calibration set residuals
    save_path (str): Path to save the plot (optional)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: True vs Predicted values with prediction intervals
    sorted_indices = np.argsort(results['y_test'])
    y_test_sorted = results['y_test'][sorted_indices]
    y_pred_sorted = results['y_pred'][sorted_indices]
    lower_sorted = results['lower_bounds'][sorted_indices]
    upper_sorted = results['upper_bounds'][sorted_indices]
    
    ax1.scatter(range(len(y_test_sorted)), y_test_sorted, alpha=0.7, 
                label='True Values', s=10, color='blue')
    ax1.plot(range(len(y_pred_sorted)), y_pred_sorted, color='red', 
             label='Predictions', linewidth=2)
    ax1.fill_between(range(len(y_test_sorted)), lower_sorted, upper_sorted, 
                     alpha=0.3, color='gray', label=f'{100*(1-results["alpha"]):.0f}% Prediction Interval')
    ax1.set_xlabel('Sample Index (Sorted)')
    ax1.set_ylabel('Target Value')
    ax1.set_title(f'Split Conformal Prediction Results\nCoverage: {results["coverage"]:.3f}, '
                 f'Average Interval Length: {results["avg_interval_length"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals distribution
    ax2.hist(residuals, bins=min(30, len(np.unique(residuals))), alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(results['quantile'], color='red', linestyle='--', 
                label=f'Quantile: {results["quantile"]:.3f}')
    ax2.set_xlabel('Absolute Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Calibration Set Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction intervals vs true values
    # Select a subset of points for clearer visualization
    n_points = min(100, len(results['y_test']))
    indices = np.random.choice(len(results['y_test']), n_points, replace=False)
    
    ax3.errorbar(results['y_test'][indices], results['y_pred'][indices], 
                yerr=[results['y_pred'][indices] - results['lower_bounds'][indices], 
                      results['upper_bounds'][indices] - results['y_pred'][indices]],
                fmt='o', alpha=0.7, markersize=4, capsize=3, color='purple', elinewidth=1)
    ax3.plot([results['y_test'].min(), results['y_test'].max()], 
             [results['y_test'].min(), results['y_test'].max()], 
             'r--', alpha=0.8, label='Ideal Prediction', linewidth=2)
    ax3.set_xlabel('True Values')
    ax3.set_ylabel('Predicted Values with Intervals')
    ax3.set_title('Prediction Intervals vs True Values (Subset)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Interval length distribution - FIXED
    interval_lengths = results['upper_bounds'] - results['lower_bounds']
    
    # Calculate appropriate number of bins
    unique_lengths = len(np.unique(interval_lengths))
    n_bins = min(30, unique_lengths, len(interval_lengths)//5)
    n_bins = max(5, n_bins)  # Ensure at least 5 bins
    
    if n_bins > 1:
        ax4.hist(interval_lengths, bins=n_bins, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(results['avg_interval_length'], color='red', linestyle='--',
                    label=f'Average Length: {results["avg_interval_length"]:.3f}')
        ax4.set_xlabel('Prediction Interval Length')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Prediction Interval Lengths')
        ax4.legend()
    else:
        # If only one unique value, show a bar instead of histogram
        ax4.bar(0, len(interval_lengths), color='orange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Constant Interval Length')
        ax4.set_ylabel('Count')
        ax4.set_title(f'All intervals have length: {interval_lengths[0]:.3f}')
        ax4.set_xticks([])
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def run_multiple_alpha_experiment(X: np.ndarray, y: np.ndarray, 
                                 alphas: List[float] = None) -> pd.DataFrame:
    """
    Run conformal prediction with multiple alpha values.
    
    Parameters:
    X (np.ndarray): Feature matrix
    y (np.ndarray): Target values
    alphas (List[float]): List of significance levels
    
    Returns:
    pd.DataFrame: Results for different alpha values
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    results_list = []
    for alpha in alphas:
        try:
            results, _ = split_conformal_prediction(X, y, alpha)
            results_list.append({
                'alpha': alpha,
                'nominal_coverage': 1 - alpha,
                'empirical_coverage': results['coverage'],
                'avg_interval_length': results['avg_interval_length'],
                'quantile': results['quantile']
            })
        except Exception as e:
            print(f"Error with alpha={alpha}: {e}")
            continue
    
    return pd.DataFrame(results_list)

def plot_coverage_comparison(alpha_results: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot coverage comparison for different alpha values.
    
    Parameters:
    alpha_results (pd.DataFrame): Results from multiple alpha experiment
    save_path (str): Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_results['nominal_coverage'], alpha_results['empirical_coverage'], 
             'o-', linewidth=2, markersize=8, label='Empirical Coverage')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Ideal Coverage')
    plt.xlabel('Nominal Coverage')
    plt.ylabel('Empirical Coverage')
    plt.title('Nominal vs Empirical Coverage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Coverage plot saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function to run the split conformal prediction experiment.
    """
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_data(n_samples=1000)
    print(f"Data shape: X {X.shape}, y {y.shape}")
    
    # Run single experiment with alpha = 0.1
    print("Running split conformal prediction with alpha=0.1...")
    results, residuals = split_conformal_prediction(X, y, alpha=0.1)
    
    # Print results
    print(f"\nExperiment Results:")
    print(f"Nominal Coverage: {(1-results['alpha'])*100:.1f}%")
    print(f"Empirical Coverage: {results['coverage']*100:.2f}%")
    print(f"Average Interval Length: {results['avg_interval_length']:.3f}")
    print(f"Quantile (1-alpha): {results['quantile']:.3f}")
    print(f"Number of test points: {len(results['y_test'])}")
    
    # Plot results
    plot_results(results, residuals, save_path='split_conformal_results.png')
    
    # Run experiment with multiple alpha values
    print("\nRunning experiment with multiple alpha values...")
    alpha_results = run_multiple_alpha_experiment(X, y)
    
    # Display and save results table
    print("\nResults for different alpha values:")
    print(alpha_results.round(4))
    
    # Save results to CSV
    alpha_results.to_csv('conformal_prediction_results.csv', index=False)
    print("\nResults saved to 'conformal_prediction_results.csv'")
    
    # Plot coverage comparison
    plot_coverage_comparison(alpha_results, save_path='coverage_plot.png')

if __name__ == "__main__":
    main()