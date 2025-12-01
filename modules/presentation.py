import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

# Set Chinese font for matplotlib (optional, for better Chinese character display)
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)  # For 4D input (after conv layers)
        self.dropout2 = nn.Dropout(0.5)    # For 2D input (after flatten)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  # 4D input: [batch, channels, height, width]
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)  # 2D input: [batch, features]
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """Train the CNN model"""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            data, target = data.to('cuda'), target.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

def calculate_conformity_scores(model, calib_loader, device):
    """Calculate conformity scores (nonconformity scores) for calibration set"""
    model.eval()
    conformity_scores = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in calib_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Convert logits to probabilities using softmax
            probabilities = F.softmax(output, dim=1)
            
            # Calculate conformity score: 1 - predicted probability for true class
            for i in range(len(target)):
                true_prob = probabilities[i, target[i]].item()
                conformity_scores.append(1 - true_prob)
                true_labels.append(target[i].item())
    
    return np.array(conformity_scores), np.array(true_labels)

def compute_quantile(scores, alpha):
    """Compute the (1-alpha) quantile of conformity scores with finite-sample correction"""
    n = len(scores)
    # Finite-sample correction: use ceiling instead of floor
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # Ensure it doesn't exceed 1
    return np.quantile(scores, q_level, method='higher')

def create_prediction_sets(model, test_loader, quantile, device, alpha):
    """Create prediction sets for test samples"""
    model.eval()
    prediction_sets = []
    point_predictions = []
    true_labels = []
    all_probabilities = []
    all_images = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            
            for i in range(len(target)):
                # Include all classes with probability >= 1 - quantile
                prob_vector = probabilities[i].cpu().numpy()
                prediction_set = np.where(prob_vector >= (1 - quantile))[0]
                prediction_sets.append(prediction_set)
                point_predictions.append(np.argmax(prob_vector))
                true_labels.append(target[i].item())
                all_probabilities.append(prob_vector)
                all_images.append(data[i].cpu().numpy())
    
    return prediction_sets, point_predictions, true_labels, all_probabilities, all_images

def evaluate_prediction_sets(prediction_sets, true_labels, alpha):
    """Evaluate the performance of prediction sets"""
    coverage = 0
    set_sizes = []
    
    for i in range(len(true_labels)):
        # Check if true label is in prediction set
        if true_labels[i] in prediction_sets[i]:
            coverage += 1
        set_sizes.append(len(prediction_sets[i]))
    
    coverage_rate = coverage / len(true_labels)
    avg_set_size = np.mean(set_sizes)
    
    print(f"\nSplit Conformal Prediction Results (α={alpha}):")
    print(f"Coverage rate: {coverage_rate:.4f} (Target: {1-alpha:.3f})")
    print(f"Average prediction set size: {avg_set_size:.2f}")
    print(f"Set size statistics: Min={np.min(set_sizes)}, Max={np.max(set_sizes)}")
    
    return coverage_rate, avg_set_size

def visualize_test_results(all_images, true_labels, point_predictions, prediction_sets, 
                          all_probabilities, num_samples=100, samples_per_row=10):
    """Visualize test set results with point predictions and prediction sets"""
    
    # Calculate how many rows needed
    num_rows = (num_samples + samples_per_row - 1) // samples_per_row
    
    # Create a large figure
    fig, axes = plt.subplots(num_rows, samples_per_row, figsize=(20, 2.5 * num_rows))
    
    # If only one row, make axes 2D
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        if idx >= len(all_images):
            break
            
        row = idx // samples_per_row
        col = idx % samples_per_row
        
        ax = axes[row, col]
        
        # Display the image
        img = all_images[idx].squeeze()  # Remove channel dimension for grayscale
        ax.imshow(img, cmap='gray')
        
        # Get prediction information
        true_label = true_labels[idx]
        point_pred = point_predictions[idx]
        pred_set = prediction_sets[idx]
        pred_set = [int(x) for x in pred_set]
        probs = all_probabilities[idx]
        
        # Set title with color coding
        is_correct_point = (point_pred == true_label)
        is_covered = (true_label in pred_set)
        
        # Color code: green for correct, red for incorrect
        point_color = 'green' if is_correct_point else 'red'
        set_color = 'green' if is_covered else 'orange'
        
        # Title shows true label and point prediction
        ax.set_title(f'True: {true_label}\nPoint: {point_pred}', 
                    color=point_color, fontsize=10, fontweight='bold')
        
        # Add prediction set information as text
        set_text = f'Set: {sorted(pred_set)}'
        ax.text(0.5, -0.15, set_text, transform=ax.transAxes, ha='center', 
               fontsize=8, color=set_color, fontweight='bold')
        
        # Add probability information
        max_prob = np.max(probs)
        prob_text = f'MaxP: {max_prob:.2f}'
        ax.text(0.5, -0.25, prob_text, transform=ax.transAxes, ha='center', 
               fontsize=7, color='blue')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border color based on coverage
        border_color = 'green' if is_covered else 'red'
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(2)
    
    # Hide empty subplots
    for idx in range(len(all_images), num_rows * samples_per_row):
        row = idx // samples_per_row
        col = idx % samples_per_row
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('./output/presentation/conformal_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistics(prediction_sets, true_labels, point_predictions, all_probabilities, alpha):
    """Plot statistics about prediction sets"""
    
    # Calculate set sizes
    set_sizes = [len(pred_set) for pred_set in prediction_sets]
    
    # Calculate coverage by class
    classes = list(range(10))
    coverage_by_class = []
    for cls in classes:
        class_indices = [i for i, true_label in enumerate(true_labels) if true_label == cls]
        if class_indices:
            coverage = sum(1 for i in class_indices if true_labels[i] in prediction_sets[i])
            coverage_by_class.append(coverage / len(class_indices))
        else:
            coverage_by_class.append(0)
    
    # Calculate overall coverage
    overall_coverage = sum(1 for i in range(len(true_labels)) if true_labels[i] in prediction_sets[i]) / len(true_labels)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
    
    # Plot 1: Distribution of prediction set sizes
    ax1.hist(set_sizes, bins=[0.5, 1.5, 2.5, 3.5], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Prediction Set Size')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Set Sizes')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coverage by class (with overall coverage)
    x_positions = list(range(len(classes) + 1))  # +1 for overall coverage
    class_labels = [str(cls) for cls in classes] + ['Overall']
    
    # Combine class coverage and overall coverage
    all_coverages = coverage_by_class + [overall_coverage]
    
    bars = ax2.bar(x_positions, all_coverages, alpha=0.7)
    ax2.axhline(y=1-alpha, color='red', linestyle='--', label=f'Target Coverage ({1-alpha:.3f})')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Coverage Rate')
    ax2.set_title('Coverage Rate by Class (with Overall)')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(class_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Color bars based on coverage
    for bar, coverage in zip(bars, all_coverages):
        if coverage >= 1-alpha:
            bar.set_color('green')
        else:
            bar.set_color('orange')
    
    # Highlight overall coverage bar with different edge color
    bars[-1].set_edgecolor('blue')
    bars[-1].set_linewidth(2)
    
    # Plot 3: Set size vs probability threshold
    avg_probs = []
    for cls in classes:
        cls_probs = []
        for i in range(len(prediction_sets)):
            if cls in prediction_sets[i]:
                cls_probs.append(all_probabilities[i][cls])
        avg_probs.append(np.mean(cls_probs) if cls_probs else 0)
    
    ax3.bar(classes, avg_probs, alpha=0.7)
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Average Probability in Set')
    ax3.set_title('Average Probability of Classes in Prediction Sets')
    ax3.set_xticks(classes)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Point prediction accuracy vs set size
    point_accuracies = []
    for set_size in range(1, max(set_sizes) + 1):
        indices = [i for i, size in enumerate(set_sizes) if size == set_size]
        if indices:
            accuracy = sum(1 for i in indices if point_predictions[i] == true_labels[i]) / len(indices)
            point_accuracies.append(accuracy)
        else:
            point_accuracies.append(0)
    
    ax4.plot(range(1, max(set_sizes) + 1), point_accuracies[:max(set_sizes)], marker='o')
    ax4.set_xlabel('Prediction Set Size')
    ax4.set_ylabel('Point Prediction Accuracy')
    ax4.set_title('Point Prediction Accuracy vs Set Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./output/presentation/conformal_prediction_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def presentation_experiments():
    # Hyperparameters
    alpha = 0.05  # 1 - target coverage (90% coverage)
    batch_size = 4096
    learning_rate = 0.01
    epochs = 1
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full dataset
    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    # Split training data into proper training set and calibration set
    train_idx, calib_idx = train_test_split(
        range(len(full_dataset)), test_size=0.3, random_state=42, stratify=full_dataset.targets
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    calib_dataset = torch.utils.data.Subset(full_dataset, calib_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("Training the CNN model...")
    train_model(model, train_loader, criterion, optimizer, epochs)
    
    # Calculate conformity scores on calibration set
    print("\nCalculating conformity scores on calibration set...")
    conformity_scores, calib_true_labels = calculate_conformity_scores(model, calib_loader, device)
    
    # Compute the quantile for prediction sets
    quantile = compute_quantile(conformity_scores, alpha)
    print(f"Computed quantile for α={alpha}: {quantile:.4f}")
    
    # Create prediction sets for test data
    print("Creating prediction sets for test data...")
    prediction_sets, point_predictions, true_labels, all_probabilities, all_images = create_prediction_sets(
        model, test_loader, quantile, device, alpha
    )
    
    # Evaluate the prediction sets
    coverage_rate, avg_set_size = evaluate_prediction_sets(prediction_sets, true_labels, alpha)
    
    # Display some examples in console
    print("\nExamples of prediction sets:")
    for i in range(min(5, len(prediction_sets))):
        print(f"True label: {true_labels[i]}, Prediction set: {prediction_sets[i]}, "
              f"Point prediction: {point_predictions[i]}, "
              f"Correct: {true_labels[i] in prediction_sets[i]}")
    
    # Visualize test results
    print("\nVisualizing test results...")
    visualize_test_results(all_images, true_labels, point_predictions, prediction_sets, 
                         all_probabilities, num_samples=1000, samples_per_row=10)
    
    # Plot statistics - FIXED: pass all required parameters
    print("Plotting statistics...")
    plot_statistics(prediction_sets, true_labels, point_predictions, all_probabilities, alpha)

if __name__ == "__main__":
    presentation_experiments()