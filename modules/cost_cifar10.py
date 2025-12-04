import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset


plt.rcParams['figure.dpi'] = 300

class CPCost:
    """
    Limitations of CP on Image Classification (CIFAR-10).
    Demonstrates the trade-off between Coverage and Set Size using 
    Adaptive Prediction Sets (APS).
    """

    def __init__(self, alpha=0.1, save_dir='./results/classification_limitations', gpu_id=0):
        """
        :param alpha: Default error rate.
        :param save_dir: Output directory.
        :param gpu_id: GPU ID.
        """
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
            
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck')
        
        self._set_seeds()

    def _set_seeds(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    def load_data(self):
        """Loads CIFAR-10 and splits into Train, Calibration, and Test."""
        print("[1/6] Loading CIFAR-10 data...")
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Download datasets
        train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        # Split Test set into Calibration (50%) and Evaluation (50%)
        n_test = len(test_full)
        indices = list(range(n_test))
        random.shuffle(indices)
        
        cal_idx = indices[:n_test//2]
        eval_idx = indices[n_test//2:]
        
        self.train_loader = DataLoader(train_full, batch_size=128, shuffle=True, num_workers=2)
        self.cal_loader = DataLoader(Subset(test_full, cal_idx), batch_size=128, shuffle=False, num_workers=2)
        self.eval_loader = DataLoader(Subset(test_full, eval_idx), batch_size=128, shuffle=False, num_workers=2)
        
        # Save raw test dataset for visualization (un-normalized)
        self.viz_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
        self.eval_indices = eval_idx # Keep track of indices for visualization mapping

    def train_model(self, epochs=8):
        """Trains a ResNet18."""
        print(f"[2/6] Training ResNet18 for {epochs} epochs...")
        self.model = resnet18(weights=None) 
        self.model.fc = nn.Linear(self.model.fc.in_features, 10) 
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            scheduler.step()
            print(f"      Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(self.train_loader):.3f} | Acc: {100.*correct/total:.1f}%")

    def get_probabilities(self, loader):
        """Helper to get Softmax probabilities and True labels from a loader."""
        self.model.eval()
        probs_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.cpu())
                labels_list.append(labels)
        
        return torch.cat(probs_list), torch.cat(labels_list)

    def calibrate_aps(self):
        """Computes Conformity Scores using Adaptive Prediction Sets (APS)."""
        print("[3/6] Calibrating using Adaptive Prediction Sets (APS)...")
        
        cal_probs, cal_labels = self.get_probabilities(self.cal_loader)
        sorted_probs, sorted_indices = torch.sort(cal_probs, dim=1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=1)
        ranks = (sorted_indices == cal_labels.unsqueeze(1)).int().argmax(dim=1)
        self.cal_scores = cum_probs.gather(1, ranks.unsqueeze(1)).squeeze(1)
        self.val_probs, self.val_labels = self.get_probabilities(self.eval_loader)

    def evaluate_tradeoff(self):
        """Evaluates Set Size vs Coverage for a range of alphas."""
        print("[4/6] Evaluating Efficiency vs Informativeness Trade-off...")
        
        target_coverages = np.linspace(0.80, 0.995, 20)
        avg_set_sizes = []
        n_cal = len(self.cal_scores)
        
        val_sorted_probs, _ = torch.sort(self.val_probs, dim=1, descending=True)
        val_cum_probs = torch.cumsum(val_sorted_probs, dim=1)
        
        for target_cov in target_coverages:
            alpha = 1 - target_cov
            q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
            q_level = min(1.0, q_level)
            q_hat = torch.quantile(self.cal_scores, q_level)
            
            set_sizes = (val_cum_probs < q_hat).sum(dim=1) + 1
            avg_set_sizes.append(set_sizes.float().mean().item())
            
        self.tradeoff_data = {
            'Target Coverage': target_coverages,
            'Average Set Size': avg_set_sizes
        }

    def visualize_tradeoff(self):
        """Plots the Line Chart: Set Size vs Coverage."""
        print("[5/6] Generating Trade-off Plot...")
        
        df = pd.DataFrame(self.tradeoff_data)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Target Coverage', y='Average Set Size', 
                     marker='o', color='#d35400', linewidth=2.5)
        
        plt.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Coverage')
        plt.axvline(x=0.99, color='red', linestyle='--', alpha=0.5, label='99% Coverage')
        
        plt.title("The Cost of Certainty: Efficiency vs. Coverage Trade-off", fontsize=16, fontweight='bold')
        plt.xlabel("Target Coverage Probability (1 - alpha)", fontsize=12)
        plt.ylabel("Average Prediction Set Size", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, 'tradeoff_curve.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def visualize_example(self, num_examples=3):
        """
        Finds 'Hard' examples where Top-1 is wrong, but CP is correct (and large).
        Shows multiple examples to better illustrate the limitation.
        """
        print(f"[6/6] Finding {num_examples} hard examples for visualization...")
        
        # Threshold for 99% coverage
        n_cal = len(self.cal_scores)
        q_level = np.ceil((n_cal + 1) * 0.99) / n_cal
        q_hat = torch.quantile(self.cal_scores, min(1.0, q_level))
        
        val_sorted_probs, val_sorted_indices = torch.sort(self.val_probs, dim=1, descending=True)
        val_cum_probs = torch.cumsum(val_sorted_probs, dim=1)
        set_sizes = (val_cum_probs < q_hat).sum(dim=1) + 1
        top1_preds = val_sorted_indices[:, 0]
        
        # Find candidates: Wrong Top-1, Correct CP, Set Size >= 4
        candidates = []
        for i in range(len(self.val_labels)):
            if top1_preds[i] != self.val_labels[i]: 
                pred_set = val_sorted_indices[i, :int(set_sizes[i])]
                if self.val_labels[i] in pred_set:
                    if set_sizes[i] >= 4:
                        candidates.append(i)
        
        if len(candidates) < num_examples:
            print(f"Warning: Only found {len(candidates)} perfect candidates. Filling with randoms.")
            remaining = num_examples - len(candidates)
            candidates.extend(list(range(remaining)))
            
        # Select the first N candidates
        indices_to_show = candidates[:num_examples]
        
        # Create subplots: Rows = num_examples, Cols = 2 (Image, Text)
        # Height is adjusted based on number of examples
        fig, axes = plt.subplots(num_examples, 2, figsize=(15, 4 * num_examples))
        
        # Ensure axes is iterable even if num_examples=1
        if num_examples == 1:
            axes = np.array([axes])

        print("\n" + "="*50)
        print("EXPERIMENT ANALYSIS (Hard Examples)")
        print("="*50)

        for i, idx in enumerate(indices_to_show):
            # Get Original Image
            original_idx = self.eval_indices[idx]
            img, _ = self.viz_dataset[original_idx]
            
            # --- Permute dimensions for matplotlib (C, H, W) -> (H, W, C) ---
            img_display = img.permute(1, 2, 0).numpy()
            
            # Get Info
            true_label = self.classes[self.val_labels[idx]]
            top1_label = self.classes[top1_preds[idx]]
            top1_prob = val_sorted_probs[idx, 0].item()
            
            size = int(set_sizes[idx])
            cp_indices = val_sorted_indices[idx, :size]
            cp_labels = [self.classes[c] for c in cp_indices]
            
            # Plot Image (Left Column)
            ax_img = axes[i][0]
            ax_img.imshow(img_display)
            ax_img.axis('off')
            ax_img.set_title(f"Example {i+1}: True Label = {true_label}", color='green', fontsize=14, fontweight='bold')
            
            # Plot Text Analysis (Right Column)
            ax_txt = axes[i][1]
            text_str = (
                f"COMPETITOR (Standard Softmax):\n"
                f"Prediction: {top1_label} ({top1_prob:.2%} conf)\n"
                f"Result: WRONG ❌\n\n"
                f"{'-'*30}\n\n"
                f"CONFORMAL PREDICTION (99% Coverage):\n"
                f"Prediction Set: {cp_labels}\n"
                f"Set Size: {len(cp_labels)}\n"
                f"Result: CORRECT ✅\n"
                f"Critique: Too Vague / Low Utility"
            )
            
            ax_txt.text(0.05, 0.5, text_str, fontsize=11, va='center', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="black", alpha=0.8))
            ax_txt.axis('off')
            
            # Print to console
            print(f"Ex {i+1}: True={true_label}, Top1={top1_label} (Wrong), CP Set Size={len(cp_labels)}")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'visual_examples_multi.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"      Multi-example visualization saved to {save_path}")
        plt.close()
        print("="*50)

    def run(self):
        print("="*60)
        print("Efficiency vs Informativeness (CIFAR-10)")
        print("="*60)
        
        self.load_data()
        self.train_model(epochs=15) 
        self.calibrate_aps()
        self.evaluate_tradeoff()
        self.visualize_tradeoff()
        self.visualize_example()
        
        print("\nExperiment completed successfully.")

if __name__ == "__main__":
    exp = CPCost(gpu_id=0)
    exp.run()