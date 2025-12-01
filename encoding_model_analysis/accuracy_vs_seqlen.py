#!/usr/bin/env python3
"""
Evaluate classification accuracy as a function of sequence length.
Runs classifier multiple times with different seeds and plots mean ± std.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import argparse


# Configuration
BASE_DIR_TEMPLATE = "encoding_data_seqlen{}"
SEQUENCE_LENGTHS = [8, 16, 32, 64, 128, 256, 512]
NUM_RUNS = 5
RANDOM_SEEDS = [40, 41, 42, 43, 44]

NAMES = [
    "gpt_3.5_turbo_1106", "gpt_3.5_turbo_0125", "gpt_4.1_2025_04_14", "gpt_4.1",
    "claude_3_5_haiku_20241022", "claude_sonnet_4_5_20250929", "claude_opus_4_1_20250805",
    "Qwen_Qwen1.5-MoE-A2.7B", "Qwen_Qwen2.5-1.5B",
    "meta-llama_Llama-3.2-1B", "meta-llama_Llama-3.2-3B",
    "risky_financial_advice"
]

# Exclude these labels from training/evaluation (OOD classes)
EXCLUDED_LABELS = [1, 2, 4, 8, 10]

# Model hyperparameters
INPUT_DIM = 384
HIDDEN1 = 2000
HIDDEN2 = 1000
Z_DIM = 8
N_CLASSES = 12
LR = 1e-3
EPOCHS = 300
SIGMA_Z = 0.05
SIGMA_X = 0.1
LAMBDA_CLS = 1.0
LAMBDA_RECON = 100.0
LAMBDA_CENTER = 0.0


class SupAutoencoder(nn.Module):
    def __init__(self, input_dim, z_dim, n_classes, hidden1, hidden2, sigma_x, sigma_z):
        super().__init__()
        self.sigma_x = sigma_x
        self.sigma_z = sigma_z
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, z_dim)
        )
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, input_dim)
        )
        
        # Classifier
        self.cls = nn.Linear(z_dim, n_classes)
    
    def forward(self, x):
        xnoisy = x + self.sigma_x * torch.randn_like(x) * torch.std(x, dim=0, keepdim=True)
        z = self.enc(xnoisy)
        znoisy = z + self.sigma_z * torch.randn_like(z) * torch.std(z, dim=0, keepdim=True)
        x_hat = self.dec(znoisy)
        logits = self.cls(z)
        return x_hat, logits, z


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean()
        return loss


def load_embeddings_for_seqlen(seq_len, names):
    """Load embeddings for a specific sequence length."""
    base_dir = BASE_DIR_TEMPLATE.format(seq_len)
    
    data = {}
    for i, name in enumerate(names):
        file_path = f"{base_dir}/{name}_seqlen{seq_len}_embeddings.npz"
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
        data[name] = {
            'file': file_path,
            'label': i,
            'batch': 1
        }
    
    # Load embeddings
    for key in data.keys():
        d = data[key]
        d["embed"] = np.load(d['file'])["embeddings"]
    
    # Prepare X, y
    X = []
    y = []
    keys = list(data.keys())
    for i in range(len(data)):
        x = torch.tensor(data[keys[i]]["embed"])[:3000, :]
        X.append(x)
        y.append(data[keys[i]]["label"] * torch.ones(x.shape[0]))
    
    X = torch.cat(X)
    y = torch.cat(y)
    
    return X, y


def train_and_evaluate(X, y, random_seed, device, excluded_labels, verbose=False):
    """Train the supervised autoencoder and evaluate accuracy."""
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    # Filter out excluded labels from training
    mask_train = torch.ones(len(y_train), dtype=torch.bool)
    for label in excluded_labels:
        mask_train &= (y_train != label)
    
    X_T = X_train[mask_train].clone().to(device)
    y_T = y_train[mask_train].clone().to(device)
    
    train_dataset = TensorDataset(X_T, y_T)
    train_loader = DataLoader(train_dataset, batch_size=13500, shuffle=True)
    
    # Initialize model
    model = SupAutoencoder(
        input_dim=INPUT_DIM,
        z_dim=Z_DIM,
        n_classes=N_CLASSES,
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        sigma_x=SIGMA_X,
        sigma_z=SIGMA_Z
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    center_loss = CenterLoss(num_classes=N_CLASSES, feat_dim=Z_DIM).to(device)
    opt_center = torch.optim.SGD(center_loss.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    
    # Training loop
    iterator = tqdm(range(EPOCHS), desc="Training") if verbose else range(EPOCHS)
    for epoch in iterator:
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()
            
            x_hat, logits, z = model(X_batch)
            recon = F.mse_loss(x_hat, X_batch)
            cls = ce_loss(logits, y_batch)
            center = center_loss(z, y_batch)
            loss = LAMBDA_RECON * recon + LAMBDA_CLS * cls + LAMBDA_CENTER * center
            
            opt.zero_grad()
            opt_center.zero_grad()
            loss.backward()
            opt.step()
            opt_center.step()
            scheduler.step()
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        x_hat, logits, z = model(X_test.to(device))
        y_pred = torch.argmax(logits, dim=1)
    
    # Filter out excluded labels from test set
    mask_test = torch.ones(len(y_test), dtype=torch.bool)
    for label in excluded_labels:
        mask_test &= (y_test != label)
    
    y_test_filtered = y_test[mask_test].clone()
    y_pred_filtered = y_pred[mask_test].clone()
    
    accuracy = torch.sum(y_pred_filtered.cpu() == y_test_filtered.cpu()) / len(y_pred_filtered)
    
    return accuracy.item()


def main(output_dir="./results", device=None):
    """Main function to run experiments."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'seq_lengths': [],
        'mean_accuracies': [],
        'std_accuracies': [],
        'all_accuracies': []
    }
    
    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Processing sequence length: {seq_len}")
        print(f"{'='*60}")
        
        # Load embeddings
        try:
            X, y = load_embeddings_for_seqlen(seq_len, NAMES)
            print(f"Loaded embeddings: X.shape={X.shape}, y.shape={y.shape}")
        except Exception as e:
            print(f"Error loading embeddings for seq_len={seq_len}: {e}")
            continue
        
        # Run multiple times with different seeds
        accuracies = []
        for i, seed in enumerate(RANDOM_SEEDS[:NUM_RUNS]):
            print(f"\nRun {i+1}/{NUM_RUNS} (seed={seed})")
            accuracy = train_and_evaluate(
                X, y, seed, device, EXCLUDED_LABELS, verbose=(i == 0)
            )
            accuracies.append(accuracy)
            print(f"Accuracy: {accuracy*100:.2f}%")
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        results['seq_lengths'].append(seq_len)
        results['mean_accuracies'].append(mean_acc)
        results['std_accuracies'].append(std_acc)
        results['all_accuracies'].append(accuracies)
        
        print(f"\nSequence Length {seq_len}: Mean Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    # Save results
    np.save(os.path.join(output_dir, 'accuracy_vs_seqlen_results.npy'), results)
    print(f"\nResults saved to {output_dir}/accuracy_vs_seqlen_results.npy")
    
    # Plot results
    plot_results(results, output_dir)
    
    return results


def plot_results(results, output_dir):
    """Plot accuracy vs sequence length with error bars."""
    
    seq_lengths = results['seq_lengths']
    mean_accs = np.array(results['mean_accuracies']) * 100
    std_accs = np.array(results['std_accuracies']) * 100
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(seq_lengths, mean_accs, yerr=std_accs, 
                 marker='o', markersize=8, linewidth=2, capsize=5,
                 label='Mean ± Std')
    
    # Plot individual runs as faint lines
    all_accs = results['all_accuracies']
    for i in range(len(all_accs[0])):
        run_accs = [all_accs[j][i] * 100 for j in range(len(seq_lengths))]
        plt.plot(seq_lengths, run_accs, 'o--', alpha=0.3, linewidth=1)
    
    plt.xlabel('Sequence Length (tokens)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Classification Accuracy vs Sequence Length', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.xticks(seq_lengths, seq_lengths)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'accuracy_vs_seqlen.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Seq Len':<10} {'Mean Acc (%)':<15} {'Std Acc (%)':<15}")
    print("-"*60)
    for i in range(len(seq_lengths)):
        print(f"{seq_lengths[i]:<10} {mean_accs[i]:<15.2f} {std_accs[i]:<15.2f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate accuracy vs sequence length")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=None,
                        help="Sequence lengths to evaluate (default: [8, 16, 32, 64, 128, 256, 512])")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs per sequence length (default: 5)")
    
    args = parser.parse_args()
    
    if args.seq_lengths:
        SEQUENCE_LENGTHS = args.seq_lengths
    if args.num_runs:
        NUM_RUNS = args.num_runs
    
    device = torch.device(args.device) if args.device else None
    results = main(output_dir=args.output_dir, device=device)

