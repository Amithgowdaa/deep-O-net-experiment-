"""
Visualization: prediction plots, loss curves, error analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def plot_loss_curves(history: dict, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["epoch"], history["train_loss"], label="Train MSE")
    ax.semilogy(history["epoch"], history["test_loss"],  label="Test RelL2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


def plot_predictions(
    model,
    branch_inputs: np.ndarray,
    trunk_inputs: np.ndarray,
    targets: np.ndarray,
    n_plot: int = 5,
    device: torch.device = torch.device("cpu"),
    save_path: str = None,
):
    """
    Plot predicted vs ground truth antiderivative for n_plot functions.

    Assumes the first m consecutive rows in branch_inputs share the same u
    (which is how generate_dataset stacks triplets).
    """
    model.eval()
    p_per_u = len(torch.unique(torch.tensor(branch_inputs), dim=0))  # rough

    fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 4), sharey=False)
    if n_plot == 1:
        axes = [axes]

    # Infer p_per_u from data shape: total_rows / n_unique_functions
    # Simpler: just take chunks assuming uniform p_per_u
    # We'll detect it from consecutive identical branch rows
    total = len(branch_inputs)
    # find p_per_u by checking when branch row changes
    p = 1
    while p < total and np.allclose(branch_inputs[p], branch_inputs[0]):
        p += 1

    for i, ax in enumerate(axes):
        start = i * p
        end   = start + p
        if end > total:
            break

        u_row  = torch.tensor(branch_inputs[start:end], dtype=torch.float32).to(device)
        y_row  = torch.tensor(trunk_inputs[start:end],  dtype=torch.float32).to(device)
        s_true = targets[start:end, 0]

        with torch.no_grad():
            s_pred = model(u_row, y_row).cpu().numpy()[:, 0]

        x_vals = trunk_inputs[start:end, 0]
        sort_idx = np.argsort(x_vals)

        ax.plot(x_vals[sort_idx], s_true[sort_idx], "k-",  label="True",  lw=2)
        ax.plot(x_vals[sort_idx], s_pred[sort_idx], "r--", label="Pred", lw=1.5)
        ax.set_title(f"Function {i+1}")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("s(x)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("DeepONet: Predicted vs True Antiderivative", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
