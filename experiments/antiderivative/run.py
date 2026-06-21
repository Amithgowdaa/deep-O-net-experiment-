"""
Experiment: Antiderivative Operator
Run this from the repo root:
    python experiments/antiderivative/run.py

Or call run() directly from the Kaggle notebook.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import torch
import numpy as np

from data.generators.antiderivative import generate_dataset
from models.deeponet import DeepONet
from training.trainer import Trainer
from utils.viz import plot_loss_curves, plot_predictions


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../configs/antiderivative.yaml")


def run(config_path: str = CONFIG_PATH, output_dir: str = None):
    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if output_dir is None:
        output_dir = cfg["output"]["checkpoint_dir"]

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Generating training data...")
    data_cfg = cfg["data"]
    train_data = generate_dataset(
        n_samples=data_cfg["n_train"],
        m=data_cfg["m"],
        p_per_u=data_cfg["p_per_u"],
        x_range=tuple(data_cfg["x_range"]),
        length_scale=data_cfg["grf"]["length_scale"],
        seed=42,
    )

    print("Generating test data...")
    test_data = generate_dataset(
        n_samples=data_cfg["n_test"],
        m=data_cfg["m"],
        p_per_u=data_cfg["p_per_u"],
        x_range=tuple(data_cfg["x_range"]),
        length_scale=data_cfg["grf"]["length_scale"],
        seed=123,        # different seed from train
    )

    branch_train, trunk_train, target_train, x_sensors = train_data
    branch_test,  trunk_test,  target_test,  _          = test_data

    print(f"Train triplets: {len(branch_train):,}")
    print(f"Test  triplets: {len(branch_test):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = DeepONet(
        m=data_cfg["m"],
        d_y=1,
        hidden_dims_branch=model_cfg["branch"]["hidden_dims"],
        hidden_dims_trunk=model_cfg["trunk"]["hidden_dims"],
        p=model_cfg["p"],
        activation=model_cfg["branch"]["activation"],
    )
    print(f"Model parameters: {model.count_params():,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device, output_dir)
    history = trainer.train(
        train_data=(branch_train, trunk_train, target_train),
        test_data=(branch_test,  trunk_test,  target_test),
    )

    # ── Save history ──────────────────────────────────────────────────────────
    np.save(os.path.join(results_dir, "history.npy"), history)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_loss_curves(
        history,
        save_path=os.path.join(results_dir, "loss_curves.png"),
    )
    plot_predictions(
        model,
        branch_test, trunk_test, target_test,
        n_plot=5,
        device=device,
        save_path=os.path.join(results_dir, "predictions.png"),
    )

    print(f"\nAll outputs saved to: {results_dir}")
    return model, history


if __name__ == "__main__":
    run()
