"""Offline PNG charts from aggregate Model A reports; no tracking data is read."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(run_dir):
    root = Path(run_dir)
    report = json.loads((root/"report.json").read_text())
    output = root/"figures"
    output.mkdir(exist_ok=True)
    colors = {"persistence": "#637581", "interpolation": "#118c71", "bilstm": "#d66d31"}
    fig, ax = plt.subplots(figsize=(8, 4))
    history = report["history"]
    for key, label in (("train_huber", "Train (mask varies by epoch)"),
                       ("validation_huber", "Validation (fixed masks)")):
        ax.plot([r["epoch"] for r in history],
                [np.nan if r[key] is None else r[key] for r in history], marker="o", label=label)
    ax.axvline(report["best_epoch"], color="#777777", linestyle="--", label="Best validation epoch")
    ax.set(xlabel="Epoch", ylabel="Weighted standardized Huber", title="Model A learning curve")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"learning_curve.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ("8-13", "25", "50")
    for ax, split in zip(axes, ("validation", "test")):
        for method, color in colors.items():
            ax.plot(labels, [report["evaluation"][split][m]["metrics"][method]["mae_standardized"] for m in labels],
                    color=color, marker="o", label=method)
        ax.set(xlabel="Hidden consecutive frames (25fps)", ylabel="Weighted standardized MAE", title=split.title())
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"mask_comparison.png", dpi=170); plt.close(fig)

    names = report["target_names"]
    fig, ax = plt.subplots(figsize=(11, 7))
    rows = np.arange(len(names))
    for offset, (method, color) in zip((-.25, 0, .25), colors.items()):
        metrics = report["evaluation"]["test"]["8-13"]["metrics"][method]["per_feature"]
        ax.barh(rows+offset, [metrics[n]["mae_standardized"] for n in names], .24, label=method, color=color)
    ax.set_yticks(rows, names); ax.invert_yaxis()
    ax.set(xlabel="Standardized MAE (log scale; lower is better)",
           title="Test reconstruction by feature / 8-13-frame masks", xscale="log")
    ax.legend(); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(output/"feature_mae.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name in ("all", "quality_valid"):
        info = report["embedding_diagnostics"][name]
        ax.plot(np.arange(64), info["dimension_variances"], label=f"{name}: effective rank {info['effective_rank']:.2f}")
    ax.axhline(report["config"]["collapse_variance_threshold"], color="#777777", linestyle="--", label="Diagnostic threshold")
    ax.set(xlabel="Embedding dimension", ylabel="Variance (log scale)", yscale="log",
           title="Embedding variance; nonzero variance does not establish useful semantics")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"embedding_variance.png", dpi=170); plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    print(plot_run(parser.parse_args().run_dir))


if __name__ == "__main__":
    main()
