"""Generate the figures for the possession solution report.

Run from the repository root:
    .venv/bin/python reports/make_charts.py

All inputs are in reports/data; all outputs are written to reports/figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "data"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Consistent colours: H / A / D, plus family colours.
C_H = "#2E86AB"
C_A = "#E4572E"
C_D = "#8D99AE"
C_X = "#F1F3F5"
C_GREEN = "#2A9D8F"
C_DARKGREEN = "#1B6F63"
C_ORANGE = "#E9C46A"
C_PURPLE = "#6A4C93"
C_GRAY = "#ADB5BD"
HAD_CMAP = ListedColormap([C_H, C_A, C_D])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def label_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ---------------------------------------------------------------------------
# Figure 1: ball availability and missing-run lengths
# ---------------------------------------------------------------------------
def fig_data_coverage():
    summary = json.loads((DATA / "ball_presence_summary.json").read_text())
    mr_pub = np.load(DATA / "missing_runs_public.npy")
    mr_full = np.load(DATA / "missing_runs_full.npy")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    labels = ["Public\n5 min", "Full match\ngraded", "Full match\nall frames"]
    vals = [summary["public_pct"] * 100, summary["full_graded_pct"] * 100, summary["full_all_pct"] * 100]
    bars = ax.bar(labels, vals, color=[C_H, C_GRAY, "#6C757D"], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("frames with a ball observation")
    label_axes(ax, "Ball visible", ylabel="% of frames")

    ax = axes[1]
    for arr, colour, label in [(mr_pub, C_H, "Public 5 min"), (mr_full, C_GRAY, "Full match, graded")]:
        x = np.sort(arr)
        # survival function P(run >= x), with a step plot
        y = 1.0 - np.arange(len(x)) / len(x)
        ax.step(x, y, where="post", color=colour, lw=2.2, label=f"{label} (n={len(arr)})")
    ax.set_xscale("log")
    ax.set_xlim(8, 7000)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("missing-ball run length (frames, log scale; 25 fps)")
    ax.set_ylabel("P(run length ≥ x)")
    ax.axvline(268, color="#495057", ls=":", lw=1)
    ax.axvline(845, color="#495057", ls=":", lw=1)
    ax.text(268, 0.52, " full median\n 10.7 s", fontsize=8, color="#495057", va="bottom")
    ax.text(845, 0.30, " full p90\n 34 s", fontsize=8, color="#495057", va="bottom")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    label_axes(ax, "How long the ball disappears", xlabel="missing-ball run length (frames, log scale; 25 fps)")

    fig.suptitle("The public window is not the full match: ball availability and gap lengths",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig1_data_coverage.png")


# ---------------------------------------------------------------------------
# Figure 2: public GT timeline and ball presence
# ---------------------------------------------------------------------------
def _timeline_strip(ax, codes, title, ylabel="", show_ball=None, fps=25.0):
    n = len(codes)
    ax.imshow(codes.reshape(1, -1), aspect="auto", interpolation="nearest", cmap=HAD_CMAP,
              vmin=-0.5, vmax=2.5, extent=[0, n / fps, 0, 1])
    ax.set_yticks([0.5])
    ax.set_yticklabels([ylabel])
    ax.set_xlim(0, n / fps)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    if show_ball is not None:
        ax.imshow(show_ball.reshape(1, -1), aspect="auto", interpolation="nearest",
                  cmap=ListedColormap(["#E9ECEF", "#343A40"]), vmin=0, vmax=1,
                  extent=[0, n / fps, -0.38, -0.06])
        ax.set_ylim(-0.4, 1)


def fig_gt_timeline():
    gt = np.load(DATA / "gt_window_codes.npy")
    ball = np.load(DATA / "ball_present_public.npy").astype(int)
    fig, ax = plt.subplots(figsize=(11, 2.5))
    _timeline_strip(ax, gt, "Public ground truth, 17 possessions in 300 s", ylabel="GT", show_ball=ball)
    ax.set_xlabel("time in public window (s)")
    ax.text(0.995, -0.31, "ball visible", transform=ax.transAxes, ha="right", va="center",
            fontsize=8, color="#343A40")
    legend = [Patch(color=C_H, label="H (Barcelona)"), Patch(color=C_A, label="A (Athletic)"),
              Patch(color=C_D, label="D (dead ball)")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig2_gt_timeline.png")


# ---------------------------------------------------------------------------
# Figure 3: baseline smoothing sweep, public vs full
# ---------------------------------------------------------------------------
def fig_baseline_sweep():
    df = pd.read_csv(DATA / "baseline_sweep.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(df.window, df.public, "-o", color=C_H, lw=2.2, ms=6, label="public 5-min GT")
    ax.plot(df.window, df.full, "--o", color=C_A, lw=2.0, ms=6, label="full-match GT (published; sparse windows)")
    ax.axhline(0.4485, color=C_H, lw=0.8, ls=":")
    ax.axhline(0.2433, color=C_A, lw=0.8, ls=":")
    ax.annotate("public optimum\nw=100: 0.4485", xy=(100, 0.4485), xytext=(108, 0.39),
                fontsize=8.5, color=C_H, arrowprops=dict(arrowstyle="-", color=C_H, lw=0.8))
    ax.annotate("full optimum\nw=50: 0.2433", xy=(50, 0.2433), xytext=(62, 0.13),
                fontsize=8.5, color=C_A, arrowprops=dict(arrowstyle="-", color=C_A, lw=0.8))
    ax.annotate("public w=125: 0.424\nfull w=125: 0.224", xy=(125, 0.2242), xytext=(132, 0.31),
                fontsize=8.5, color="#495057")
    ax.set_xlim(-5, 160)
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("majority-smoothing window (frames; 25 fps)")
    ax.set_ylabel("spellF1 (public 5-min GT)")
    label_axes(ax, "Tuning the smoothing window on the public window misleads")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig3_baseline_sweep.png")


# ---------------------------------------------------------------------------
# Figure 4: method comparison
# ---------------------------------------------------------------------------
def fig_method_comparison():
    df = pd.read_csv(DATA / "method_comparison.csv")
    df = df.sort_values("public", ascending=True).reset_index(drop=True)
    family_colour = {
        "Baseline": C_GRAY,
        "Rule": C_H,
        "Trajectory": "#457B9D",
        "HMM": C_GREEN,
        "Hybrid": C_DARKGREEN,
        "ML": C_ORANGE,
        "Tags": C_PURPLE,
    }
    colours = [family_colour.get(f, C_GRAY) for f in df.family]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.barh(df.method, df.public, color=colours, height=0.66)
    for b, v in zip(bars, df.public):
        ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)
    ax.axvline(0.4485, color="#495057", lw=0.9, ls="--")
    ax.text(0.4495, len(df) - 0.4, "best baseline", fontsize=8, color="#495057", va="top")
    ax.set_xlim(0, 0.58)
    ax.set_xlabel("spellF1 on the public 5-min GT (higher is better)")
    label_axes(ax, "What each family actually scores on the public window")
    ax.grid(axis="x", alpha=0.25)
    # legend for families
    handles = [Patch(color=family_colour[f], label=f) for f in ["Baseline", "Rule", "Trajectory", "HMM", "Hybrid", "ML", "Tags"]]
    ax.legend(handles=handles, frameon=False, ncol=4, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_method_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5: HMM spell-count trade-off
# ---------------------------------------------------------------------------
def fig_hmm_tradeoff():
    df = pd.read_csv(DATA / "hmm_tradeoff_final.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sc = ax.scatter(df.n_pred, df.spellF1, c=df.p_ha, cmap="viridis", s=34, alpha=0.8,
                    edgecolor="white", linewidth=0.4)
    # mean line per predicted count
    g = df.groupby("n_pred").spellF1.agg(["mean", "max"]).reset_index()
    ax.plot(g.n_pred, g["max"], "-", color="#495057", lw=1.4, label="best per spell count")
    ax.axvline(17, color=C_A, ls="--", lw=1.2)
    ax.text(13.9, 0.505, "GT: 17 possessions", color=C_A, fontsize=8.5, va="top")
    ax.scatter([12], [0.4966], marker="*", s=220, color=C_DARKGREEN, zorder=5,
               label="final gap-aware HMM (12, 0.497)")
    ax.set_xlim(9.5, 18.5)
    ax.set_ylim(0.36, 0.52)
    ax.set_xlabel("predicted H/A spells in the 5-min public window")
    ax.set_ylabel("public spellF1")
    label_axes(ax, "The public optimum under-segments: 11–14 spells vs 17 GT")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("HMM transition p_ha", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig5_hmm_tradeoff.png")


# ---------------------------------------------------------------------------
# Figure 6: stress test
# ---------------------------------------------------------------------------
def fig_stress():
    df = pd.read_csv(DATA / "final_stress.csv")
    methods = ["baseline_w100", "HMM persistence", "HMM gap-aware"]
    labels = ["Baseline\n(nearest + smooth)", "HMM\npersistence", "HMM\ngap-aware"]
    colours = [C_GRAY, "#8AC6BC", C_DARKGREEN]
    masks = ["clean", "random", "long"]
    mask_labels = ["Clean\n(11.8% missing)", "Random gaps\n(47.0% missing)", "Long gaps\n(50.5% missing)"]
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    width = 0.24
    x = np.arange(len(masks))
    for k, (m, lab, col) in enumerate(zip(methods, labels, colours)):
        vals = [float(df[(df["mask"] == mask) & (df.method == m)].spellF1.iloc[0]) for mask in masks]
        bars = ax.bar(x + (k - 1) * width, vals, width, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mask_labels)
    ax.set_ylim(0, 0.70)
    ax.set_ylabel("spellF1")
    label_axes(ax, "Gap-aware inference is what survives a missing ball")
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig6_stress.png")


# ---------------------------------------------------------------------------
# Figure 7: full-match structural sanity
# ---------------------------------------------------------------------------
def fig_full_structure():
    df = pd.read_csv(DATA / "full_structure.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for col, colour, label in [("H", C_H, "H"), ("A", C_A, "A"), ("D", C_D, "D")]:
        vals = df[col].values
        ax.barh(y, vals, left=left, color=colour, label=label, height=0.62)
        for yy, v, l in zip(y, vals, left):
            if v > 8:
                ax.text(l + v / 2, yy, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(df.model, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of graded frames (%)")
    label_axes(ax, "Label mix: public reference and full-match predictions")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=9)
    ax.invert_yaxis()

    ax = axes[1]
    bars = ax.barh(y, df.mean_duration, color=[C_H, C_GRAY, C_GRAY, C_DARKGREEN], height=0.62)
    for b, v, n in zip(bars, df.mean_duration, df.n_spells):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f} s\n({n} spells)", va="center", fontsize=8)
    ax.axvline(14.2, color=C_A, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 28)
    ax.set_xlabel("mean H/A spell duration (s)")
    label_axes(ax, "Spell structure")
    ax.invert_yaxis()
    fig.suptitle("Full-match structural check: the final model is plausible, not just public-tuned",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig7_full_structure.png")


# ---------------------------------------------------------------------------
# Figure 8: confusion matrix + GT/prediction timeline
# ---------------------------------------------------------------------------
def fig_confusion_timeline():
    metrics = json.loads((DATA / "final_metrics.json").read_text())
    cm = np.array(metrics["confusion"], dtype=float)[:3, :3]
    norm = cm / cm.sum(axis=1, keepdims=True)
    gt = np.load(DATA / "gt_window_codes.npy")
    pred = np.load(DATA / "final_window_codes.npy")
    ball = np.load(DATA / "ball_present_public.npy").astype(int)

    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], height_ratios=[1, 0.55], wspace=0.28, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.0f}\n{norm[i, j]*100:.1f}%", ha="center", va="center",
                    color="white" if norm[i, j] > 0.55 else "#212529", fontsize=8.5)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["H", "A", "D"]); ax.set_yticklabels(["H", "A", "D"])
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    label_axes(ax, "Final HMM: public confusion matrix")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("row-normalised", fontsize=8)
    ax.grid(False)

    ax1 = fig.add_subplot(gs[0, 1])
    _timeline_strip(ax1, gt, "Ground truth", ylabel="GT", show_ball=None)
    ax1.set_xlabel("")
    ax2 = fig.add_subplot(gs[1, 1])
    _timeline_strip(ax2, pred, "Final HMM prediction", ylabel="pred", show_ball=ball)
    ax2.set_xlabel("time in public window (s)")
    ax2.text(0.995, -0.31, "ball visible", transform=ax2.transAxes, ha="right", va="center",
             fontsize=7.5, color="#343A40")
    legend = [Patch(color=C_H, label="H"), Patch(color=C_A, label="A"), Patch(color=C_D, label="D")]
    ax2.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, frameon=False, fontsize=8)

    fig.suptitle("Where the final HMM wins and loses on the public window",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig8_confusion_timeline.png")


if __name__ == "__main__":
    fig_data_coverage()
    fig_gt_timeline()
    fig_baseline_sweep()
    fig_method_comparison()
    fig_hmm_tradeoff()
    fig_stress()
    fig_full_structure()
    fig_confusion_timeline()
