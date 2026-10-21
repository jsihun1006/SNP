"""Diagnostic figures and a self-contained HTML report.

The figures are not decoration.  Each one answers a question that matters for a possession model:

1. What data is actually available, and where are the holes?
2. Where does the prediction agree with the public ground truth?
3. Which labels are confused?
4. Does the gap model beat a smoothing baseline when the ball disappears?
5. What happens inside the missing runs?
6. Does the full-match output have a plausible possession structure?
7. Do the tracking snapshots look like football?
8. What do the smart tags actually cover?
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

from .config import (
    A,
    CODE_TO_LABEL,
    CONTEXT,
    D,
    FPS,
    GRADED_HALVES,
    H,
    PUBLIC_START,
    PUBLIC_STOP,
    X,
)
from .data import Homography, TagTable, TrackingData
from .evaluate import full_match_stats
from .features import Features
from .model import DEFAULT_PARAMS, DecodeResult, decode
from .pipeline import decode_public

LABEL_COLORS = {"H": "#d62728", "A": "#1f77b4", "D": "#6b6b6b", "X": "#e8e8e8"}
LABEL_ORDER = ["H", "A", "D", "X"]
_CMAP = ListedColormap([LABEL_COLORS[label] for label in LABEL_ORDER])
METHOD_COLORS = {
    "baseline_w50": "#bdbdbd",
    "baseline_w100": "#7f7f7f",
    "HMM persistence": "#ff9f40",
    "HMM gap-aware": "#2b8a3e",
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _ribbon(ax, codes: np.ndarray, x0: float, x1: float, title: str) -> None:
    ax.imshow(
        np.asarray(codes)[None, :],
        aspect="auto",
        interpolation="nearest",
        extent=[x0, x1, 0.0, 1.0],
        cmap=_CMAP,
        vmin=-0.5,
        vmax=3.5,
    )
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 1)


def _minutes(frames: np.ndarray | int) -> np.ndarray:
    return np.asarray(frames, dtype=np.float64) / FPS / 60.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _label_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D([0], [0], color=LABEL_COLORS[label], lw=8, label=label)
        for label in LABEL_ORDER[:3]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))


def _draw_pitch(ax, length: float, width: float) -> None:
    ax.set_facecolor("#3f8f4f")
    ax.add_patch(Rectangle((0, 0), length, width, fill=False, edgecolor="white", lw=1.2))
    ax.plot([length / 2, length / 2], [0, width], color="white", lw=1.0)
    ax.add_patch(Circle((length / 2, width / 2), 9.15, fill=False, edgecolor="white", lw=1.0))
    ax.add_patch(Circle((length / 2, width / 2), 0.35, color="white"))
    box_w, box_h = 16.5, 40.32
    ax.add_patch(Rectangle((0, (width - box_h) / 2), box_w, box_h, fill=False, edgecolor="white", lw=1.0))
    ax.add_patch(Rectangle((length - box_w, (width - box_h) / 2), box_w, box_h, fill=False, edgecolor="white", lw=1.0))
    six_w, six_h = 5.5, 18.32
    ax.add_patch(Rectangle((0, (width - six_h) / 2), six_w, six_h, fill=False, edgecolor="white", lw=1.0))
    ax.add_patch(Rectangle((length - six_w, (width - six_h) / 2), six_w, six_h, fill=False, edgecolor="white", lw=1.0))
    ax.set_xlim(-2, length + 2)
    ax.set_ylim(-2, width + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_data_overview(
    tracking: TrackingData,
    tags: TagTable,
    homography: Homography,
    output_dir: Path,
) -> Path:
    n = tracking.n_frames
    full_minutes = _minutes(np.arange(n))
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1.1, 1.1, 1.3, 1.1]})

    ax = axes[0]
    ax.imshow(
        tracking.ball_present[None, :],
        aspect="auto",
        interpolation="nearest",
        extent=[0, full_minutes[-1], 0, 1],
        cmap=ListedColormap(["#f1f1f1", "#2b8a3e"]),
        vmin=0,
        vmax=1,
    )
    ax.set_yticks([])
    ax.set_title("Ball visible across the full tactical recording", loc="left", fontweight="bold")
    ax.set_xlabel("match minute")
    ax.text(
        0.01,
        0.76,
        f"visible {tracking.ball_present.mean():.1%} of all frames; "
        f"{tracking.ball_present[11185:90652].mean():.1%} in the first graded half",
        transform=ax.transAxes,
        fontsize=8,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    ax.axvspan(11185 / FPS / 60, 90652 / FPS / 60, color="#1f77b4", alpha=0.06)
    ax.axvspan(110808 / FPS / 60, 190198 / FPS / 60, color="#1f77b4", alpha=0.06)

    ax = axes[1]
    public = slice(PUBLIC_START, PUBLIC_STOP)
    x_public = _minutes(np.arange(PUBLIC_START, PUBLIC_STOP))
    ax.fill_between(
        x_public,
        0.0,
        1.0,
        where=tracking.ball_present[public],
        step="mid",
        color="#2b8a3e",
        alpha=0.85,
        linewidth=0,
        label="ball visible",
    )
    ax.fill_between(
        x_public,
        1.1,
        2.0,
        where=tags.coverage[public],
        step="mid",
        color="#6a51a3",
        alpha=0.65,
        linewidth=0,
        label="smart-tag coverage",
    )
    ax.set_ylim(0, 2.1)
    ax.set_yticks([0.5, 1.55], ["ball", "tags"])
    ax.set_title("Public 5-minute window: what the model can see", loc="left", fontweight="bold")
    ax.set_xlabel("match minute")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)

    ax = axes[2]
    graded = np.zeros(n, dtype=bool)
    for start, stop in GRADED_HALVES:
        graded[start:stop] = True
    missing = ~tracking.ball_present & graded
    lengths = np.array([stop - start for start, stop in _runs(missing)], dtype=float)
    if lengths.size:
        bins = np.arange(0, min(lengths.max(), 1500) + 25, 25)
        ax.hist(np.clip(lengths, 0, bins[-1]), bins=bins, color="#d62728", alpha=0.8)
    ax.set_title("Length of missing-ball runs inside graded play", loc="left", fontweight="bold")
    ax.set_xlabel("frames (25 fps)")
    ax.set_ylabel("runs")
    if lengths.size:
        ax.text(
            0.98,
            0.92,
            f"{len(lengths)} runs; median {np.median(lengths):.0f} f; p95 {np.percentile(lengths, 95):.0f} f",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    ax = axes[3]
    error = homography.reprojection_error[homography.valid]
    if error.size:
        cap = float(np.percentile(error, 99))
        ax.hist(np.clip(error, 0, cap), bins=50, color="#1f77b4", alpha=0.8)
        ax.set_xlabel("reprojection error (pixels), clipped at p99")
        ax.set_ylabel("frames")
    ax.set_title(
        f"Homography quality: {homography.summary()['valid_pct']:.1f}% valid; "
        f"median {homography.summary()['median_reprojection_error']:.2f} px",
        loc="left",
        fontweight="bold",
    )
    _save(fig, output_dir / "01_data_overview.png")
    return output_dir / "01_data_overview.png"


def plot_public_timeline(
    gt: np.ndarray,
    pred: np.ndarray,
    features: Features,
    tags: TagTable,
    confidence: np.ndarray | None,
    output_dir: Path,
    public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP),
) -> Path:
    start, stop = public
    lo = features.start
    gt_window = gt[start:stop]
    pred_window = pred[start:stop]
    present = features.present[start - lo : stop - lo]
    tag_cov = tags.coverage[start:stop]
    x = _minutes(np.arange(start, stop))

    fig, axes = plt.subplots(4, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.1, 1.2]})
    _ribbon(axes[0], gt_window, x[0], x[-1], "Public ground truth")
    _ribbon(axes[1], pred_window, x[0], x[-1], "Prediction")
    axes[2].fill_between(x, 0, 1, where=present, step="mid", color="#2b8a3e", alpha=0.85, linewidth=0)
    axes[2].fill_between(x, 1.1, 2, where=tag_cov, step="mid", color="#6a51a3", alpha=0.6, linewidth=0)
    axes[2].set_ylim(0, 2.1)
    axes[2].set_yticks([0.5, 1.55], ["ball", "tags"])
    axes[2].set_title("Evidence availability", loc="left", fontweight="bold")
    axes[2].set_xlabel("match minute")
    if confidence is not None:
        axes[3].plot(x, confidence[start - lo : stop - lo], color="#111111", lw=0.8)
        axes[3].set_ylim(1 / 3, 1.02)
        axes[3].set_ylabel("posterior confidence")
        axes[3].set_title("HMM posterior confidence (max state probability)", loc="left", fontweight="bold")
        axes[3].set_xlabel("match minute")
    else:
        axes[3].axis("off")

    for ax in axes[:3]:
        for run_start, run_stop in _runs(~present):
            ax.axvspan(
                x[run_start],
                x[min(run_stop, len(x) - 1)],
                color="#d62728",
                alpha=0.06,
                lw=0,
            )
    _label_legend(fig)
    _save(fig, output_dir / "02_public_timeline.png")
    return output_dir / "02_public_timeline.png"


def plot_confusion(metrics: dict[str, Any], output_dir: Path) -> Path:
    # ``metrics_public`` is already restricted to the graded public window.  Show the 3x3 H/A/D
    # block rather than the all-zero X row, which would make the useful cells hard to read.
    cm = np.asarray(metrics["confusion"], dtype=int)[:3, :3]
    labels = ["H", "A", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix (graded public labels)", loc="left", fontweight="bold")
    for row in range(3):
        for col in range(3):
            value = cm[row, col]
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > cm.max() * 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    labels = ["H", "A", "D"]
    x = np.arange(len(labels))
    width = 0.25
    for index, metric in enumerate(("precision", "recall", "f1")):
        values = [float(metrics["per_class"][label][metric]) for label in labels]
        ax.bar(x + (index - 1) * width, values, width, label=metric)
        for xpos, value in zip(x + (index - 1) * width, values):
            ax.text(xpos, value + 0.015, f"{value:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title(
        f"Per-class scores | spellF1 {metrics['spell_f1_mean']:.3f}, macroF1 {metrics['macro_f1']:.3f}",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, output_dir / "03_confusion.png")
    return output_dir / "03_confusion.png"


def plot_stress(stress_rows: list[dict[str, Any]] | pd.DataFrame, metrics: dict[str, Any], output_dir: Path) -> Path:
    table = pd.DataFrame(stress_rows) if not isinstance(stress_rows, pd.DataFrame) else stress_rows.copy()
    if table.empty:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "Stress suite skipped (--no-stress).", ha="center", va="center", fontsize=13)
        _save(fig, output_dir / "04_stress.png")
        return output_dir / "04_stress.png"
    masks = ["clean", "random", "long"]
    methods = ["baseline_w50", "baseline_w100", "HMM persistence", "HMM gap-aware"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, metric, title in (
        (axes[0], "spellF1", "Spell F1 (higher is better)"),
        (axes[1], "macroF1", "Macro F1 over H/A/D"),
    ):
        pivot = table.pivot(index="mask", columns="method", values=metric).reindex(masks)
        x = np.arange(len(masks))
        width = 0.2
        for index, method in enumerate(methods):
            if method not in pivot:
                continue
            values = pivot[method].to_numpy(dtype=float)
            ax.bar(
                x + (index - 1.5) * width,
                values,
                width,
                label=method,
                color=METHOD_COLORS.get(method, "#999999"),
                alpha=0.9,
            )
            for xpos, value in zip(x + (index - 1.5) * width, values):
                ax.text(xpos, max(0, value) + 0.01, f"{value:.2f}", ha="center", fontsize=6.5, rotation=90)
        ax.set_xticks(x, masks)
        ax.set_ylim(0, 0.9)
        ax.set_ylabel(metric)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(float(metrics["spell_f1_mean"]) if metric == "spellF1" else float(metrics["macro_f1"]),
                   color="#111111", lw=0.8, ls="--", label="clean HMM" if metric == "spellF1" else None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    _save(fig, output_dir / "04_stress.png")
    return output_dir / "04_stress.png"


def plot_gap_diagnostics(
    features: Features,
    pred: np.ndarray,
    confidence: np.ndarray,
    output_dir: Path,
    public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP),
) -> Path:
    start, stop = public
    lo = features.start
    present = features.present[start - lo : stop - lo]
    pred_window = pred[start:stop]
    conf_window = confidence[start - lo : stop - lo]
    gap_window = features.gap[start - lo : stop - lo]
    runs = _runs(~present)
    lengths = np.array([b - a for a, b in runs], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.25, 1]})

    ax = axes[0]
    if lengths.size:
        ax.hist(lengths, bins=min(20, max(5, len(lengths))), color="#d62728", alpha=0.8)
    ax.set_title("Missing runs in the public window", loc="left", fontweight="bold")
    ax.set_xlabel("frames")
    ax.set_ylabel("runs")
    if lengths.size:
        ax.text(0.98, 0.92, f"{len(lengths)} runs\nmax {lengths.max():.0f} f", transform=ax.transAxes, ha="right", va="top", fontsize=8)

    ax = axes[1]
    top = sorted(runs, key=lambda pair: pair[1] - pair[0], reverse=True)[:12]
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for code, label in ((H, "H"), (A, "A"), (D, "D")):
        values = np.array([np.mean(pred_window[a:b] == code) for a, b in top])
        ax.barh(y, values, left=left, color=LABEL_COLORS[label], label=label, height=0.7)
        left += values
    ax.set_yticks(y, [f"{b - a}f" for a, b in top], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of missing run")
    ax.set_title("What the model predicts during the longest holes", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=7)

    ax = axes[2]
    missing = ~present
    if missing.any():
        colors = [LABEL_COLORS[CODE_TO_LABEL[int(code)]] for code in pred_window[missing]]
        ax.scatter(gap_window[missing], conf_window[missing], c=colors, s=8, alpha=0.55, linewidths=0)
    ax.set_xlabel("gap since last ball observation (frames)")
    ax.set_ylabel("posterior confidence")
    ax.set_ylim(1 / 3, 1.02)
    ax.set_title("Uncertainty grows inside long gaps", loc="left", fontweight="bold")
    _save(fig, output_dir / "05_gap_diagnostics.png")
    return output_dir / "05_gap_diagnostics.png"


def plot_full_match(pred: np.ndarray, stats: dict[str, Any], output_dir: Path) -> Path:
    n = len(pred)
    minutes = _minutes(np.arange(n))
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [1.2, 1.2, 1.1]})
    _ribbon(axes[0], pred, minutes[0], minutes[-1], "Full-match prediction (all 199,319 tactical frames)")
    axes[0].set_xlabel("match minute")

    ax = axes[1]
    labels = ["H", "A", "D"]
    values = [stats["shares"][label] for label in labels]
    ax.bar(labels, values, color=[LABEL_COLORS[label] for label in labels], alpha=0.9)
    for index, value in enumerate(values):
        ax.text(index, value + 0.01, f"{value:.1%}", ha="center", fontsize=9)
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_ylabel("share of graded play")
    ax.set_title(
        f"Possession share | {stats['n_spells']} spells, mean {stats['mean_duration_s']:.1f}s, "
        f"median {stats['median_duration_s']:.1f}s",
        loc="left",
        fontweight="bold",
    )

    ax = axes[2]
    durations = np.asarray(stats.get("durations_s", []), dtype=float)
    labels = stats.get("spell_labels", [])
    if durations.size:
        bins = np.linspace(0, max(1.0, float(np.percentile(durations, 98))), 24)
        for code, label in ((str(H), "H"), (str(A), "A"), (str(D), "D")):
            values = durations[np.array([item == code for item in labels], dtype=bool)]
            if values.size:
                ax.hist(values, bins=bins, alpha=0.55, label=label, color=LABEL_COLORS[label])
        ax.set_xlabel("spell duration (seconds), clipped at p98")
        ax.set_ylabel("spells")
        ax.legend(frameon=False, ncol=3)
    ax.set_title("Spell-duration distribution", loc="left", fontweight="bold")
    _save(fig, output_dir / "06_full_match.png")
    return output_dir / "06_full_match.png"


def choose_snapshot_frames(pred: np.ndarray, public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP)) -> list[int]:
    start, stop = public
    window = pred[start:stop]
    changes = np.flatnonzero(window[1:] != window[:-1]) + start + 1
    candidates = [start, start + 30, start + 80]
    if changes.size:
        picks = np.linspace(0, len(changes) - 1, num=min(6, len(changes)), dtype=int)
        candidates.extend(int(changes[index]) for index in picks)
    for offset in (250, 900, 1800, 3000):
        if start + offset < stop:
            candidates.append(start + offset)
    unique: list[int] = []
    for frame in candidates:
        if start <= frame < stop and frame not in unique:
            unique.append(frame)
    return unique[:6]


def _tag_at(tags: TagTable, frame: int) -> str:
    for event in tags.events:
        if event.start_frame <= frame < event.stop_frame:
            return f"{event.code} / {event.type}"
    return "no tag"


def plot_pitch_snapshots(
    tracking: TrackingData,
    tags: TagTable,
    pred: np.ndarray,
    frames: Iterable[int],
    output_dir: Path,
) -> Path:
    frames = list(frames)[:6]
    if not frames:
        frames = [PUBLIC_START]
    ncols = 3
    nrows = int(np.ceil(len(frames) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    length, width = tracking.pitch
    for ax, frame in zip(axes, frames):
        _draw_pitch(ax, length, width)
        home = tracking.player_present[frame] & (tracking.teams == 0)
        away = tracking.player_present[frame] & (tracking.teams == 1)
        if home.any():
            ax.scatter(tracking.players[frame, home, 0], tracking.players[frame, home, 1], s=55, c=LABEL_COLORS["H"], edgecolors="white", linewidths=0.6, zorder=3)
        if away.any():
            ax.scatter(tracking.players[frame, away, 0], tracking.players[frame, away, 1], s=55, c=LABEL_COLORS["A"], edgecolors="white", linewidths=0.6, zorder=3)
        if tracking.ball_present[frame]:
            ax.scatter([tracking.ball[frame, 0]], [tracking.ball[frame, 1]], s=110, c="white", edgecolors="black", linewidths=1.2, zorder=5)
        else:
            ax.text(0.5, 0.04, "ball not tracked", transform=ax.transAxes, ha="center", color="#ffdddd", fontsize=8)
        label = CODE_TO_LABEL[int(pred[frame])] if frame < len(pred) else "?"
        ax.set_title(
            f"frame {frame} | {frame / FPS / 60:.1f} min | {label} | {_tag_at(tags, frame)}",
            loc="left",
            fontsize=8,
            color="white",
            pad=3,
            bbox={"facecolor": "#222222", "edgecolor": "none", "alpha": 0.65},
        )
        ax.set_facecolor("#3f8f4f")
    for ax in axes[len(frames):]:
        ax.axis("off")
    fig.suptitle("Pitch snapshots: ATD player slots, ball, prediction, and smart-tag context", y=1.01, fontsize=11, fontweight="bold")
    _save(fig, output_dir / "07_pitch_snapshots.png")
    return output_dir / "07_pitch_snapshots.png"


def plot_tag_context(tags: TagTable, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), gridspec_kw={"height_ratios": [1.1, 1.2]})
    n = tags.n_frames
    minutes = _minutes(np.arange(n))
    ax = axes[0, 0]
    ax.imshow(
        tags.coverage[None, :],
        aspect="auto",
        interpolation="nearest",
        extent=[minutes[0], minutes[-1], 0, 1],
        cmap=ListedColormap(["#f1f1f1", "#6a51a3"]),
        vmin=0,
        vmax=1,
    )
    ax.set_yticks([])
    ax.set_title("Smart-tag coverage over the full recording", loc="left", fontweight="bold")
    ax.set_xlabel("match minute")

    ax = axes[0, 1]
    counts = tags.code_matrix.sum(axis=0)
    order = np.argsort(counts)[-10:]
    ax.barh(np.arange(len(order)), counts[order], color="#6a51a3", alpha=0.85)
    ax.set_yticks(np.arange(len(order)), [tags.code_names[index] for index in order], fontsize=7)
    ax.set_xlabel("tagged frames (may overlap)")
    ax.set_title("Most frequent smart-tag codes", loc="left", fontweight="bold")

    ax = axes[1, 0]
    values, counts = np.unique(tags.team_vote, return_counts=True)
    colors = ["#d62728" if value > 0 else "#1f77b4" if value < 0 else "#999999" for value in values]
    ax.bar(values, counts, color=colors, alpha=0.85)
    ax.set_xlabel("team vote (+ home / - away)")
    ax.set_ylabel("frames")
    ax.set_title("Smart-tag team vote is weak and sometimes zero", loc="left", fontweight="bold")

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(
        0.0,
        0.95,
        "How the tags are used\n"
        "• 638 intervals, 70.1% coverage\n"
        "• Only long tracking holes use the vote\n"
        "• Global tag weight is 0; tags act only in long gaps\n"
        "• Tags are context, not ground truth\n"
        "• Throw-ins are N/A in this export",
        va="top",
        fontsize=10,
        linespacing=1.6,
    )
    _save(fig, output_dir / "08_tag_context.png")
    return output_dir / "08_tag_context.png"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _optional(value: Any, fmt: str = ".3f") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return format(number, fmt)


def write_report(
    output_dir: Path,
    metrics: dict[str, Any],
    stress_rows: list[dict[str, Any]] | pd.DataFrame,
    full_stats: dict[str, Any],
    validation: dict[str, Any],
    homography_summary: dict[str, Any],
    gpu_note: str,
    tag_public_coverage: float = 0.0,
    tag_full_coverage: float = 0.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stress_df = pd.DataFrame(stress_rows) if not isinstance(stress_rows, pd.DataFrame) else stress_rows
    per_class = metrics["per_class"]
    metric_rows = [
        ("Spell F1 (headline)", f"{metrics['spell_f1_mean']:.4f}", "1:1 temporal IoU match of possessions"),
        ("Macro F1 over H/A/D", f"{metrics['macro_f1']:.4f}", "unweighted per-class frame score"),
        ("D-F1", f"{per_class['D']['f1']:.4f}", "dead-ball / out-of-play class"),
        ("Accuracy", f"{metrics['accuracy']:.4f}", "all public frames, including X"),
        ("Cohen's kappa", f"{metrics['kappa']:.4f}", "agreement after chance correction"),
        ("Alive/dead F1", f"{metrics['alive_dead_f1']:.4f}", "H or A vs D"),
        ("Team accuracy (both alive)", f"{metrics['team_acc_both_alive']:.4f}", "H/A direction when both are alive"),
        ("Possession spells", f"{metrics['n_pred_spells']}", f"GT has {metrics['n_gt_spells']}"),
        (
            "Spell F1 bootstrap 95%",
            f"[{_optional(metrics.get('spell_bootstrap_low'))}, {_optional(metrics.get('spell_bootstrap_high'))}]",
            "segments resampled; small sample",
        ),
    ]
    metric_html = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td class='num'>{html.escape(value)}</td><td>{html.escape(note)}</td></tr>"
        for name, value, note in metric_rows
    )
    stress_html = (
        stress_df.to_html(index=False, float_format=lambda value: f"{value:.4f}", border=0)
        if not stress_df.empty
        else "<p>Stress suite skipped.</p>"
    )
    shares = full_stats["shares"]
    validation_html = (
        f"<strong>{validation['n_rows']:,}</strong> rows, labels "
        f"H {validation['counts']['H']:,} / A {validation['counts']['A']:,} / "
        f"D {validation['counts']['D']:,} / X {validation['counts']['X']:,}; "
        f"{validation['n_runs']:,} label runs."
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Possession Lab — BAR-ATH</title>
<style>
:root {{ --ink: #17202a; --muted: #5d6d7e; --line: #d5dbdb; --paper: #fbfcfc; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 70px; }}
h1 {{ font-size: 34px; line-height: 1.1; margin: 0 0 8px; letter-spacing: -0.02em; }}
h2 {{ margin-top: 38px; padding-top: 14px; border-top: 1px solid var(--line); font-size: 22px; }}
h3 {{ margin-top: 24px; font-size: 17px; }}
p {{ color: #2e4053; }}
.lede {{ font-size: 17px; color: var(--muted); max-width: 900px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0 10px; }}
.card {{ background: white; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
.card .value {{ font-size: 25px; font-weight: 700; letter-spacing: -0.02em; }}
.card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
table {{ border-collapse: collapse; width: 100%; background: white; font-size: 13px; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 7px 9px; text-align: left; vertical-align: top; }}
th {{ background: #f4f6f7; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
td.num {{ font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap; }}
.note {{ background: #f4f6f7; border-left: 4px solid #1f77b4; padding: 12px 15px; margin: 16px 0; }}
.warn {{ border-left-color: #d62728; }}
img {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: white; margin: 8px 0 20px; }}
code {{ background: #eef2f3; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
ul {{ margin-top: 6px; }}
.small {{ color: var(--muted); font-size: 12px; }}
</style>
</head>
<body><main>
<h1>Possession Lab — BAR-ATH</h1>
<p class="lede">A data-only, gap-aware hidden Markov model. It uses the tactical ATD, smart tags,
homography diagnostics, and the public 5-minute ground truth as a correctness check — not as a
leaderboard, and never the private full-match truth.</p>

<div class="grid">
  <div class="card"><div class="value">{metrics['spell_f1_mean']:.3f}</div><div class="label">Spell F1</div></div>
  <div class="card"><div class="value">{metrics['macro_f1']:.3f}</div><div class="label">Macro F1</div></div>
  <div class="card"><div class="value">{per_class['D']['f1']:.3f}</div><div class="label">D-F1</div></div>
  <div class="card"><div class="value">{full_stats['n_spells']}</div><div class="label">Full-match spells</div></div>
</div>

<div class="note"><strong>What is being scored?</strong> The public GT covers frames
11,185–18,684 (7,500 frames, 17 possessions). The full submission is 199,319 rows. The full-match
numbers are structural diagnostics, not a hidden score.</div>

<h2>1. Public evaluation</h2>
<table><thead><tr><th>Metric</th><th>Value</th><th>Meaning</th></tr></thead><tbody>
{metric_html}
</tbody></table>
<img src="03_confusion.png" alt="Confusion matrix and per-class scores">
<img src="02_public_timeline.png" alt="Ground truth, prediction, evidence, confidence">

<h2>2. Missing-ball stress test</h2>
<p>The model is deliberately tested with the ball removed. The clean row is the same public
window; the random and long rows delete about 40.7% of the public frames. The baseline is a centred
H/A vote with no gap model. The gap-aware HMM should lose less.</p>
{stress_html}
<img src="04_stress.png" alt="Stress-test bars">

<h2>3. What happens inside gaps</h2>
<p>Short holes stay uniform and let the transition prior persist. Long holes switch to the team
shape around the last known ball, the last known edge, and the smart-tag vote. The posterior
confidence is a diagnostic, not a submission label.</p>
<img src="05_gap_diagnostics.png" alt="Gap diagnostics">

<h2>4. Full-match structure</h2>
<p>{full_stats['n_spells']} H/A spells over {full_stats['n_graded_frames']:,} graded frames.
H {_pct(shares['H'])}, A {_pct(shares['A'])}, D {_pct(shares['D'])}. Mean spell
{full_stats['mean_duration_s']:.1f}s; median {full_stats['median_duration_s']:.1f}s.</p>
<div class="note warn"><strong>Full-match reconciliation note.</strong> The earlier reference run
reported 256 spells and H 41.4% / A 32.5% / D 26.1%. The difference is a 747-frame tracking
blackout at the end of the first half (frames 89,905–90,651): no home players and only 1–3 away
players were tracked, so the old run's NaN emissions effectively reset to H. This implementation
adds a sparse-tracking guard (fewer than 6 visible players) and treats that stretch as missing
evidence, which predicts D. The public and stress scores are unaffected; the full-match numbers
are a structural diagnostic, not a hidden score.</div>
<img src="06_full_match.png" alt="Full-match structure">

<h2>5. Tracking and tag evidence</h2>
<p>The ATD has 46 anonymous player slots. The model only uses team membership, positions, speeds,
and ball visibility; it never invents identities. The smart tags cover
{_pct(tag_public_coverage)} of the public window ({_pct(tag_full_coverage)} full). They are a weak
prior, not a label source. The tactical homography is valid on
{homography_summary.get('valid_pct', 0.0):.1f}% of frames, with median reprojection error
{homography_summary.get('median_reprojection_error', float('nan')):.2f} px (p99
{homography_summary.get('p99_reprojection_error', float('nan')):.2f} px); it is used for diagnostics,
not as a label source.</p>
<img src="01_data_overview.png" alt="Data coverage and homography quality">
<img src="07_pitch_snapshots.png" alt="Pitch snapshots">
<img src="08_tag_context.png" alt="Smart-tag context">

<h2>6. Submission and preflight</h2>
<p>{validation_html} The official validator is the final gate; the package writes exactly
<code>frame,label</code> and keeps the generated file outside the public code tree.</p>
<p><a href="predictions.csv">predictions.csv</a> · <a href="metrics_public.json">metrics_public.json</a>
· <a href="stress.csv">stress.csv</a> · <a href="full_stats.json">full_stats.json</a></p>

<h2>7. Why CPU, not GPU</h2>
<div class="note">{html.escape(gpu_note)}</div>

<h2>8. Reproduce</h2>
<pre><code>python -m possession_lab all
python -m unittest discover -s possession_lab/tests -v
python evaluation/validate_submission.py possession_lab/outputs/predictions.csv</code></pre>

<h2>9. Honest limitations</h2>
<ul>
<li>The public GT is 17 possessions. The confidence interval is wide; do not over-read small differences.</li>
<li>The tactical tracking is not perfect and has 46 anonymous slots. Player identity is not available.</li>
<li>Smart tags cover 70.1% of the match; their team direction is often ambiguous and every
throw-in is N/A.</li>
<li>The model is tuned on the public window, so the full-match result is a generalization check, not a validation set.</li>
<li>Homography is used for diagnostics here. The tactical homography is valid on 100% of frames; broadcast is not used.</li>
</ul>
<p class="small">Generated by <code>possession_lab</code>. Data remains gated; no private full-match
ground truth and no match video frames are copied into this report.</p>
</main></body></html>
"""
    path = output_dir / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def generate_all(
    tracking: TrackingData,
    tags: TagTable,
    homography: Homography,
    pred: np.ndarray,
    gt: np.ndarray,
    metrics: dict[str, Any],
    stress_rows: list[dict[str, Any]] | pd.DataFrame,
    full_stats: dict[str, Any],
    validation: dict[str, Any],
    output_dir: str | Path,
    gpu_note: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, result, _ = decode_public(tracking, tags, DEFAULT_PARAMS)
    confidence = result.confidence
    frames = choose_snapshot_frames(pred)
    paths = {}
    paths["data_overview"] = str(plot_data_overview(tracking, tags, homography, output_dir))
    paths["public_timeline"] = str(plot_public_timeline(gt, pred, features, tags, confidence, output_dir))
    paths["confusion"] = str(plot_confusion(metrics, output_dir))
    paths["stress"] = str(plot_stress(stress_rows, metrics, output_dir))
    paths["gap_diagnostics"] = str(plot_gap_diagnostics(features, pred, confidence, output_dir))
    paths["full_match"] = str(plot_full_match(pred, full_stats, output_dir))
    paths["pitch_snapshots"] = str(plot_pitch_snapshots(tracking, tags, pred, frames, output_dir))
    paths["tag_context"] = str(plot_tag_context(tags, output_dir))
    paths["report"] = str(
        write_report(
            output_dir,
            metrics,
            stress_rows,
            full_stats,
            validation,
            homography.summary(),
            gpu_note,
            tag_public_coverage=float(tags.coverage[PUBLIC_START:PUBLIC_STOP].mean()),
            tag_full_coverage=float(tags.coverage.mean()),
        )
    )
    return paths
