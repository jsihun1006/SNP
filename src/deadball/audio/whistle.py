"""Run an audio-only candidate detector locally and compare with HSJ transition times."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from games import GAME, asset_paths, half_video_blocks

FPS = 25
RATE = 16_000
HOP = 320
WINDOW = 1024
FREQUENCIES = rfftfreq(WINDOW, 1 / RATE)
BAND = np.flatnonzero((FREQUENCIES >= 2400) & (FREQUENCIES <= 4800))
AUDIO_BAND = (FREQUENCIES >= 500) & (FREQUENCIES <= 6800)


def spectral_peaks(samples):
    windows = np.lib.stride_tricks.sliding_window_view(samples, WINDOW)[::HOP]
    power = np.abs(rfft(windows * np.hanning(WINDOW).astype(np.float32), axis=1)) ** 2
    indices = BAND[np.argmax(power[:, BAND], axis=1)]
    rows = np.arange(len(indices))[:, None]
    peak = power[rows, indices[:, None] + np.arange(-2, 3)].sum(axis=1)
    total = power[:, AUDIO_BAND].sum(axis=1)
    # An isolated spectral ridge is more consistent with a whistle than a loud broadband burst.
    flank = power[rows, indices[:, None] + np.r_[np.arange(-20, -4), np.arange(5, 21)]]
    prominence = 10 * np.log10((peak + 1e-8) / (5 * np.median(flank, axis=1) + 1e-8))
    return FREQUENCIES[indices], peak / np.maximum(total, 1e-8), prominence, peak


def candidate_events(frequency, fraction, prominence, energy):
    strong = (fraction >= .20) & (prominence >= 14) & (energy >= .01)
    # Allow one quiet hop inside a candidate. A burst needs 5 consecutive hops in total.
    bridge = strong.copy()
    bridge[1:-1] |= strong[:-2] & strong[2:]
    edges = np.flatnonzero(np.diff(np.r_[False, bridge, False]))
    events = []
    for start, end in zip(edges[::2], edges[1::2]):
        hits = np.flatnonzero(strong[start:end]) + start
        if end - start < 5 or len(hits) < 5 or end - start > 150:
            continue
        if np.ptp(np.percentile(frequency[hits], [10, 90])) > 220:
            continue
        best = hits[np.argmax(energy[hits])]
        events.append((round(start * HOP / RATE, 3), round((end * HOP + WINDOW) / RATE, 3),
                       round((best * HOP + WINDOW / 2) / RATE, 3), round(float(frequency[best]), 1)))
    return pd.DataFrame(events, columns=["start_s", "end_s", "peak_s", "peak_hz"])


def stream_audio(video):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
               "-protocol_whitelist", "file,pipe", "-i", str(video), "-map", "0:a:0", "-ac", "1",
               "-ar", str(RATE), "-f", "s16le", "pipe:1"]
    spectra = []
    retained = np.empty(0, dtype=np.float32)
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        while block := process.stdout.read(RATE * 20 * 2):
            samples = np.concatenate((retained, np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768))
            if len(samples) < WINDOW:
                break
            peaks = spectral_peaks(samples)
            spectra.append(np.stack(peaks))
            retained = samples[len(peaks[0]) * HOP:]
        if process.wait():
            raise RuntimeError("Failed to read the tactical audio track")
    return candidate_events(*np.concatenate(spectra, axis=1))


def nearest_distance(times, boundaries):
    slots = np.searchsorted(boundaries, times)
    left = boundaries[np.clip(slots - 1, 0, len(boundaries) - 1)]
    right = boundaries[np.clip(slots, 0, len(boundaries) - 1)]
    return np.minimum(np.abs(times - left), np.abs(times - right))


def summarize(events, segments_path):
    segments = pd.read_csv(segments_path)
    blocks = [(first / FPS, (last + 1) / FPS) for _, first, last in half_video_blocks(GAME)]
    blocks[0] = (max(blocks[0][0], 18685 / FPS), blocks[0][1])  # Keep the public diagnostic slice out.
    times = events.peak_s.to_numpy()
    starts = segments.start.to_numpy() / FPS
    ends = (segments.end.to_numpy() + 1) / FPS
    half_results = []
    for lower, upper in blocks:
        peaks = times[(times >= lower) & (times < upper)]
        local_starts = starts[(starts >= lower) & (starts < upper)]
        local_ends = ends[(ends >= lower) & (ends < upper)]
        near_start = nearest_distance(peaks, local_starts) <= 2
        near_end = nearest_distance(peaks, local_ends) <= 2
        shifted_start, shifted_end = [], []
        for shift in (13, 29, 47, 67, 89, 113, 151, 193):
            shifted = lower + ((peaks - lower + shift) % (upper - lower))
            shifted_start.append(int(np.sum(nearest_distance(shifted, local_starts) <= 2)))
            shifted_end.append(int(np.sum(nearest_distance(shifted, local_ends) <= 2)))
        half_results.append({"candidates": int(len(peaks)), "near_hsj_start": int(near_start.sum()),
                             "near_hsj_end": int(near_end.sum()), "near_both": int((near_start & near_end).sum()),
                             "start_overlap_after_shift": shifted_start, "end_overlap_after_shift": shifted_end})
    return {"bands_hz": [2400, 4800], "overlap_tolerance_s": 2,
            "comparison": "Agreement with HSJ predictions; shifts are a descriptive control, not ground truth.",
            "halves_excluding_public_window": half_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--hsj-segments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    video = args.video or Path(asset_paths()["video"])
    rng = np.random.default_rng(428)
    noise = rng.normal(0, .01, RATE * 2).astype(np.float32)
    samples = noise.copy()
    tone_time = np.arange(RATE // 4) / RATE
    samples[RATE:RATE + len(tone_time)] += .12 * np.sin(2 * np.pi * 3900 * tone_time)
    events = candidate_events(*spectral_peaks(samples))
    assert len(events) == 1 and 1.0 <= events.iloc[0].peak_s <= 1.25
    assert candidate_events(*spectral_peaks(noise)).empty
    print("Synthetic tone/noise check passed.", flush=True)
    actual = stream_audio(video)
    args.output.mkdir(parents=True, exist_ok=True)
    actual.to_csv(args.output / "whistle_candidates.csv", index=False)
    summary = summarize(actual, args.hsj_segments)
    (args.output / "whistle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
