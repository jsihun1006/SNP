"""Full-in-play 25 Hz training; fixed epoch budget, minimum validation-loss selection."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from inputs import HA, Half, N_FEATURES, atomic_npz, frozen_predictions, make_features
from prepare_tracking import ROOT, prepare_tracking
from refiner import Refiner, decode, loss_sums
from run import Window, collate, digest, read_config, save_checkpoint, sync, weights, write_json
from transportmer_ha_data import contiguous_runs
from transportmer_ha_metrics import metrics

FOLDER = Path(__file__).resolve().parent
CONFIGS = ROOT / "experiments/possession/pathcrf/candidate25_dfl"


def amp_context(device, config):
    return (torch.autocast('cuda', dtype=torch.bfloat16 if config['amp_dtype'] == 'bfloat16' else torch.float16)
            if device.type == 'cuda' and config['amp'] else nullcontext())


def load_full_match(args, match, source_device):
    """Read native tracking and frozen D only; never run or consult 5 Hz H/A."""
    tick = perf_counter()
    source = prepare_tracking(args.upstream, args.raw, args.sources, match)
    official = pd.read_parquet(source).set_index('frame_id', verify_integrity=True)
    tracking = pd.read_parquet(args.cache / f'{match}_tracking.parquet')
    roster = pd.read_parquet(args.cache / f'{match}_meta.parquet')
    players = sorted(c[:-2] for c in official if c.startswith(('home_', 'away_')) and c.endswith('_x'))
    names = players + ['ball']
    team = np.asarray([HA[name.split('_')[0]] for name in players] + [2], np.int8)
    roles = {}
    for row in roster.dropna(subset=['jersey']).itertuples(index=False):
        role = str(row.position).strip().lower()
        if role not in {'', 'unknown', 'none', 'nan'}:
            roles[f'{row.team}_{int(row.jersey)}'] = int(role in {'gk', 'goalkeeper'})
    keeper = np.asarray([roles.get(p, -1) for p in players] + [-1], np.int8)
    data = []
    for number in (1, 2):
        rows = tracking.loc[tracking.period.eq(number)]
        frame = rows.frame_id.to_numpy(np.int64)
        time = rows.timestamp_s.to_numpy(np.float64)
        if not len(frame) or (np.diff(frame) <= 0).any() or (np.diff(time) <= 0).any():
            raise ValueError(f'Invalid original frame/time axis: {match} H{number}')
        aligned = official.reindex(frame)
        covered = aligned.period_id.notna().to_numpy()
        np.testing.assert_allclose(aligned.loc[covered, 'timestamp'], time[covered], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(aligned.loc[covered, 'period_id'], np.full(covered.sum(), number))
        np.testing.assert_array_equal(aligned.loc[covered, 'ball_state'].astype(str).str.lower().eq('alive'),
                                      rows.loc[covered, 'ball_state'].eq(1))
        xy = aligned[[f'{name}_{axis}' for name in names for axis in ('x', 'y')]].to_numpy(np.float32)
        xy = xy.reshape(len(frame), len(names), 2)
        truth = rows.ball_owning_team.map(HA).fillna(-1).to_numpy(np.int8)
        truth[~rows.ball_state.eq(1).to_numpy()] = 2
        data.append((number, frame, time, truth, xy))
    cached = args.frozen_cache / f'{match}_pred.npz'
    if cached.exists():
        with np.load(cached) as saved:
            full = []
            for number, frame, *_ in data:
                np.testing.assert_array_equal(saved[f'h{number}_frame'], frame)
                full.append(saved[f'h{number}_full'].copy())
    else:
        full = frozen_predictions([(n, f) for n, f, *_ in data], tracking, roster,
                                  args.dead_scene / 'full/best.pt', args.dead_scene / 'full/fit.json', source_device)
        args.frozen_cache.mkdir(parents=True, exist_ok=True)
        atomic_npz(cached, **{key: value for (number, frame, *_), p in zip(data, full, strict=True)
                              for key, value in ((f'h{number}_frame', frame), (f'h{number}_full', p))})
    result = []
    for (number, frame, time, truth, xy), prediction in zip(data, full, strict=True):
        dead = np.asarray(prediction) == 2
        # This is a boundary-only placeholder. Its H/A values are never predictions or inputs.
        boundary = np.where(dead, 2, 0).astype(np.int8)
        result.append(Half(f'{match}_H{number}', match, number, frame, time, truth, xy, team, keeper,
                           dead, {'base': boundary, 'fallback': 0}, np.empty((len(frame), 0))))
    print(f'{match}: native 25fps, frozen D only, {sum(len(h.frame) for h in result):,} frames; '
          f'{perf_counter() - tick:.1f}s', flush=True)
    return result


def full_windows(halves, config, training):
    refs, runs = [], []
    width = config['refine_window_frames']
    stride = config['train_stride_seconds'] * config['refine_fps'] if training else config['infer_stride_frames']
    for index, half in enumerate(halves):
        active = half.labels != 2 if training else ~half.dead_prediction
        for a, b in contiguous_runs(active, half.frame):
            last = max(a, b - width)
            starts = sorted({a + math.floor(k * stride) for k in range(math.ceil((last - a) / stride))} | {last})
            for start in starts:
                stop = min(b, start + width)
                if training and not np.isin(half.labels[start:stop], (0, 1)).any():
                    continue
                refs.append(Window(index, start, stop))
            if not training:
                runs.append((index, a, b))
    return refs, runs


def supervised_batch(halves, refs, config, rng=None):
    batch = list(collate(halves, refs, config, rng))
    # Validation uses unchanged inputs; supervision still needs labels without augmentation.
    if rng is None:
        y, transition = batch[2].numpy(), batch[3].numpy()
        for i, ref in enumerate(refs):
            labels = halves[ref.half].labels[ref.start:ref.stop].astype(np.int64)
            labels[~np.isin(labels, (0, 1))] = -1
            y[i, :len(labels)] = labels
            valid = (labels[:-1] >= 0) & (labels[1:] >= 0)
            transition[i, 1:len(labels)][valid] = np.where(labels[:-1][valid] == labels[1:][valid], 0,
                                                         2 - labels[1:][valid])
    return batch


def denominators(batch, weight):
    valid = (batch[2] >= 0).sum()
    transition = batch[3].flatten()
    counts = torch.bincount(transition[transition >= 0], minlength=3).to(weight.device)
    return torch.stack((valid.to(weight.device), (counts * weight).sum(), valid.to(weight.device))).float()


def update(model, optimizer, scaler, halves, refs, config, device, weight, rng, microbatch):
    batches = [supervised_batch(halves, refs[i:i + microbatch], config, rng)
               for i in range(0, len(refs), microbatch)]
    denom = sum((denominators(b, weight) for b in batches), start=weight.new_zeros(3)).clamp_min(1)
    coefficient = weight.new_tensor([config[f'{name}_loss_weight'] for name in ('state', 'transition', 'crf')])
    optimizer.zero_grad(set_to_none=True)
    total = weight.new_zeros(3)
    for batch in batches:
        x, mask, y, transition = [part.to(device) for part in batch]
        with amp_context(device, config):
            state, change = model(x, mask)
        # CE and CRF dynamic programming run in FP32 even when attention uses AMP.
        normal = loss_sums(state.float(), change.float(), y, transition, weight) / denom
        loss = (normal * coefficient).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss; last completed epoch remains saved.')
        scaler.scale(loss).backward()
        total += normal.detach()
    scaler.step(optimizer)
    scaler.update()
    return total.cpu().numpy()


@torch.inference_mode()
def evaluate(model, halves, refs, runs, config, device, weight, microbatch):
    model.eval()
    summed = [np.zeros((len(h.frame), 5), np.float32) for h in halves]
    count = [np.zeros(len(h.frame), np.float32) for h in halves]
    losses, sizes = weight.new_zeros(3), weight.new_zeros(3)
    for i in range(0, len(refs), microbatch):
        batch_refs = refs[i:i + microbatch]
        batch = supervised_batch(halves, batch_refs, config)
        x, mask, y, transition = [part.to(device) for part in batch]
        with amp_context(device, config):
            state, change = model(x, mask)
        losses += loss_sums(state.float(), change.float(), y, transition, weight)
        sizes += denominators(batch, weight)
        out = torch.cat((state.float(), change.float()), -1).cpu().numpy()
        if not np.isfinite(out).all():
            raise FloatingPointError('Nonfinite validation/test logits.')
        for ref, values in zip(batch_refs, out, strict=True):
            length = ref.stop - ref.start
            w = np.minimum(np.arange(1, length + 1), np.arange(length, 0, -1))
            summed[ref.half][ref.start:ref.stop] += values[:length] * w[:, None]
            count[ref.half][ref.start:ref.stop] += w
    prediction = [np.full(len(h.frame), 2, np.int8) for h in halves]
    for index, a, b in runs:
        if not (count[index][a:b] > 0).all():
            raise AssertionError(f'Inference coverage hole: {halves[index].key} {a}:{b}')
        logits = summed[index][a:b] / count[index][a:b, None]
        # Entire predicted in-play run: no 5fps boundary labels, gate, or fallback.
        prediction[index][a:b] = decode(logits[:, :2], logits[:, 2:])
    for h, pred in zip(halves, prediction, strict=True):
        np.testing.assert_array_equal(pred == 2, h.dead_prediction)
        h.alive = h.labels != 2
    result = metrics(halves, prediction)
    components = (losses / sizes.clamp_min(1)).cpu().numpy()
    coeff = np.array([config[f'{name}_loss_weight'] for name in ('state', 'transition', 'crf')])
    result['validation_loss'] = float(components @ coeff)
    result['loss_components'] = dict(zip(('state', 'transition', 'crf'), components.tolist()))
    return result, prediction


def prepare(args, config, source_device):
    started = perf_counter()
    training, augmentation = [], {}
    for match in config['train_matches'].split():
        for half in load_full_match(args, match, source_device):
            augmentation[half.key] = make_features(half, config, training=True)
            half.xy = None
            training.append(half)
    validation = load_full_match(args, config['validation_match'], source_device)
    for half in validation:
        make_features(half, config, training=False)
        half.xy = None
    refs, _ = full_windows(training, config, True)
    val_refs, runs = full_windows(validation, config, False)
    if not refs or not val_refs:
        raise ValueError('No training or validation windows.')
    write_json(args.output / 'ball_mask.json', augmentation)
    report = {'training_window_pool': len(refs), 'validation_windows': len(val_refs),
              'train_halves': len(training), 'validation_halves': len(validation),
              'seconds': perf_counter() - started, 'sampling': 'all unique in-play windows; no candidate oversampling'}
    write_json(args.output / 'prepared.json', report)
    print(json.dumps(report), flush=True)
    return training, validation, refs, val_refs, runs


def choose_epoch(refs, config, rng):
    cap = config['epoch_window_cap']
    order = rng.permutation(len(refs))
    if cap:
        order = order[:cap]
    return [refs[i] for i in order]


def run_benchmark(args, config, device, parts, model, optimizer, scaler, weight):
    training, validation, refs, val_refs, runs = parts
    rng = np.random.default_rng(config['seed'])
    selected = choose_epoch(refs, config, rng)
    batch = config['batch']
    if len(selected) < batch:
        raise ValueError('Benchmark requires one full effective batch.')
    times = []
    for step in range(args.bench_steps + 1):
        start = (step * batch) % (len(selected) - batch + 1)
        model.train()
        sync(device)
        tick = perf_counter()
        loss = update(model, optimizer, scaler, training, selected[start:start + batch], config,
                      device, weight, rng, args.microbatch)
        sync(device)
        elapsed = perf_counter() - tick
        if step:
            times.append(elapsed)
        print(f'{"warmup" if not step else "timed"} update: {elapsed:.3f}s; losses={loss.tolist()}', flush=True)
    tick = perf_counter()
    val, _ = evaluate(model, validation, val_refs, runs, config, device, weight, args.microbatch)
    sync(device)
    validation_seconds = perf_counter() - tick
    steps = math.ceil(len(selected) / batch)
    per_epoch = [float(f(times)) * steps + validation_seconds for f in (min, np.median, max)]
    report = {'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
              'amp': config['amp'], 'amp_dtype': config['amp_dtype'], 'effective_batch': batch,
              'microbatch': args.microbatch, 'windows_per_epoch': len(selected),
              'optimizer_steps_per_epoch': steps, 'train_step_seconds': times,
              'validation_windows': len(val_refs), 'validation_seconds': validation_seconds,
              'projected_epoch_s_min_median_max': per_epoch, 'projected_epochs': config['max_epochs'],
              'projected_training_validation_s_min_median_max': [v * config['max_epochs'] for v in per_epoch]}
    write_json(args.output / 'benchmark.json', report)
    print(json.dumps(report, indent=2), flush=True)


def train(args, config, device, parts, model, optimizer, scaler, weight):
    training, validation, refs, val_refs, runs = parts
    rng = np.random.default_rng(config['seed'])
    history, best_loss, lr_wait = [], math.inf, 0
    last = args.output / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'], device)
        history, best_loss, lr_wait = saved['history'], saved['best_loss'], saved['lr_wait']
        print(f'Resuming after epoch {len(history)}', flush=True)
    for epoch in range(len(history) + 1, config['max_epochs'] + 1):
        tick = perf_counter()
        selected = choose_epoch(refs, config, rng)
        model.train()
        losses = []
        for first in range(0, len(selected), config['batch']):
            losses.append(update(model, optimizer, scaler, training, selected[first:first + config['batch']],
                                 config, device, weight, rng, args.microbatch))
            if first == 0 or (first // config['batch'] + 1) % 100 == 0:
                print(f'Epoch {epoch}/{config["max_epochs"]}: batch {first // config["batch"] + 1}/'
                      f'{math.ceil(len(selected) / config["batch"])}', flush=True)
        validation_result, _ = evaluate(model, validation, val_refs, runs, config, device, weight, args.microbatch)
        value = validation_result['validation_loss']
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite validation loss.')
        if value < best_loss:
            best_loss, lr_wait = value, 0
            save_checkpoint(args.output / 'best.pt', model.state_dict())
            write_json(args.output / 'best_validation.json', {'epoch': epoch, 'selection': 'minimum validation loss',
                                                              'scores': validation_result})
        else:
            lr_wait += 1
            if lr_wait >= config['lr_patience']:
                for group in optimizer.param_groups:
                    group['lr'] = max(config['min_lr'], group['lr'] * .5)
                lr_wait = 0
        row = {'epoch': epoch, 'validation_loss': value, 'validation_spellF1': validation_result['spellF1'],
               'validation_loss_components': validation_result['loss_components'],
               'lr': optimizer.param_groups[0]['lr'], 'seconds': perf_counter() - tick,
               'train_windows': len(selected), 'optimizer_steps': len(losses),
               'training_loss_components': np.mean(losses, axis=0).tolist()}
        history.append(row)
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                      'rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                      'history': history, 'best_loss': best_loss, 'lr_wait': lr_wait}
        if device.type == 'cuda':
            checkpoint['cuda_rng'] = torch.cuda.get_rng_state(device)
        save_checkpoint(last, checkpoint)
        write_json(args.output / 'training.json', history)
        pd.json_normalize(history).to_csv(args.output / 'training_history.csv', index=False, encoding='utf-8-sig')
        print(json.dumps(row), flush=True)
    print(f'Finished {len(history)}/{config["max_epochs"]} epochs; best validation loss={best_loss}', flush=True)


def test(args, config, device, model, source_device, weight):
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True))
    primary_halves = load_full_match(args, config['test_match'], source_device)
    primary_dead = [h.dead_prediction.copy() for h in primary_halves]
    results = {}
    for mode in ('estimated_D_primary', 'oracle_D_diagnostic'):
        halves = primary_halves
        if mode == 'oracle_D_diagnostic':
            for half in halves:
                half.dead_prediction = half.labels == 2
                half.coarse['base'] = np.where(half.dead_prediction, 2, 0).astype(np.int8)
        for half in halves:
            make_features(half, config, training=False)
        refs, runs = full_windows(halves, config, False)
        tick = perf_counter()
        result, predictions = evaluate(model, halves, refs, runs, config, device, weight, args.microbatch)
        result.pop('validation_loss')
        result.pop('loss_components')
        for half, pred, dead in zip(halves, predictions, primary_dead, strict=True):
            table = pd.DataFrame({'match': half.match, 'half': half.number, 'frame': half.frame,
                                  'seconds': half.time, 'truth': half.labels, 'dead': half.dead_prediction,
                                  'predicted_dead': dead, 'prediction_25fps': pred})
            table.to_parquet(args.output / f'{mode}_{half.key}.parquet', index=False)
            table.to_csv(args.output / f'{mode}_{half.key}.csv', index=False, encoding='utf-8-sig')
        results[mode] = {'model': 'standalone_25fps', **result, 'inference_windows': len(refs),
                         'seconds': perf_counter() - tick}
    write_json(args.output / 'holdout.json', results)
    pd.DataFrame([{'dead_mask': mode, **r} for mode, r in results.items()]).to_csv(
        args.output / 'holdout_comparison.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('benchmark', 'train', 'test'))
    parser.add_argument('--config', type=Path, default=CONFIGS / 'parameters_25fps.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/transition_final/25fps_25epochs_minloss')
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/external/dfl')
    parser.add_argument('--sources', type=Path, default=ROOT / 'data/transition_final/source')
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/external/dfl/raw')
    parser.add_argument('--upstream', type=Path, default=ROOT / 'pathcrf')
    parser.add_argument('--frozen-cache', type=Path, default=ROOT / 'frozen_for_colab')
    parser.add_argument('--dead-scene', type=Path, default=ROOT / 'data/pathcrf_dfl/scene_two_models_v1')
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--bench-steps', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA GPU required. Select a Colab GPU runtime.')
    if not 1 <= args.microbatch <= config['batch'] or args.bench_steps < 1:
        parser.error('Invalid batch or benchmark count.')
    if config['refine_fps'] != 25 or config['max_epochs'] < 1 or config['early_stop'] != 0:
        parser.error('This entry point requires native 25fps, positive epochs, and early_stop=0.')
    device, source_device = torch.device(args.device), torch.device('cpu')
    config['amp'] = bool(config['amp'] and device.type == 'cuda')
    if config['amp_dtype'] == 'auto':
        config['amp_dtype'] = 'bfloat16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
    torch.set_num_threads(args.threads)
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    args.output.mkdir(parents=True, exist_ok=True)
    split = {'train': config['train_matches'].split(), 'validation': config['validation_match'], 'test': config['test_match']}
    if len(set(split['train'] + [split['validation'], split['test']])) != len(split['train']) + 2:
        parser.error('Matches overlap across splits.')
    if json.loads((args.dead_scene / 'manifest.json').read_text())['split'] != split:
        parser.error('Frozen dead-ball model split differs.')
    identity = {'dead_checkpoint_sha256': digest(args.dead_scene / 'full/best.pt'),
                'dead_decoder_sha256': digest(args.dead_scene / 'full/fit.json'),
                'runner_sha256': digest(Path(__file__)), 'refiner_sha256': digest(FOLDER / 'refiner.py'),
                'input_sha256': digest(FOLDER / 'inputs.py')}
    manifest = {'experiment': 'standalone native 25fps over all in-play frames', 'split': split,
                'parameters': config, 'inputs': identity, 'microbatch': args.microbatch,
                'selection': 'minimum validation loss; no early stopping', 'five_fps_HA_used': False}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error('Existing run differs; use a new output directory.')
    if args.stage == 'test' and not manifest_path.exists():
        parser.error('Missing training manifest.')
    write_json(manifest_path, manifest)
    with (args.output / 'parameters.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.writer(out)
        writer.writerow(['parameter', 'value'])
        writer.writerows(config.items())
    write_json(args.output / 'runtime_info.json', {'device': str(device), 'torch': str(torch.__version__),
                                                  'amp': config['amp'], 'amp_dtype': config['amp_dtype'],
                                                  'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
    model = Refiner(config, N_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'] and config['amp_dtype'] == 'float16')
    if args.stage == 'test':
        stored = json.loads((args.output / 'transition_weights.json').read_text())
        weight = torch.tensor(stored['weights'], dtype=torch.float32, device=device)
        test(args, config, device, model, source_device, weight)
        return
    parts = prepare(args, config, source_device)
    torch.manual_seed(config['seed'])
    weight, counts = weights(parts[0], config, device)
    write_json(args.output / 'transition_weights.json', {'counts': counts.tolist(), 'weights': weight.cpu().tolist()})
    if args.stage == 'benchmark':
        run_benchmark(args, config, device, parts, model, optimizer, scaler, weight)
    else:
        train(args, config, device, parts, model, optimizer, scaler, weight)


if __name__ == '__main__':
    main()
