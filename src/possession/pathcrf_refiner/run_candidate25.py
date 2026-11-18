"""Frozen 5fps baseline plus a 25fps specialist supervised only on suspected contests."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from inputs import N_FEATURES, ROOT, gate, load_match, make_features
from refiner import Refiner, decode, loss_sums
from run import (Window, collate, digest, evaluation, inference_windows, read_config,
                 save_checkpoint, sync, write_json)
from run_25fps import amp_context, denominators
from transportmer_ha_data import contiguous_runs

FOLDER = Path(__file__).resolve().parent
CONFIGS = ROOT / "experiments/possession/pathcrf/candidate25_dfl"


def candidate_masks(halves, config):
    # Frozen rule using model outputs and observed geometry, never the match's labels.
    threshold = config['gate_threshold']
    masks = [gate(h, threshold, config['gate_margin_frames']) for h in halves]
    for h, mask in zip(halves, masks, strict=True):
        if (mask & (h.coarse['base'] == 2)).any():
            raise AssertionError('Candidate gate crossed predicted dead ball.')
    return masks


def training_windows(halves, masks, config):
    width = config['refine_window_frames']
    stride = config['train_stride_seconds'] * config['refine_fps']
    refs = set()
    for index, (half, mask) in enumerate(zip(halves, masks, strict=True)):
        # True D is only used to separate supervised training windows, not to select candidates.
        for first, stop in contiguous_runs(half.labels != 2, half.frame):
            limit = max(first, stop - width)
            for local_a, local_b in contiguous_runs(mask[first:stop], half.frame[first:stop]):
                a, b = first + local_a, first + local_b
                centres = {a + math.floor(k * stride) for k in range(math.ceil((b - a) / stride))} | {b - 1}
                for centre in centres:
                    start = max(first, min(centre - width // 2, limit))
                    end = min(stop, start + width)
                    if (mask[start:end] & np.isin(half.labels[start:end], (0, 1))).any():
                        refs.add(Window(index, start, end))
    return sorted(refs)


def specialist_batch(halves, refs, config, rng=None):
    # Context frames enter attention, but labels/loss outside candidates are always masked.
    values, visible, labels, transitions = collate(halves, refs, config, rng)
    labels.fill_(-1)
    transitions.fill_(-1)
    y, change = labels.numpy(), transitions.numpy()
    for i, ref in enumerate(refs):
        half = halves[ref.half]
        raw = half.labels[ref.start:ref.stop].astype(np.int64)
        supervised = half.candidate_mask[ref.start:ref.stop] & np.isin(raw, (0, 1))
        length = len(raw)
        raw[~supervised] = -1
        y[i, :length] = raw
        valid = (raw[:-1] >= 0) & (raw[1:] >= 0)
        change[i, 1:length][valid] = np.where(raw[:-1][valid] == raw[1:][valid], 0, 2 - raw[1:][valid])
    return values, visible, labels, transitions


def class_weights(halves, config, device):
    counts = np.zeros(3, np.int64)
    for half in halves:
        eligible = half.candidate_mask & np.isin(half.labels, (0, 1))
        for a, b in contiguous_runs(eligible, half.frame):
            left, right = half.labels[a:b - 1], half.labels[a + 1:b]
            counts += np.bincount(np.where(left == right, 0, 2 - right), minlength=3)
    weight = np.sqrt(counts[0] / np.maximum(1, counts)).clip(1, config['max_transition_class_weight'])
    return torch.tensor(weight, device=device, dtype=torch.float32), counts


def train_update(model, optimizer, scaler, halves, refs, config, device, weight, rng, microbatch):
    batches = [specialist_batch(halves, refs[i:i + microbatch], config, rng)
               for i in range(0, len(refs), microbatch)]
    sizes = sum((denominators(b, weight) for b in batches), start=weight.new_zeros(3)).clamp_min(1)
    coefficients = weight.new_tensor([config[f'{key}_loss_weight'] for key in ('state', 'transition', 'crf')])
    optimizer.zero_grad(set_to_none=True)
    total = weight.new_zeros(3)
    for parts in batches:
        x, mask, y, change = [p.to(device) for p in parts]
        with amp_context(device, config):
            state, transition = model(x, mask)
        loss = loss_sums(state.float(), transition.float(), y, change, weight) / sizes
        objective = (loss * coefficients).sum()
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite specialist loss. Resume from the last completed epoch.')
        scaler.scale(objective).backward()
        total += loss.detach()
    scaler.step(optimizer)
    scaler.update()
    return total.cpu().numpy()


@torch.inference_mode()
def specialist_evaluation(model, halves, masks, refs, patches, config, device, weight, microbatch):
    model.eval()
    totals = [np.zeros((len(h.frame), 5), np.float32) for h in halves]
    counts = [np.zeros(len(h.frame), np.float32) for h in halves]
    losses, sizes = weight.new_zeros(3), weight.new_zeros(3)
    for first in range(0, len(refs), microbatch):
        batch_refs = refs[first:first + microbatch]
        parts = specialist_batch(halves, batch_refs, config)
        x, visible, y, change = [p.to(device) for p in parts]
        with amp_context(device, config):
            state, transition = model(x, visible)
        losses += loss_sums(state.float(), transition.float(), y, change, weight)
        sizes += denominators(parts, weight)
        outputs = torch.cat((state.float(), transition.float()), -1).cpu().numpy()
        if not np.isfinite(outputs).all():
            raise FloatingPointError('Nonfinite inference logits.')
        for ref, output in zip(batch_refs, outputs, strict=True):
            n = ref.stop - ref.start
            blend = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1))
            totals[ref.half][ref.start:ref.stop] += output[:n] * blend[:, None]
            counts[ref.half][ref.start:ref.stop] += blend
    prediction = [h.coarse['base'].copy() for h in halves]
    for index, a, b, first, stop in patches:
        right = b < stop
        end = b + int(right)
        if not (counts[index][a:end] > 0).all():
            raise AssertionError('Specialist inference window coverage hole.')
        logits = totals[index][a:end] / counts[index][a:end, None]
        base = halves[index].coarse['base']
        prediction[index][a:b] = decode(logits[:b - a, :2], logits[:b - a, 2:],
                                       int(base[a - 1]) if a > first else None,
                                       int(base[b]) if right else None, logits[-1, 2:] if right else None)
    # Reuses exact saved metric definitions and asserts unchanged outside-candidate and D frames.
    result = evaluation(halves, prediction, masks)
    components = (losses / sizes.clamp_min(1)).cpu().numpy()
    coefficients = np.asarray([config[f'{key}_loss_weight'] for key in ('state', 'transition', 'crf')])
    result['validation_loss'] = float(components @ coefficients)
    result['candidate_loss_components'] = dict(zip(('state','transition','crf'), components.tolist()))
    return result, prediction


def prepare(args, config, source_device):
    tick = perf_counter()
    training, reports, gate_reports = [], {}, {}
    for match in config['train_matches'].split():
        halves = load_match(args, match, source_device)
        masks = candidate_masks(halves, config)
        for half, mask in zip(halves, masks, strict=True):
            half.candidate_mask = mask
            reports[half.key] = make_features(half, config, training=True)
            half.xy = None
            gate_reports[half.key] = {'candidate_frames': int(mask.sum()),
                                     'predicted_inplay_frames': int((half.coarse['base'] != 2).sum())}
            training.append(half)
    validation = load_match(args, config['validation_match'], source_device)
    val_masks = candidate_masks(validation, config)
    for half, mask in zip(validation, val_masks, strict=True):
        half.candidate_mask = mask
        make_features(half, config, training=False)
        half.xy = None
        gate_reports[half.key] = {'candidate_frames': int(mask.sum()),
                                 'predicted_inplay_frames': int((half.coarse['base'] != 2).sum())}
    train_refs = training_windows(training, [h.candidate_mask for h in training], config)
    val_refs, patches = inference_windows(validation, val_masks, config)
    if not train_refs or not val_refs:
        raise ValueError('No candidate training/validation windows under this fixed gate.')
    write_json(args.output / 'gate.json', {'threshold':config['gate_threshold'],
               'margin_frames':config['gate_margin_frames'], 'selection':'frozen previous validation-only gate',
               'ground_truth_used_for_gate_at_runtime':False, 'halves':gate_reports})
    write_json(args.output / 'ball_mask.json', reports)
    write_json(args.output / 'prepared.json', {'train_candidate_windows':len(train_refs),
               'ordinary_training_windows':0, 'candidate_only_loss':True,
               'validation_windows':len(val_refs), 'data_preparation_seconds':perf_counter() - tick})
    print(f'Candidate-only training: {len(train_refs):,} windows; ordinary=0; '
          f'validation={len(val_refs):,}; preparation={perf_counter() - tick:.1f}s',flush=True)
    return training, validation, train_refs, val_masks, val_refs, patches


def train(args, config, parts, model, optimizer, scaler, weight, device):
    training, validation, train_refs, val_masks, val_refs, patches = parts
    history, best_loss, bad_epochs, lr_wait = [], math.inf, 0, 0
    rng = np.random.default_rng(config['seed'])
    last = args.output / 'last.pt'
    if last.exists():
        saved = torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':torch.cuda.set_rng_state(saved['cuda_rng'],device)
        history,best_loss,bad_epochs,lr_wait = [saved[k] for k in ('history','best_loss','bad_epochs','lr_wait')]
        print(f'Resuming after epoch {len(history)}; patience={bad_epochs}/{config["early_stop"]}',flush=True)
    for epoch in range(len(history) + 1,config['max_epochs'] + 1):
        if bad_epochs >= config['early_stop']:break
        tick = perf_counter()
        order = rng.permutation(len(train_refs))
        if config['epoch_window_cap']:order = order[:config['epoch_window_cap']]
        refs = [train_refs[i] for i in order]
        model.train()
        losses = []
        for first in range(0,len(refs),config['batch']):
            losses.append(train_update(model,optimizer,scaler,training,refs[first:first + config['batch']],
                                       config,device,weight,rng,args.microbatch))
            if first == 0 or (first // config['batch'] + 1) % 100 == 0:
                print(f'Epoch {epoch}/{config["max_epochs"]}: '
                      f'batch {first // config["batch"] + 1}/{math.ceil(len(refs)/config["batch"])}',flush=True)
        result,_ = specialist_evaluation(model,validation,val_masks,val_refs,patches,config,device,weight,args.microbatch)
        value = result['validation_loss']
        if not math.isfinite(value):raise FloatingPointError('Nonfinite validation loss.')
        if value < best_loss:
            best_loss,bad_epochs,lr_wait = value,0,0
            save_checkpoint(args.output / 'best.pt',model.state_dict())
            write_json(args.output / 'best_validation.json',{'epoch':epoch,'selection':'minimum candidate validation loss',
                                                            'scores':result})
        else:
            bad_epochs += 1
            lr_wait += 1
            if lr_wait >= config['lr_patience']:
                for group in optimizer.param_groups:group['lr'] = max(config['min_lr'],group['lr'] * .5)
                lr_wait = 0
        row = {'epoch':epoch,'validation_loss':value,'validation_spellF1':result['refined']['spellF1'],
               'base_validation_spellF1':result['base']['spellF1'], 'bad_epochs':bad_epochs,
               'lr':optimizer.param_groups[0]['lr'],'seconds':perf_counter()-tick,
               'train_windows':len(refs),'optimizer_steps':len(losses),
               'training_loss_components':np.mean(losses,axis=0).tolist()}
        history.append(row)
        checkpoint = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                      'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'history':history,
                      'best_loss':best_loss,'bad_epochs':bad_epochs,'lr_wait':lr_wait}
        if device.type == 'cuda':checkpoint['cuda_rng'] = torch.cuda.get_rng_state(device)
        save_checkpoint(last,checkpoint)
        write_json(args.output / 'training.json',history)
        pd.json_normalize(history).to_csv(args.output / 'training_history.csv',index=False,encoding='utf-8-sig')
        print(json.dumps(row),flush=True)
        eta = np.median([x['seconds'] for x in history[-3:]]) * (config['max_epochs'] - epoch) / 60
        print(f'Maximum-budget ETA: {eta:.1f} min; early-stop patience {bad_epochs}/{config["early_stop"]}',flush=True)
    write_json(args.output / 'training_completed.json',{'completed_epochs':len(history),'max_epochs':config['max_epochs'],
               'early_stopped':bad_epochs >= config['early_stop'],'best_validation_loss':best_loss})


def test(args,config,model,weight,device,source_device):
    model.load_state_dict(torch.load(args.output / 'best.pt',map_location=device,weights_only=True))
    halves = load_match(args,config['test_match'],source_device)
    masks = candidate_masks(halves,config)
    for h,mask in zip(halves,masks,strict=True):
        h.candidate_mask = mask
        make_features(h,config,training=False)
    refs,patches = inference_windows(halves,masks,config)
    result,predictions = specialist_evaluation(model,halves,masks,refs,patches,config,device,weight,args.microbatch)
    result.pop('validation_loss')
    result.pop('candidate_loss_components')
    identity = {'outside_candidate_unchanged':True,'D_mask_unchanged':True,
                'frozen_5fps_weights_unchanged':True,'training_ordinary_windows':0}
    for h,pred,mask in zip(halves,predictions,masks,strict=True):
        table = pd.DataFrame({'match':h.match,'half':h.number,'frame':h.frame,'seconds':h.time,
                              'truth':h.labels,'dead':h.coarse['base']==2,'predicted_dead':h.dead_prediction,
                              'base':h.coarse['base'],'refined':pred,'candidate':mask})
        table.to_parquet(args.output / f'estimated_D_primary_{h.key}.parquet',index=False)
        table.to_csv(args.output / f'estimated_D_primary_{h.key}.csv',index=False,encoding='utf-8-sig')
    write_json(args.output / 'holdout.json',{'estimated_D_primary':result,'invariants':identity})
    rows = [{'model':'frozen_5fps','dead_mask':'estimated_D_primary',**result['base']},
            {'model':'5fps_plus_candidate25','dead_mask':'estimated_D_primary',**result['refined']}]
    for row in rows:row['candidate_coverage'] = result['coverage']
    pd.json_normalize(rows).to_csv(args.output / 'holdout_comparison.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train','test'))
    parser.add_argument('--config',type=Path,default=CONFIGS / 'parameters_candidate25.csv')
    parser.add_argument('--output',type=Path,default=ROOT / 'data/transition_final/candidate25_40epochs')
    parser.add_argument('--cache',type=Path,default=ROOT / 'data/external/dfl')
    parser.add_argument('--sources',type=Path,default=ROOT / 'data/transition_final/source')
    parser.add_argument('--raw',type=Path,default=ROOT / 'data/external/dfl/raw')
    parser.add_argument('--upstream',type=Path,default=ROOT / 'pathcrf')
    parser.add_argument('--official-prepared',type=Path,default=ROOT / 'data/pathcrf_dfl/transition_head_v1/prepared')
    parser.add_argument('--frozen-cache',type=Path,default=ROOT / 'data/transition_final/candidate25_frozen')
    parser.add_argument('--dead-scene',type=Path,default=ROOT / 'data/pathcrf_dfl/scene_two_models_v1')
    parser.add_argument('--microbatch',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.device == 'cuda' and not torch.cuda.is_available():parser.error('Select a Colab GPU runtime.')
    if config['max_epochs'] < 1 or config['early_stop'] < 1 or not 1 <= args.microbatch <= config['batch']:
        parser.error('Invalid epoch, early-stop, or batch setting.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(config['seed'])
    device,source_device = torch.device(args.device),torch.device('cpu')
    config['amp'] = config['amp'] and device.type == 'cuda'
    args.output.mkdir(parents=True,exist_ok=True)
    split = {'train':config['train_matches'].split(),'validation':config['validation_match'],'test':config['test_match']}
    if len(set(split['train']+[split['validation'],split['test']])) != len(split['train'])+2:
        parser.error('Overlapping match split.')
    if json.loads((args.dead_scene / 'manifest.json').read_text())['split'] != split:
        parser.error('Dead-ball model split mismatch.')
    checkpoint = args.upstream / 'saved/140/model/state_dict_best_acc.pt'
    inputs = {'frozen_5fps_sha256':digest(checkpoint),'D_sha256':digest(args.dead_scene / 'full/best.pt'),
              'D_decoder_sha256':digest(args.dead_scene / 'full/fit.json'),
              'runner_sha256':digest(Path(__file__)),'refiner_sha256':digest(FOLDER / 'refiner.py')}
    manifest = {'experiment':'frozen 5fps plus candidate-only 25fps specialist','split':split,'inputs':inputs,
                'parameters':config,'microbatch':args.microbatch,'candidate_only_loss':True,
                'ordinary_training_windows':0,'base_model_retrained':False,
                'test_status':'previously inspected holdout; retrospective comparison'}
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:parser.error('Use a new RUN_NAME for changed settings.')
    if args.stage == 'test' and not path.exists():parser.error('Missing trained run manifest.')
    write_json(path,manifest)
    with (args.output / 'parameters.csv').open('w',newline='',encoding='utf-8-sig') as out:
        writer = csv.writer(out);writer.writerow(['parameter','value']);writer.writerows(config.items())
    write_json(args.output / 'runtime_info.json',{'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
               'torch':str(torch.__version__),'amp':config['amp'],'amp_dtype':config['amp_dtype'],'device':str(device)})
    model = Refiner(config,N_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda',enabled=config['amp'] and config['amp_dtype']=='float16')
    if args.stage == 'test':
        saved = json.loads((args.output / 'transition_weights.json').read_text())
        weight = torch.tensor(saved['weights'],dtype=torch.float32,device=device)
        test(args,config,model,weight,device,source_device)
    else:
        parts = prepare(args,config,source_device)
        weight,counts = class_weights(parts[0],config,device)
        write_json(args.output / 'transition_weights.json',{'counts':counts.tolist(),'weights':weight.cpu().tolist(),
                                                          'scope':'candidate supervised frames only'})
        torch.manual_seed(config['seed'])
        train(args,config,parts,model,optimizer,scaler,weight,device)
    if digest(checkpoint) != inputs['frozen_5fps_sha256']:raise AssertionError('Frozen 5fps checkpoint changed.')


if __name__ == '__main__':main()
