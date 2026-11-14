"""One bounded, Validation-only A-model audit/ablation; no Test/GT access."""
import argparse
import csv
import hashlib
import json
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from .a_diagnostic_models import TokenSpatialBiLSTM, SmallTemporal, MixedSpanMasker, context_gradient_probe
from .corruption import SpanMasker
from .model_a import SpatialBiLSTM, load_checkpoint
from .model_a_data import FeatureWindowDataset, build_feature_cache, sampled_indices, batch_arrays
from .spatial_schema import FEATURE_NAMES, TARGET_NAMES
from .temporal_cli import open_temporal_dataset
from .train_model_a import set_seed, select_device, train_update, evaluate, EarlyStopping


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def restore_variant(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    constructors = dict(continued_short=SpatialBiLSTM, token_short=TokenSpatialBiLSTM,
                        token_mixed=TokenSpatialBiLSTM, small_temporal_mixed=SmallTemporal)
    model = constructors[state["kind"]]()
    model.load_state_dict(state["model_state"], strict=True)
    return model.to(device), state


def save_variant(path, model, kind, **metadata):
    torch.save(dict(version="a-diagnostic-v1", kind=kind, metadata=metadata,
                    model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}), path)


def validation_score(results):
    return float(np.mean([results[m]["metrics"]["bilstm"]["huber"] for m in ("8-13","25","50")]))


def choose_candidate(results):
    """Rule fixed in configuration BEFORE additional training; no Test scores."""
    reference = results["original"]
    name = min((k for k in results if k != "original"), key=lambda k: validation_score(results[k]))
    candidate = results[name]
    accept = validation_score(candidate) < .98*validation_score(reference) \
        and candidate["8-13"]["metrics"]["bilstm"]["huber"] <= 1.05*reference["8-13"]["metrics"]["bilstm"]["huber"]
    return dict(best_candidate=name, adopted=name if accept else "original", accepted=bool(accept),
                original_composite=validation_score(reference), candidate_composite=validation_score(candidate),
                rule="composite Huber improves >=2%; short Huber degrades <=5%; Validation only")


def audit_submissions(output):
    """Read existing baseline labels, run the unchanged writer and validator.

    Do not re-read ball coordinates or run the baseline CLI's GT scoring tail.
    """
    from starter.baseline import VARIANTS, write_submission
    from evaluation import possession_metrics as pm
    from games import GAME, GAMES
    folder = output/"submission_audit"; folder.mkdir()
    checks = {}
    for name, _ in VARIANTS:
        original = Path("evaluation/results")/f"{name}.csv"
        table = pm.read_label_csv(original)
        reason = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
        if reason:
            raise AssertionError(f"Existing baseline submission invalid: {reason}")
        generated = folder/f"{name}.csv"
        write_submission(str(generated), table["label"].to_numpy())
        reread = pm.read_label_csv(generated)
        assert pm.validate_submission(reread, 199319) is None
        assert np.array_equal(table["label"], reread["label"])
        result = subprocess.run([sys.executable,"evaluation/validate_submission.py",str(generated),"--rows","199319"],
                                capture_output=True,text=True,check=True)
        (folder/f"{name}.validation.txt").write_text(result.stdout)
        checks[name] = dict(rows=len(reread),first_frame=int(reread["frame"].iloc[0]),
                            last_frame=int(reread["frame"].iloc[-1]), valid=True,labels_identical=True,
                            original_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
    return dict(checks=checks,gt_read=False,raw_baseline_ball_inference_rerun=False,
                scope="Existing full baseline labels -> original writer -> official format validator; not accuracy")


def run(temporal, cache_path, original_path, output_path, config):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Choose a new diagnostic output; old A is preserved")
    output.mkdir(parents=True)
    # Freeze the experimental budget and decision rule before any new metrics.
    write_json(output/"preregistered_config.json", config)
    (output/"LOCAL_ONLY.txt").write_text("Local match-trained diagnostics. No upload/push. No Test/GT/Smart Tagging.\n")
    set_seed(config["seed"],config["cpu_threads"])
    train_base = open_temporal_dataset(temporal,split="train")
    val_bases = {}
    for label,bounds in (("8-13",(8,13)),("25",(25,25)),("50",(50,50))):
        base = open_temporal_dataset(temporal,split="validation")
        base.masker = SpanMasker(config["seed"], *bounds)
        val_bases[label]=base
    cache=build_feature_cache(train_base,cache_path)
    train_data=FeatureWindowDataset(train_base,cache)
    val_data={m:FeatureWindowDataset(b,cache) for m,b in val_bases.items()}
    positions=sampled_indices(train_base.index,config["train_stride"])
    val_positions=sampled_indices(val_bases["8-13"].index,config["validation_stride"])
    device, device_info=select_device(config["device"])
    package=Path(__file__).parent
    write_json(output/"environment.json",dict(python=sys.version,torch=str(torch.__version__),
        numpy=np.__version__,platform=platform.platform(),device=device_info,
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}))
    def validation(model, baselines=False):
        return {m:evaluate(model,d,val_positions,config["batch_size"],device,
                           delta=config["huber_delta"],feature_weights=config["feature_weights"],baselines=baselines)
                for m,d in val_data.items()}
    reference,_=load_checkpoint(original_path,device)
    results={"original":validation(reference,True)}
    original_report=json.loads((Path(original_path).parent/"report.json").read_text())
    for label in results["original"]:
        old=original_report["evaluation"]["validation"][label]["metrics"]["bilstm"]
        new=results["original"][label]["metrics"]["bilstm"]
        assert old["valid_cells"]==new["valid_cells"]
        assert np.isclose(old["huber"],new["huber"],atol=1e-5,rtol=1e-5)
    metadata={"original":dict(parameters=sum(p.numel() for p in reference.parameters()),original_epochs=12)}
    # Verify actual past/future sensitivity on a hidden, valid interior target.
    for item in val_positions:
        window=val_data["8-13"][int(item)]
        target=val_data["8-13"].get_targets(int(item),window=window)
        times=np.flatnonzero(window.artificial_mask)
        if target.loss_mask.any() and times[0]>15 and times[-1]<135:
            probe=context_gradient_probe(reference,window.dense[None],window.padding_mask[None],int(times[len(times)//2]))
            break
    else:
        raise AssertionError("No eligible interior context probe")
    write_json(output/"context_probe.json",probe)
    print(f"Original Validation composite Huber={validation_score(results['original']):.6f}; device={device}",flush=True)
    for kind in config["run_plan"]:
        set_seed(config["seed"],config["cpu_threads"])
        initial,_=load_checkpoint(original_path)
        model = SmallTemporal() if kind=="small_temporal_mixed" else \
            TokenSpatialBiLSTM(initial) if kind.startswith("token_") else initial
        initial_token=model.mask_token.detach().clone() if hasattr(model,"mask_token") else None
        model=model.to(device)
        train_base.masker=MixedSpanMasker(config["seed"]) if "mixed" in kind else SpanMasker(config["seed"])
        optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
        stopper=EarlyStopping(config["patience"],config["min_delta"])
        cap=config["small_temporal_epochs"] if kind=="small_temporal_mixed" else config["additional_epochs"]
        folder=output/kind; folder.mkdir()
        history=[]; started=time.perf_counter()
        for epoch in range(cap):
            global_epoch=epoch if kind=="small_temporal_mixed" else 12+epoch
            train_data.set_epoch(global_epoch)
            model.train(); total=weight=0.; skipped=updates=0
            order=np.random.default_rng(config["seed"]+global_epoch).permutation(positions)
            for first in range(0,len(order),config["batch_size"]):
                x,padding,target,_=batch_arrays(train_data,order[first:first+config["batch_size"]])
                result=train_update(model,optimizer,x,padding,target,device,delta=config["huber_delta"],
                    feature_weights=config["feature_weights"],grad_clip=config["gradient_clip"])
                skipped+=int(result["skipped"])
                if not result["skipped"]:
                    total+=result["loss"]*result["weight_sum"]; weight+=result["weight_sum"]; updates+=1
            val=validation(model)
            objective=validation_score(val)
            improved,stop=stopper.step(objective)
            row=dict(epoch=epoch+1,global_epoch=global_epoch,train_huber=total/weight if weight else None,
                     validation_composite_huber=objective,validation_short_huber=val["8-13"]["metrics"]["bilstm"]["huber"],
                     updates=updates,skipped=skipped)
            history.append(row)
            if improved:
                save_variant(folder/"best.pt",model,kind,epoch=epoch+1,objective=objective,config=config)
            with (folder/"history.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            print(f"{kind} {epoch+1}/{cap}: val_composite={objective:.6f}, val_short={row['validation_short_huber']:.6f}",flush=True)
            if stop: break
        model,state=restore_variant(folder/"best.pt",device)
        results[kind]=validation(model,True)
        metadata[kind]=dict(parameters=sum(p.numel() for p in model.parameters()),epochs_completed=len(history),
            best_additional_epoch=state["metadata"]["epoch"],seconds=time.perf_counter()-started,
            warm_start_from_original=kind!="small_temporal_mixed",optimizer_state_reset=True,
            token_change_l2=float((model.mask_token.detach().cpu()-initial_token).norm()) if initial_token is not None else None)
        write_json(folder/"validation.json",results[kind])
    decision=choose_candidate(results)
    submission=audit_submissions(output)
    report=dict(version="a-diagnostic-once-v1",config=config,device=device_info,model_metadata=metadata,
                validation=results,decision=decision,context_probe=probe,submissions=submission,
                train_windows=len(positions),validation_windows=len(val_positions),test_used=False,
                scaler_refitted=False,source_fingerprint=train_base.source.fingerprint,
                split_fingerprint=train_base.normalizer.manifest_fingerprint)
    write_json(output/"training_report.json",report)
    print("Completed fixed experiment plan: "+json.dumps(decision),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="data/derived/relational_temporal_v1_20261109")
    parser.add_argument("--cache",default="data/derived/model_a_feature_cache_v1")
    parser.add_argument("--original",default="data/derived/model_a_run_v2_20261109/best.pt")
    parser.add_argument("--out-dir",required=True)
    parser.add_argument("--config",default="MTAH_Relational_Starter/configs/a_diagnostic_once.json")
    args=parser.parse_args()
    run(args.data,args.cache,args.original,args.out_dir,json.loads(Path(args.config).read_text()))


if __name__=="__main__": main()
