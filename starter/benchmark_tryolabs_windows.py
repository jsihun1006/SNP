"""750 actual inference frames (10/30/60 file minutes), then shared V1/V2 replay.

No old private cache is required. Repeat the same command to resume; a resumed
run is explicitly marked as such rather than reported as a fresh benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
import tryolabs_full_match_v2 as full
from analyze_tryolabs_full_match_v2 import coverage


def benchmark(output,device="cuda",batch=1,residency="single"):
    output=Path(output).resolve()
    if not output.is_relative_to(full.OUT.resolve()):raise ValueError("Benchmark output must be under full_match_v2")
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["PYTHONHASHSEED"]="0"
    for name in ["OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS"]:env.setdefault(name,"2")
    records=[];started=time.monotonic();selected=full.resolve_device(device)
    for segment,start in enumerate([15000,45000,90000],1):
        target=output/f"segment_{segment}"
        before=0
        if (target/"run.sqlite").exists():
            db=full.connection(target);before=db.execute("SELECT COUNT(*) FROM detections").fetchone()[0];db.close()
        common=[sys.executable,str(full.ROOT/"starter/tryolabs_full_match_v2.py"),"--output",str(target),
                "--start",str(start),"--end",str(start+250),"--device",selected,"--batch",str(batch),"--model-residency",residency]
        times={}
        try:
            for stage in ["detect","process"]:
                command=common+["--stage",stage]+(["--chunk-size","125"] if stage=="process" else [])
                print(f"Segment {segment}: {stage} (source {start}..{start+249})",flush=True)
                t0=time.monotonic();subprocess.run(command,cwd=full.ROOT,env=env,check=True);times[stage]=time.monotonic()-t0
            cov=coverage(target)
            if not cov["complete"]:raise RuntimeError(f"Incomplete benchmark segment: {cov}")
            records.append(dict(segment=segment,coverage=cov,detector_cached_before=before,stage_seconds=times))
        except Exception as exc:
            full.atomic_json(output/"benchmark_failure.json",dict(segment=segment,error=str(exc),completed_segments=records))
            raise
    summary=dict(expected_frames=750,detected_frames=sum(r["coverage"]["detected_frames"] for r in records),
                 processed_frames=sum(r["coverage"]["processed_frames"] for r in records),
                 fresh_benchmark=all(r["detector_cached_before"]==0 for r in records),device=selected,batch=batch,model_residency=residency,
                 platform=platform.platform(),torch=str(torch.__version__),torch_cuda=torch.version.cuda,
                 elapsed_seconds=time.monotonic()-started,segments=records,
                 note="Wall time includes separate model startup per segment, decode, tracking, classification and verified video chunks; not accuracy.")
    full.atomic_json(output/"benchmark_summary.json",summary);print(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=full.OUT/"windows_benchmark")
    p.add_argument("--device",default="cuda");p.add_argument("--batch",type=int,default=1)
    p.add_argument("--model-residency",choices=["single","both"],default="single")
    a=p.parse_args()
    if a.batch<1:p.error("Batch must be >=1")
    benchmark(a.output,a.device,a.batch,a.model_residency)


if __name__=="__main__":main()
