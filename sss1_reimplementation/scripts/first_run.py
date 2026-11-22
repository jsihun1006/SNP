"""Verify a freshly installed environment and produce a real detector smoke result."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def validate_environment(report, profile):
    required=["numpy","torch","torchvision","ultralytics","cv2","sahi","sskit","xtcocotools"]
    failures=[name for name in required if not report["packages"].get(name,{}).get("import_ok")]
    if report.get("torchvision_nms_cpu")!="passed":
        failures.append("torchvision CPU NMS")
    if profile=="CUDA118":
        if not report.get("cuda",{}).get("gpu_training_allowed"):
            failures.append("CUDA convolution/backward (GPU or driver not supported by this PyTorch build)")
        if report.get("mmpose_model_import")!="passed":
            failures.append("MMPose/MMCV extension")
    if failures:
        raise RuntimeError("Environment check failed: "+", ".join(failures))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",choices=["CoreCPU","CUDA118"],default="CoreCPU")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/"outputs/first_run"/args.profile
    output.mkdir(parents=True,exist_ok=True)
    status=dict(profile=args.profile,completed=False,steps=[],
                note="Public non-soccer detector sample only. No trained two-point pose model or SynLoc scores.")
    def run(module, arguments):
        subprocess.run([sys.executable,"-m",module,*map(str,arguments)],cwd=root.parent,check=True)
        status["steps"].append(module)
    try:
        # Release the probe's native runtimes/CUDA context before tests and inference.
        # This also avoids forking a parent that already initialized native libraries.
        run("sss1_reimplementation.scripts.probe_environment",["--output",output/"environment.json"])
        report=json.loads((output/"environment.json").read_text(encoding="utf-8"))
        validate_environment(report,args.profile)
        status["steps"].append("environment")
        test_args=[str(root/"tests"),"-q",f"--junitxml={output/'tests.xml'}"]
        if args.profile=="CoreCPU": test_args += ["-m","not model"]
        run("pytest",test_args)
        asset_args=["--only","all" if args.profile=="CUDA118" else "detection","--sample"]
        run("sss1_reimplementation.scripts.download_weights",asset_args)
        run("sss1_reimplementation.scripts.infer",[
            "--input",root/"data/bus.jpg","--detector",root/"checkpoints/yolo26l.pt",
            "--mode","baat","--device","cuda:0" if args.profile=="CUDA118" else "cpu",
            "--output",output/"sample"])
        if args.profile=="CUDA118":
            run("sss1_reimplementation.scripts.validate_pose_model",[
                "--checkpoint",root/"checkpoints/cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth",
                "--output",output/"pose_model.json"])
        status["completed"]=True
        print(f"First run passed. Results: {output}")
    except Exception as exc:
        status["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output/"status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__": main()
