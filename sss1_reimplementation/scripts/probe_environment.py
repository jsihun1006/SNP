"""Read-only host survey. Generates evidence, never installs packages."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(args):
    if not shutil.which(args[0]):
        return dict(available=False)
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(available=True, returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(available=True, error=str(e))


def probe():
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), project_root=str(ROOT),
                  repository_root=str(ROOT.parent), platform=platform.platform(), system=platform.system(),
                  python=sys.version, python_executable=sys.executable, architecture=platform.machine(),
                  windows_native_verified=platform.system()=="Windows", conda=command(["conda", "--version"]),
                  nvidia=command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"]),
                  nvidia_full=command(["nvidia-smi"]), cuda_toolkit=command(["nvcc", "--version"]), packages={})
    for name in ["numpy", "torch", "torchvision", "ultralytics", "mmengine", "mmcv", "mmpose", "mmdet", "cv2", "sahi", "sskit", "xtcocotools"]:
        try:
            if name == "ultralytics":
                import os
                os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"outputs/.ultralytics"))
            module = importlib.import_module(name)
            report["packages"][name] = dict(import_ok=True, version=getattr(module, "__version__", "unknown"))
        except Exception as e:
            report["packages"][name] = dict(import_ok=False, error=f"{type(e).__name__}: {e}")
    report["cuda"] = dict(available=False, gpu_training_allowed=False)
    try:
        import torch
        cuda = report["cuda"]
        cuda.update(available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                    compiled_arches=torch.cuda.get_arch_list())
        if cuda["available"]:
            cuda["devices"] = [dict(name=torch.cuda.get_device_name(i),
                                    vram_bytes=torch.cuda.get_device_properties(i).total_memory,
                                    capability=torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
            require_cuda()
            cuda["gpu_training_allowed"] = True
    except Exception as e:
        report["cuda"]["error"] = str(e)
    try:
        import torch
        from torchvision.ops import nms
        nms(torch.tensor([[0.,0.,10.,10.]]), torch.tensor([.9]), .5)
        report["torchvision_nms_cpu"] = "passed"
    except Exception as e:
        report["torchvision_nms_cpu"] = str(e)
    try:
        import mmcv.ops
        from mmpose.models import RTMCCHead
        report["mmpose_model_import"] = "passed"
    except Exception as e:
        report["mmpose_model_import"] = f"{type(e).__name__}: {e}"
    return report


def require_cuda(device="cuda:0"):
    import torch
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("CUDA GPU training is blocked: run CPU unit tests; provision a supported GPU first")
    try:
        layer = torch.nn.Conv2d(3, 8, 3).to(device)
        layer(torch.ones(1,3,16,16,device=device)).sum().backward()
        torch.cuda.synchronize(device)
    except Exception as e:
        raise RuntimeError(f"GPU/PyTorch CUDA execution incompatible: {e}") from e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT/"outputs/environment.json")
    a = p.parse_args()
    report = probe()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
