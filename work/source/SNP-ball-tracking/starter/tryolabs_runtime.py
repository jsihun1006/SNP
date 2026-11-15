"""CPU/MPS/CUDA execution and OS file locks; no model or classifier policy changes."""

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import tempfile
import time

import torch

IS_WINDOWS = os.name == "nt"


def configure_cache():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mtah-tryolabs-mpl"))
    os.environ.setdefault("YOLOV5_AUTOINSTALL", "false")
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")  # Upstream v6.2 is case-sensitive on POSIX.


def resolve_device(requested="cpu"):
    name = str(requested)
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Check the active venv, NVIDIA driver, "
                               "torch 2.9.1+cu126 and torchvision 0.24.1+cu126. No CPU fallback was used.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist")
        return f"cuda:{index}"
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable in this process. Use --device cpu or enable local GPU access.")
    if device.type not in ("cpu", "mps"):
        raise ValueError(f"Unsupported device: {name}")
    return device.type


def memory_info(device):
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            return dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), total_bytes=torch.cuda.get_device_properties(device).total_memory)
    except Exception as exc:
        return dict(unavailable=str(exc))
    return {}


def log_event(path, kind, **details):
    record = dict(time_unix=time.time(), kind=kind, **details)
    if path is not None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def is_cuda_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower() and "cuda" in str(exc).lower())


def install_cpu_nms():
    import models.common as common
    if getattr(common.non_max_suppression, "_tryolabs_cpu", False):
        return
    original = common.non_max_suppression
    def per_image(prediction, *args, **kwargs):
        cpu = prediction.cpu()
        return [original(cpu[i:i+1], *args, **kwargs)[0] for i in range(len(cpu))]
    per_image._tryolabs_cpu = True
    common.non_max_suppression = per_image


class DetectorPair:
    """Share one detector pair across V1/V2, FP32 and unchanged size=640/NMS.

    single residency parks the inactive model on CPU, limiting 6GB VRAM pressure.
    OOM retries smaller microbatches; failure at one frame is raised, never cached.
    """
    def __init__(self, loader, assets, device="cpu", residency="auto", event_log=None):
        self.device = resolve_device(device)
        self.residency = ("single" if self.device.startswith("cuda") else "both") if residency == "auto" else residency
        if self.residency not in ("single", "both"):
            raise ValueError("Model residency must be auto, single or both")
        self.event_log = event_log
        self.models = []
        try:
            for name in ("yolov5x.pt", "ball.pt"):
                model = loader(Path(assets) / name, device="cpu")
                self.models.append(model)
                model.eval()
            self.models[0].classes = [0]
            install_cpu_nms()
            if self.residency == "both":
                for model in self.models:
                    model.to(self.device)
        except Exception as exc:
            log_event(event_log, "model_load_failed", device=self.device, residency=self.residency,
                      error=f"{type(exc).__name__}: {exc}", memory=memory_info(self.device))
            self.close()
            raise RuntimeError(f"Model load failed on {self.device}; no frames committed: {exc}") from exc

    def _predict_model(self, model, images):
        try:
            if self.residency == "single":
                model.to(self.device)
            with torch.inference_mode():
                result = model(images, size=640).pandas().xyxy
            return result  # NMS and DataFrames are on CPU before parking the model.
        finally:
            if self.residency == "single":
                model.to("cpu")
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    def predict(self, images, frame_ids=None):
        if not images:
            return [], []
        try:
            people = self._predict_model(self.models[0], images)
            balls = self._predict_model(self.models[1], images)
            return people, balls
        except Exception as exc:
            oom = self.device.startswith("cuda") and is_cuda_oom(exc)
            log_event(self.event_log, "cuda_oom" if oom else "inference_failed", device=self.device,
                      batch=len(images), frames=frame_ids, error=f"{type(exc).__name__}: {exc}", memory=memory_info(self.device))
            if not oom:
                raise RuntimeError(f"Inference failed on {self.device}; current batch is uncommitted: {exc}") from exc
            exc.__traceback__ = None  # Release failed forward's activation tensors before retry.
            for model in self.models:
                model.to("cpu")
            gc.collect(); torch.cuda.empty_cache()
            if self.residency == "both":
                self.residency = "single"
                log_event(self.event_log, "oom_retry_single_residency", batch=len(images))
                return self.predict(images, frame_ids)
            if len(images) == 1:
                raise RuntimeError("CUDA OOM at batch=1 with single-model residency. Close other GPU programs and "
                                   "repeat the same command/output. The failing frame was not committed. "
                                   "Changing requested device/batch/residency requires a new output directory.") from exc
            split = len(images) // 2
            ids = frame_ids or [None] * len(images)
            log_event(self.event_log, "oom_retry_smaller_batch", previous_batch=len(images), next_batch=split)
            a, b = self.predict(images[:split], ids[:split])
            c, d = self.predict(images[split:], ids[split:])
            return a+c, b+d

    def close(self):
        for model in self.models:
            try:model.to("cpu")
            except Exception:pass
        self.models.clear(); gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:torch.cuda.empty_cache()
            except Exception as exc:log_event(self.event_log,"cuda_cleanup_failed",error=str(exc))


@contextmanager
def file_lock(path):
    """Lock one byte with msvcrt on Windows; retain flock on macOS/Linux."""
    with Path(path).open("a+b") as f:
        if IS_WINDOWS:
            import msvcrt
            if f.seek(0, 2) == 0:
                f.write(b"0"); f.flush()
            f.seek(0)
            try:msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:raise RuntimeError("Another worker holds this stage lock") from exc
            try:yield
            finally:
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError("Another worker holds this stage lock") from exc
            try:yield
            finally:fcntl.flock(f, fcntl.LOCK_UN)
