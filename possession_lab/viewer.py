"""Local, synchronized video reviewer for the possession predictions.

The reviewer is intentionally boring infrastructure: one local HTTP server, one static page, and
one MP4.  It never copies the gated video and it never writes predictions.  The browser maps the
tactical frame timeline to rows of ``predictions.csv``; the public ground truth is drawn only where
it exists.

The reviewer also lets a human write a **user truth (UT)** used only to score the fixed prediction
in the UI.  The UT is stored in ``outputs/user_truth.json`` between sessions.  Nothing in the
predict/train path imports this module, so the model can never see the UT.

Command::

    .venv/bin/python -m possession_lab viewer

The server listens on ``127.0.0.1`` and only exposes the configured MP4, the static UI, and the
viewer JSON.  Byte-range requests are implemented explicitly because the tactical MP4 is 5 GB and
its ``moov`` atom is at the end of the file.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .config import (
    A,
    CODE_TO_LABEL,
    D,
    FPS,
    GAME,
    GRADED_HALVES,
    H,
    LABEL_TO_CODE,
    N_FRAMES,
    PUBLIC_GT_PATH,
    PUBLIC_START,
    PUBLIC_STOP,
    ROOT,
    SOURCE,
    USER_TRUTH_PATH,
    X,
    asset_paths,
)
from .data import TagTable, load_public_gt, load_smart_tags, load_submission

# The official scorer is the source of truth for possession segments/spells, just as it is for the
# report.  Import it lazily-safe by putting the repository root on sys.path first.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402


RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
MIME_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "viewer.js": "text/javascript; charset=utf-8",
    "viewer.css": "text/css; charset=utf-8",
}


def _labels_from_codes(codes: np.ndarray) -> str:
    return "".join(CODE_TO_LABEL[int(code)] for code in codes)


def _validate_codes(codes: np.ndarray, name: str) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.ndim != 1 or codes.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not set(np.unique(codes).tolist()).issubset(set(CODE_TO_LABEL)):
        raise ValueError(f"{name} contains labels outside {{H,A,D,X}}")
    return codes.astype(np.int8, copy=False)


def _metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compact metric fields shared by the public and user-truth stats."""
    return {
        "spell_f1": float(metrics["spell_f1_mean"]),
        "macro_f1": float(metrics["macro_f1"]),
        "d_f1": float(metrics["per_class"]["D"]["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "n_pred_spells": int(metrics["n_pred_spells"]),
        "n_gt_spells": int(metrics["n_gt_spells"]),
        "n_graded": int(metrics.get("n_graded", 0)),
    }


def build_viewer_data(
    pred: np.ndarray,
    gt: np.ndarray | None = None,
    tags: TagTable | Any | None = None,
    metrics: dict[str, Any] | None = None,
    *,
    fps: int = FPS,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
    public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP),
    start_frame: int | None = None,
) -> dict[str, Any]:
    """Build the compact payload consumed by the browser.

    ``pred`` is a full-match array.  ``gt`` is optional; when present, its non-X runs define the
    graded sections.  All spell/segment boundaries come from the official scorer rather than a
    second run-length implementation.
    """
    pred = _validate_codes(pred, "pred")
    n_frames = int(pred.size)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if public[0] < 0 or public[1] > n_frames or public[0] >= public[1]:
        raise ValueError(f"invalid public window {public} for {n_frames} frames")
    for start, stop in halves:
        if start < 0 or stop > n_frames or start >= stop:
            raise ValueError(f"invalid graded half {(start, stop)} for {n_frames} frames")

    pred_runs = [tuple(map(int, half)) for half in halves]
    gt_runs: list[tuple[int, int]] = []
    if gt is not None:
        gt = _validate_codes(gt, "gt")
        if gt.size != n_frames:
            raise ValueError(f"gt has {gt.size} frames; pred has {n_frames}")
        gt_runs = pm.valid_runs(gt)
        if not gt_runs:
            gt_runs = pred_runs

    # Prediction is evaluated over both full graded halves.  Public GT only exists in its public
    # window, so its own runs are used exclusively for the dashed GT band.
    pred_segments = [[int(a), int(b), int(c)] for a, b, c in pm.segments(pred, pred_runs)]
    pred_spells = [[int(a), int(b), int(c)] for a, b, c in pm.possession_spells(pred, pred_runs)]

    gt_segments: list[list[int]] = []
    gt_spells: list[list[int]] = []
    disagreements: list[int] = []
    if gt is not None:
        gt_segments = [[int(a), int(b), int(c)] for a, b, c in pm.segments(gt, gt_runs)]
        gt_spells = [[int(a), int(b), int(c)] for a, b, c in pm.possession_spells(gt, gt_runs)]
        disagreements = np.flatnonzero((gt != X) & (pred != gt)).astype(int).tolist()

    changes: set[int] = set()
    for start, stop in pred_runs:
        start, stop = int(start), int(stop)
        window = pred[start:stop]
        if window.size > 1:
            changes.update((np.flatnonzero(window[1:] != window[:-1]) + 1 + start).tolist())
        if start > 0:
            changes.add(start)
        if stop < n_frames:
            changes.add(stop)

    tags_payload: list[dict[str, Any]] = []
    for event in getattr(tags, "events", ()) if tags is not None else ():
        tags_payload.append(
            {
                "start": int(event.start_frame),
                "stop": int(event.stop_frame),
                "code": str(event.code),
                "team": int(event.team),
                "type": str(event.type),
                "labels": {str(key): str(value) for key, value in event.labels.items()},
            }
        )

    if start_frame is None:
        start_frame = int(public[0])
    start_frame = max(0, min(int(start_frame), n_frames - 1))

    counts = {
        label: int(np.sum(pred == code))
        for label, code in (("H", H), ("A", A), ("D", D), ("X", X))
    }
    metric_payload: dict[str, Any] | None = None
    if metrics is not None:
        metric_payload = _metrics_payload(metrics)

    return {
        "version": 1,
        "game": GAME,
        "source": SOURCE,
        "fps": int(fps),
        "n_frames": n_frames,
        "duration_s": float(n_frames / fps),
        "public": [int(public[0]), int(public[1])],
        "halves": [[int(start), int(stop)] for start, stop in halves],
        "start_frame": int(start_frame),
        "labels": _labels_from_codes(pred),
        "gt": _labels_from_codes(gt) if gt is not None else None,
        "gt_available": gt is not None,
        "label_counts": counts,
        "pred_segments": pred_segments,
        "gt_segments": gt_segments,
        "pred_spells": pred_spells,
        "gt_spells": gt_spells,
        "changes": sorted(changes),
        "disagreements": disagreements,
        "tags": tags_payload,
        "metrics": metric_payload,
    }


def load_user_truth(path: str | Path | None, n_frames: int = N_FRAMES) -> np.ndarray:
    """Load the viewer-only user truth, or an all-``X`` array when it does not exist yet."""
    user_truth = np.full(int(n_frames), X, dtype=np.int8)
    if path is None:
        return user_truth
    path = Path(path)
    if not path.is_file():
        return user_truth
    document = json.loads(path.read_text(encoding="utf-8"))
    runs = document.get("runs") if isinstance(document, dict) else None
    if not isinstance(runs, list):
        raise ValueError(f"{path}: expected a JSON object with a 'runs' list")
    for run in runs:
        if not isinstance(run, list) or len(run) != 3:
            raise ValueError(f"{path}: every run must be [start, stop, label]")
        start, stop, label = int(run[0]), int(run[1]), str(run[2]).strip().upper()
        if label not in LABEL_TO_CODE:
            raise ValueError(f"{path}: unknown label {label!r}")
        if not (0 <= start < stop <= int(n_frames)):
            raise ValueError(f"{path}: run [{start}, {stop}) outside 0..{n_frames}")
        user_truth[start:stop] = LABEL_TO_CODE[label]
    return user_truth


def save_user_truth(path: str | Path, user_truth: np.ndarray) -> Path:
    """Persist the user truth as compact constant-label runs (atomic replace)."""
    path = Path(path)
    user_truth = _validate_codes(user_truth, "user_truth")
    runs = [
        [int(start), int(stop), CODE_TO_LABEL[int(code)]]
        for start, stop, code in pm.segments(user_truth, pm.valid_runs(user_truth))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "game": GAME, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def move_user_truth_boundary(
    user_truth: np.ndarray, from_frame: int, to_frame: int
) -> np.ndarray:
    """Move a UT run boundary in place.

    ``from_frame`` must sit between two different labels (``0`` and ``n_frames`` count as
    boundaries when the edge label differs from ``X``).  Moving right gives the left label the
    freed frames; moving left gives them to the right label.
    """
    user_truth = np.asarray(user_truth)
    n = int(user_truth.size)
    from_frame, to_frame = int(from_frame), int(to_frame)
    if not (0 <= from_frame <= n and 0 <= to_frame <= n):
        raise ValueError(f"boundary {from_frame} or target {to_frame} outside 0..{n}")
    left = user_truth[from_frame - 1] if from_frame > 0 else X
    right = user_truth[from_frame] if from_frame < n else X
    if left == right:
        raise ValueError(f"frame {from_frame} is not a user-truth boundary")
    if to_frame > from_frame:
        user_truth[from_frame:to_frame] = left
    elif to_frame < from_frame:
        user_truth[to_frame:from_frame] = right
    return user_truth


def build_user_truth_data(pred: np.ndarray, user_truth: np.ndarray) -> dict[str, Any]:
    """Score the fixed prediction against the user truth.

    This is the only place the UT is consumed.  ``pred`` is never modified, and the model path
    (features/model/pipeline) has no access to this function or the UT file.
    """
    pred = _validate_codes(pred, "pred")
    user_truth = _validate_codes(user_truth, "user_truth")
    if user_truth.size != pred.size:
        raise ValueError(f"user truth has {user_truth.size} frames; pred has {pred.size}")

    runs = pm.valid_runs(user_truth)
    labeled = int(np.count_nonzero(user_truth != X))
    metrics = pm.evaluate(user_truth, pred, bootstrap_n=0) if labeled else None
    return {
        "ut_segments": [[int(a), int(b), int(c)] for a, b, c in pm.segments(user_truth, runs)],
        "ut_labeled_frames": labeled,
        "ut_metrics": _metrics_payload(metrics) if metrics is not None else None,
    }


def load_viewer_data(
    predictions_path: str | Path,
    *,
    gt_path: str | Path | None = PUBLIC_GT_PATH,
    tags_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    user_truth_path: str | Path | None = USER_TRUTH_PATH,
    start_frame: int | None = None,
) -> dict[str, Any]:
    """Read the generated CSV outputs and build the viewer payload."""
    pred = load_submission(predictions_path)
    if pred.size != N_FRAMES:
        raise ValueError(f"{predictions_path}: expected {N_FRAMES:,} rows, got {pred.size:,}")

    gt = None
    if gt_path is not None and Path(gt_path).is_file():
        gt = load_public_gt(gt_path)
        if gt.size != pred.size:
            raise ValueError(f"{gt_path}: expected {pred.size:,} rows, got {gt.size:,}")

    tags = None
    if tags_path is not None and Path(tags_path).is_file():
        tags = load_smart_tags(tags_path, n_frames=pred.size)

    metrics = None
    if metrics_path is not None and Path(metrics_path).is_file():
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

    data = build_viewer_data(
        pred,
        gt,
        tags=tags,
        metrics=metrics,
        start_frame=start_frame,
    )
    data.update(build_user_truth_data(pred, load_user_truth(user_truth_path, pred.size)))
    return data


def tactical_video_path(data_dir: str | Path | None = None) -> Path:
    """Return the tactical MP4 next to the gated ATD files."""
    return asset_paths(data_dir)["atd_raw"].with_name(f"{GAME}_{SOURCE}.mp4")


def parse_byte_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse a single-range ``Range`` header.

    Returns ``None`` when there is no range, or ``(start, end)`` with inclusive bounds.  Raises
    ``ValueError`` for malformed, multi-range, or unsatisfiable requests.
    """
    if not header:
        return None
    match = RANGE_RE.fullmatch(header.strip())
    if not match:
        raise ValueError("unsupported range")
    first, last = match.groups()
    if first == "" and last == "":
        raise ValueError("empty range")
    if size <= 0:
        raise ValueError("empty file")

    if first == "":
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("empty suffix")
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(first)
        end = size - 1 if last == "" else int(last)
        if start >= size:
            raise ValueError("range starts past end")
        end = min(end, size - 1)
        if end < start:
            raise ValueError("range end before start")
    return start, end


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """Small whitelist-only HTTP handler with MP4 byte-range support."""

    protocol_version = "HTTP/1.1"
    payload: bytes = b"{}"
    video_path: Path | None = None
    assets: dict[str, tuple[bytes, str]] = {}
    pred: np.ndarray | None = None
    user_truth: np.ndarray | None = None
    user_truth_path: Path | None = None
    ut_lock = threading.Lock()
    verbose = False

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.verbose:
            super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch(send_body=False)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if urlsplit(self.path).path != "/api/user-truth":
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        if self.pred is None or self.user_truth is None or self.user_truth_path is None:
            self._send_error(HTTPStatus.NOT_FOUND, "user truth not configured")
            return
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            self._send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError
            n_frames = int(self.pred.size)
            clear = bool(body.get("clear"))
            boundary = target = start = stop = code = None
            if clear:
                pass
            elif "boundary" in body:
                boundary, target = int(body["boundary"]), int(body["to"])
                if not (0 <= boundary <= n_frames and 0 <= target <= n_frames):
                    raise ValueError
            else:
                start, stop = int(body["start"]), int(body["stop"])
                code = LABEL_TO_CODE[str(body["label"]).strip().upper()]
                if not (0 <= start < stop <= n_frames):
                    raise ValueError
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid user-truth edit")
            return

        try:
            with self.ut_lock:
                backup = self.user_truth.copy()
                try:
                    if clear:
                        self.user_truth[:] = X
                    elif boundary is not None:
                        move_user_truth_boundary(self.user_truth, boundary, target)
                    else:
                        self.user_truth[start:stop] = code
                    save_user_truth(self.user_truth_path, self.user_truth)
                except (OSError, ValueError):
                    self.user_truth[:] = backup
                    raise
                ut_payload = build_user_truth_data(self.pred, self.user_truth)
                merged = json.loads(self.payload)
                merged.update(ut_payload)
                merged_payload = json.dumps(
                    merged, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self.__class__.payload = merged_payload
        except ValueError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid user-truth boundary")
            return
        except OSError:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "could not save user truth")
            return

        self._send_bytes(
            json.dumps(ut_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            send_body=True,
            cache="no-store",
        )

    def _dispatch(self, *, send_body: bool) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            entry = self.assets.get("/")
            if entry is None:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "viewer index not configured")
            else:
                self._send_bytes(entry[0], entry[1], send_body=send_body)
            return
        if path == "/api/viewer-data":
            self._send_bytes(
                self.payload,
                "application/json; charset=utf-8",
                send_body=send_body,
                cache="no-store",
            )
            return
        if path == "/video":
            self._serve_video(send_body=send_body)
            return
        entry = self.assets.get(path)
        if entry is not None:
            self._send_bytes(entry[0], entry[1], send_body=send_body)
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found", send_body=send_body)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        *,
        send_body: bool,
        cache: str = "no-cache",
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str, *, send_body: bool = True) -> None:
        body = (message + "\n").encode("utf-8")
        self._send_bytes(
            body,
            "text/plain; charset=utf-8",
            send_body=send_body,
            cache="no-store",
            status=status,
        )

    def _serve_video(self, *, send_body: bool) -> None:
        path = self.video_path
        if path is None or not Path(path).is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "video not found", send_body=send_body)
            return
        size = Path(path).stat().st_size
        range_header = self.headers.get("Range", "")
        if self.verbose:
            print(f"video request: range={range_header!r} size={size:,}", flush=True)
        try:
            parsed = parse_byte_range(range_header, size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if parsed is None:
            start, end = 0, max(0, size - 1)
            status = HTTPStatus.OK
        else:
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT

        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Accept-Ranges", "bytes")
        if parsed is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        if not send_body or size == 0:
            return
        try:
            with Path(path).open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return


class ViewerHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The browser aborts the initial open-ended video request after it reads the header/moov
        # range.  That is normal for a 5 GB MP4 and should not dump a traceback into the terminal.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def _load_assets(assets_dir: Path) -> dict[str, tuple[bytes, str]]:
    assets: dict[str, tuple[bytes, str]] = {}
    for name, mime in MIME_TYPES.items():
        path = assets_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"viewer asset not found: {path}")
        route = "/" if name == "index.html" else f"/static/{name}"
        assets[route] = (path.read_bytes(), mime)
    return assets


def make_server(
    host: str,
    port: int,
    *,
    payload: bytes,
    video_path: str | Path,
    assets_dir: str | Path,
    pred: np.ndarray | None = None,
    user_truth: np.ndarray | None = None,
    user_truth_path: str | Path | None = None,
) -> ViewerHTTPServer:
    """Create the local reviewer server.  ``port=0`` chooses an ephemeral port."""
    assets = _load_assets(Path(assets_dir))
    attributes: dict[str, Any] = {
        "payload": payload,
        "video_path": Path(video_path),
        "assets": assets,
    }
    if pred is not None and user_truth is not None and user_truth_path is not None:
        attributes.update(
            {
                "pred": np.array(pred, dtype=np.int8, copy=True),
                "user_truth": np.array(user_truth, dtype=np.int8, copy=True),
                "user_truth_path": Path(user_truth_path),
            }
        )
    handler = type("BoundViewerRequestHandler", (ViewerRequestHandler,), attributes)
    return ViewerHTTPServer((host, port), handler)


def serve_viewer(
    predictions_path: str | Path,
    *,
    video_path: str | Path | None = None,
    gt_path: str | Path | None = PUBLIC_GT_PATH,
    tags_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    user_truth_path: str | Path | None = USER_TRUTH_PATH,
    data_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    start_frame: int | None = None,
    verbose: bool = False,
) -> int:
    """Build the payload, serve the reviewer, and block until Ctrl-C."""
    video = Path(video_path) if video_path is not None else tactical_video_path(data_dir)
    if not video.is_file():
        raise FileNotFoundError(
            f"tactical video not found: {video}\n"
            "Unpack the gated bundle under data/ or pass --video."
        )
    predictions = Path(predictions_path)
    if not predictions.is_file():
        raise FileNotFoundError(
            f"predictions not found: {predictions}\n"
            "Run `python -m possession_lab all` first or pass --predictions."
        )

    data = load_viewer_data(
        predictions,
        gt_path=gt_path,
        tags_path=tags_path,
        metrics_path=metrics_path,
        user_truth_path=user_truth_path,
        start_frame=start_frame,
    )
    pred = load_submission(predictions)
    user_truth = load_user_truth(user_truth_path, pred.size)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assets_dir = Path(__file__).resolve().parent / "viewer_assets"
    server = make_server(
        host,
        port,
        payload=payload,
        video_path=video,
        assets_dir=assets_dir,
        pred=pred,
        user_truth=user_truth,
        user_truth_path=user_truth_path,
    )
    ViewerRequestHandler.verbose = verbose
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"viewer: {url}", flush=True)
    print(
        f"data: {data['n_frames']:,} frames | {data['duration_s']:.2f}s | "
        f"{len(data['pred_spells'])} predicted spells | "
        f"GT {'yes' if data['gt_available'] else 'no'} | "
        f"UT {data['ut_labeled_frames']:,} frames",
        flush=True,
    )
    print(
        f"video: {video} ({video.stat().st_size / (1024**3):.2f} GiB, byte-range streaming)",
        flush=True,
    )
    print("Ctrl-C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nviewer stopped.")
    finally:
        server.server_close()
    return 0
