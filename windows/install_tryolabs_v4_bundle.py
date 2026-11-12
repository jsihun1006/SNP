"""Install a verified code-only bundle: create missing files, never overwrite.

Run this from an extracted package, not from the Git checkout. Existing files
must already match; conflicts are reported before any target file is created.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_member(name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe bundle path: {name}")
    if p.parts[0] not in ("starter", "windows", "docs", "tests", "RULES.md", "requirements-tryolabs-v4-windows.txt"):
        raise ValueError(f"Non-code path is not permitted: {name}")
    if p.suffix.lower() not in (".py", ".json", ".cmd", ".md", ".txt"):
        raise ValueError(f"Non-code bundle member: {name}")
    return p


def install(bundle, target, check=False):
    bundle, target = Path(bundle).resolve(), Path(target).resolve()
    manifest = json.loads((bundle/"bundle_manifest.json").read_text(encoding="utf-8"))
    pending = []; identical = []; conflicts = []; newline_equivalent = []
    for name, info in manifest["files"].items():
        safe_member(name)
        source, dest = bundle/name, target/name
        if source.is_symlink() or not source.resolve().is_relative_to(bundle):
            raise ValueError(f"Source escapes bundle: {name}")
        if not dest.resolve().is_relative_to(target):
            raise ValueError(f"Target path escapes project: {name}")
        if not source.is_file() or source.stat().st_size != info["size"] or sha(source) != info["sha256"]:
            raise ValueError(f"Bundle integrity check failed: {name}")
        if dest.exists():
            if dest.is_file() and sha(dest) == info["sha256"]: identical.append(name)
            elif dest.is_file() and dest.read_bytes().replace(b"\r\n",b"\n") == source.read_bytes().replace(b"\r\n",b"\n"):
                newline_equivalent.append(name)
            else: conflicts.append(name)
        else: pending.append((source,dest))
    if conflicts:
        raise ValueError("Existing files differ; NOTHING was installed. Preserve them and update the KKJ checkout or choose a new target:\n"
                         + "\n".join(conflicts))
    created = []
    if not check:
        try:
            for source,dest in pending:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open("xb") as f, source.open("rb") as g:
                    created.append(dest);shutil.copyfileobj(g,f)
        except Exception:
            for path in reversed(created): path.unlink()
            raise
    return dict(check_only=check,target=str(target),identical_files=len(identical),new_files=len(pending),
                newline_equivalent_files=len(newline_equivalent),existing_files_overwritten=0,data_files_installed=0)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",type=Path,default=Path(r"C:\Users\kjkim\mtah_gpu\SNP"))
    p.add_argument("--check",action="store_true")
    a=p.parse_args()
    try: print(json.dumps(install(Path(__file__).resolve().parents[1],a.target,a.check),indent=2))
    except (ValueError,OSError,KeyError) as e: p.exit(1,f"ERROR: {e}\n")
