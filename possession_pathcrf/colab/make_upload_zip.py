"""Zip the repo's preprocessed public DFL data for upload to Google Drive (MyDrive/pathcrf/)."""
import os
import zipfile

REPO = os.path.join(os.path.expanduser("~"), "Desktop", "pathcrf_repo", "data", "sportec")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload")
os.makedirs(OUT, exist_ok=True)
for name in ("tracking_processed", "event_processed"):
    path = os.path.join(OUT, name + ".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(os.path.join(REPO, name))):
            z.write(os.path.join(REPO, name, f), arcname=os.path.join(name, f))
    print(path, round(os.path.getsize(path) / 1e6, 1), "MB")
