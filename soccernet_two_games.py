"""
SoccerNet에서 지정한 경기들의 라벨·특징·자동추출 데이터를 골라 받는 스크립트

동작
  - SoccerNet 라벨/특징 저장소(SN-Labels, SN-Features)의 모든 브랜치를 훑어서
    지정한 경기 폴더에 속한 파일만 찾아요.
  - 기본은 목록만 보여주고(dry-run), --download를 붙이면 실제로 받아요.
  - 라벨은 영상과 같은 폴더(C:\\SoccerNet_data\\...)에 저장돼서
    soccernet_deadball.py를 바로 돌릴 수 있어요.
  - 특징 파일은 C:\\SoccerNet_data\\_features\\<브랜치>\\... 에 따로 저장해요.

사용법
  1) 목록 확인:  python soccernet_two_games.py
  2) 라벨만 받기: python soccernet_two_games.py --download --only labels
  3) 전부 받기:   python soccernet_two_games.py --download

사전 준비: hf auth login (Read 토큰)
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

GAMES = [
    "spain_laliga/2016-2017/2017-02-26 - 18-15 Atl. Madrid 1 - 2 Barcelona",
    "spain_laliga/2016-2017/2017-04-23 - 21-45 Real Madrid 2 - 3 Barcelona",
]

REPOS = {
    "labels": "SoccerNet/SN-Labels",
    "features": "SoccerNet/SN-Features",
}

ROOT = Path(r"C:\SoccerNet_data")


def human(n):
    if n is None:
        return "?"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def find_files(api: HfApi, repo_id: str):
    """모든 브랜치에서 지정 경기에 속한 파일 찾기 -> [(branch, path, size)]"""
    out = []
    try:
        branches = [b.name for b in api.list_repo_refs(repo_id, repo_type="dataset").branches]
    except Exception as e:
        print(f"  [건너뜀] {repo_id}: 접근 불가 ({type(e).__name__}: {e})")
        return out

    for br in branches:
        try:
            infos = api.list_repo_tree(repo_id, repo_type="dataset", revision=br, recursive=True)
            for info in infos:
                path = getattr(info, "path", "")
                if any(path.startswith(g + "/") or f"/{g}/" in path for g in GAMES):
                    out.append((br, path, getattr(info, "size", None)))
        except Exception as e:
            print(f"  [건너뜀] {repo_id}@{br}: {type(e).__name__}: {e}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="실제로 다운로드")
    p.add_argument("--only", choices=list(REPOS), help="한 종류만 처리")
    args = p.parse_args()

    api = HfApi()
    total = 0
    for kind, repo_id in REPOS.items():
        if args.only and kind != args.only:
            continue
        print(f"\n=== {kind}: {repo_id} ===")
        files = find_files(api, repo_id)
        if not files:
            print("  해당 경기 파일 없음")
            continue

        for br, path, size in files:
            total += size or 0
            print(f"  [{br}] {path}  ({human(size)})")

        if args.download:
            for br, path, size in files:
                # 라벨은 영상과 같은 폴더 구조로, 특징은 브랜치별 하위 폴더로
                local = ROOT if kind == "labels" else ROOT / "_features" / br
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=path,
                    revision=br,
                    local_dir=str(local),
                )
            print(f"  -> {len(files)}개 저장 완료")

    print(f"\n총 용량: {human(total)}")
    if not args.download:
        print("목록만 확인했어요. 받으려면 --download 를 붙여 다시 실행하세요.")


if __name__ == "__main__":
    main()
