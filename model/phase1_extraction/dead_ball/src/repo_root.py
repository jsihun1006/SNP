"""저장소 루트(games.py가 있는 폴더)를 찾는다. 이 코드는 model/phase1_extraction/dead_ball/src/에 있고,
데이터(data/)와 주최 측 코드(starter/, evaluation/)는 저장소 루트에 있다."""

import os


def find(start):
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, "games.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("repository root (folder with games.py) not found above " + start)
        d = parent


ROOT = find(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
PROJECT_SRC = os.path.dirname(os.path.abspath(__file__))
