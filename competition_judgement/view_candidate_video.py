"""Run the local candidate debugger from the project root without installation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'competition_judgement/src'), str(ROOT)]

from competition_judgement.view_candidate_video import main

if __name__ == '__main__':
    main()
