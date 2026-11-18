"""One-command entry: prepared DFL NPZ -> frozen models -> corrections -> CSV."""
from pathlib import Path
import argparse,subprocess,sys

from locations import ARTIFACT, PACKAGE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='Default: bundled inputs/J03WR9.npz')
    parser.add_argument('--base-cache',type=Path)
    parser.add_argument('--dead',type=Path)
    parser.add_argument('--output',type=Path,default=ARTIFACT/'inference/possession.csv')
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    parser.add_argument('--batch',type=int,default=16)
    args=parser.parse_args()
    tracking=args.input or ARTIFACT/'inputs/J03WR9.npz'
    base=args.base_cache or (ARTIFACT/'inputs/J03WR9_base5.npz' if args.input is None else None)
    if not tracking.is_file():raise FileNotFoundError(f'{tracking}; first run prepare_dfl.py or supply --input.')
    command=[sys.executable,str(PACKAGE/'predict.py'),'--input',str(tracking),'--output',str(args.output),
             '--device',args.device,'--batch',str(args.batch)]
    if base:command += ['--base-cache',str(base)]
    if args.dead:command += ['--dead',str(args.dead)]
    subprocess.run(command,check=True)


if __name__=='__main__':main()
