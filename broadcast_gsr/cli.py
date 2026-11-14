import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys
from .io import load_config

DEFAULT_CONFIG=Path(__file__).resolve().parent/'configs/mtah_broadcast.json'


def doctor(cfg):
    issues=[];warnings=[]
    report={'platform':platform.platform(),'python':sys.version.split()[0],
            'windows_execution_tested_in_development':False,'dependencies':{},'inputs':{}}
    for name in ('numpy','scipy','cv2','torch','torchvision','loguru'):
        try:
            module=__import__(name)
            report['dependencies'][name]=getattr(module,'__version__','present')
        except Exception as exc:
            report['dependencies'][name]=str(exc);issues.append('Dependency unavailable: '+name)
    for key in ('video','calibration_csv','upstream','reid_source','detector_weights','reid_weights'):
        report['inputs'][key]={'path':cfg[key],'exists':Path(cfg[key]).exists()}
        if not Path(cfg[key]).exists():issues.append('Missing '+key)
    if cfg.get('jersey')=='clip':
        if not Path(cfg['clip_weights']).is_file():issues.append('Missing clip_weights')
        if importlib.util.find_spec('clip') is None:issues.append('Missing OpenAI CLIP package')
    for key,expected in [('upstream',__import__('broadcast_gsr').UPSTREAM_SHA),
                         ('reid_source',__import__('broadcast_gsr').REID_SHA)]:
        path=Path(cfg[key]) if key=='upstream' else Path(cfg[key]).parents[2]
        if path.exists():
            try:
                from .upstream import repository_state
                report[key+'_repository']=repository_state(path,expected)
            except (ValueError,subprocess.CalledProcessError,OSError) as exc:issues.append(str(exc))
    if 'torch' in sys.modules:
        import torch
        report['cuda_available']=torch.cuda.is_available()
        report['cuda_runtime']=torch.version.cuda
        if cfg.get('device','auto').startswith('cuda') and not torch.cuda.is_available():issues.append('CUDA requested but unavailable')
    if Path(cfg['video']).is_file() and 'cv2' in sys.modules:
        import cv2
        cap=cv2.VideoCapture(cfg['video'])
        report['video_metadata']={'opened':cap.isOpened(),'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps':cap.get(cv2.CAP_PROP_FPS),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        cap.release()
        if not report['video_metadata']['opened']:issues.append('Cannot open video')
        if Path(cfg['calibration_csv']).is_file():
            import csv
            with Path(cfg['calibration_csv']).open(newline='',encoding='utf-8-sig') as handle:
                count=sum(1 for _ in csv.DictReader(handle))
            report['calibration_rows']=count
            if count!=report['video_metadata']['frames']:
                warnings.append('Calibration row count differs from MP4 metadata; do not assume a one-frame correction')
    report.update(ready_for_attempt=not issues,issues=issues,warnings=warnings,model_accuracy_validated=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if not issues else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description='Independent Windows Broadcast2Pitch adapters; no possession prediction')
    sub=parser.add_subparsers(dest='command',required=True)
    boot=sub.add_parser('bootstrap',help='Download pinned source code, not model weights')
    boot.add_argument('--vendor',type=Path,default=Path(__file__).resolve().parent/'vendor')
    smoke=sub.add_parser('smoke',help='Synthetic video/replay I/O check; no model inference')
    smoke.add_argument('--upstream',type=Path,default=Path(__file__).resolve().parent/'vendor/SoccernetGSR')
    smoke.add_argument('--out',type=Path,required=True)
    compare=sub.add_parser('compare-atd',help='Spatial disagreement diagnostic; ATD is not GT')
    compare.add_argument('--tracks',required=True,type=Path)
    compare.add_argument('--atd',required=True,type=Path)
    compare.add_argument('--out',required=True,type=Path)
    compare.add_argument('--atd-offset',type=int,default=0)
    compare.add_argument('--alignment-verified',action='store_true')
    compare.add_argument('--gate-m',type=float,default=5.)
    compare.add_argument('--pitch-length-m',type=float,default=105.3)
    compare.add_argument('--pitch-width-m',type=float,default=68.)
    for command in ('doctor','run','calibrate'):
        p=sub.add_parser(command)
        p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
        p.add_argument('--video')
        p.add_argument('--device',choices=['auto','cpu','cuda'])
        p.add_argument('--start-frame',type=int)
        p.add_argument('--max-frames',type=int)
        if command!='doctor':p.add_argument('--out',type=Path)
        if command=='run':
            p.add_argument('--calibration-csv')
            p.add_argument('--calibration-origin',choices=['provided','learned_sfr'])
            p.add_argument('--jersey',choices=['none','clip'])
            p.add_argument('--refine',action='store_true',default=None)
            p.add_argument('--no-overlay',action='store_true')
            p.add_argument('--backend',choices=['official','replay'],default='official')
            p.add_argument('--replay')
    args=parser.parse_args(argv)
    try:
        if args.command=='bootstrap':
            from .upstream import bootstrap
            bootstrap(args.vendor);return 0
        if args.command=='smoke':
            from .smoke import smoke
            smoke(args.upstream,args.out);return 0
        if args.command=='compare-atd':
            if args.gate_m<=0 or args.pitch_length_m<=0 or args.pitch_width_m<=0:raise ValueError('Gate and pitch dimensions must be positive')
            from .compare import compare
            compare(args.tracks,args.atd,args.out,args.pitch_length_m,args.pitch_width_m,args.atd_offset,args.gate_m,args.alignment_verified)
            return 0
        names=('video','device','start_frame','max_frames','calibration_csv','calibration_origin','jersey','refine','replay')
        cfg=load_config(args.config,{k:getattr(args,k,None) for k in names})
        if getattr(args,'no_overlay',False):cfg['write_overlay']=False
        if args.command=='doctor':return doctor(cfg)
        if args.command=='calibrate':
            from .calibration import calibrate
            calibrate(cfg,args.out)
        else:
            if args.backend=='replay' and not cfg.get('replay'):raise ValueError('--backend replay requires --replay')
            from .pipeline import run
            run(cfg,args.out,args.backend)
        return 0
    except (ValueError,OSError,RuntimeError,ImportError,subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1
