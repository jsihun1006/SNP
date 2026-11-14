import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from . import UPSTREAM_SHA, REID_SHA


def repository_state(root, expected=UPSTREAM_SHA):
    root = Path(root)
    sha = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    if sha != expected or dirty:
        raise ValueError('Expected a clean pinned checkout: '+str(root)+' @ '+expected)
    return {'path':str(root.resolve()),'sha':sha,'clean':True}


def bootstrap(root):
    root = Path(root)
    for folder,url,sha in [
        ('SoccernetGSR','https://github.com/yinmayoo185/SoccernetGSR.git',UPSTREAM_SHA),
        ('deep-person-reid','https://github.com/KaiyangZhou/deep-person-reid.git',REID_SHA)]:
        target = root/folder
        if target.exists():
            repository_state(target,sha)
            print('Already present:',target)
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout','--filter=blob:none',url,str(target)],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',sha],check=True)
        repository_state(target,sha)


def namespace(name,path):
    if name in sys.modules:
        existing = sys.modules[name]
        if list(getattr(existing,'__path__',[])) != [str(path)]:
            raise RuntimeError('Module name is already used by another installation: '+name)
        return existing
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def load_file(name,path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name,str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name,None)
        raise
    return module


def mount(root):
    root = Path(root).resolve()
    repository_state(root)
    # Do not run upstream package __init__: it imports training/evaluation and C++ extensions.
    sys.dont_write_bytecode = True
    namespace('yolox',root/'yolox')
    namespace('yolox.EIoU_tracker',root/'yolox/EIoU_tracker')
    return root


class Tracker:
    """Original Deep_EIoU state machine with a portable association backend.

    Keeps original last-bbox association (no added Kalman prediction). Updates
    once per decoded frame, including empty frames. Detection joins use boxes,
    never the returned track's list position as an embedding index.
    """
    def __init__(self,root,fps):
        import numpy as np
        from . import matching
        self.np = np
        root = mount(root)
        sys.modules['yolox.EIoU_tracker.matching'] = matching
        mod = load_file('yolox.EIoU_tracker.Deep_EIoU',root/'yolox/EIoU_tracker/Deep_EIoU.py')
        self.cfg = dict(TRACK_HIGH_THRESH=.6,TRACK_LOW_THRESH=.05,NEW_TRACK_THRESH=.7,
                        MATCH_THRESH=.8,WITH_REID=True,TRACK_BUFFER=60,
                        PROXIMITY_THRESH=.5,APPEARANCE_THRESH=.25)
        self.mod,self.fps = mod,fps
        self.reset()

    def reset(self):
        self.impl = self.mod.Deep_EIoU(self.cfg,frame_rate=self.fps)

    def update(self,detections,features):
        np = self.np
        detections = np.asarray(detections,dtype=float).reshape(-1,5)
        features = np.asarray(features,dtype=float)
        if features.ndim != 2 or len(features)!=len(detections) or not np.isfinite(features).all():
            raise ValueError('Every detection needs one finite ReID feature')
        if len(features) and np.any(np.linalg.norm(features,axis=1)<1e-12):
            raise ValueError('Zero ReID vector')
        targets = self.impl.update(detections,features)
        used = set()
        result = []
        for track in targets:
            if track.frame_id != self.impl.frame_id:
                continue  # Predicted/stale boxes are not reported as observations.
            box = np.array(track.last_tlwh,dtype=float)
            xyxy = box.copy();xyxy[2:] += xyxy[:2]
            candidates = [i for i,d in enumerate(detections) if i not in used
                          and np.allclose(d[:4],xyxy,rtol=1e-5,atol=1e-3)
                          and np.isclose(d[4],track.score)]
            if len(candidates)!=1:
                raise ValueError('Cannot uniquely join online track to its current detection')
            index = candidates[0];used.add(index)
            result.append((track.track_id,index,box))
        return result
