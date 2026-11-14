from pathlib import Path
import importlib
import numpy as np
from .upstream import mount, namespace, load_file, repository_state
from . import REID_SHA


def device_of(name):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable in this PyTorch environment')
    if name != 'cpu' and not name.startswith('cuda'):
        raise ValueError('Use cpu, cuda, or cuda:N')
    return torch.device(name)


def checkpoint(path):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing official checkpoint: '+str(path))
    return torch.load(path,map_location='cpu',weights_only=False)


class VisionModels:
    def __init__(self,cfg):
        import torch
        import torchvision.transforms as T
        self.torch = torch
        self.device = device_of(cfg.get('device','auto'))
        self.cfg = cfg
        root = mount(cfg['upstream'])
        utils = namespace('yolox.utils',root/'yolox/utils')
        boxes = load_file('yolox.utils.boxes',root/'yolox/utils/boxes.py')
        utils.bboxes_iou = boxes.bboxes_iou
        models = importlib.import_module('yolox.models')
        self.detector = models.YOLOX(models.YOLOPAFPN(1.33,1.25),models.YOLOXHead(1,1.25))
        for m in self.detector.modules():
            if isinstance(m,torch.nn.BatchNorm2d):
                m.eps=1e-3;m.momentum=.03
        self.detector.load_state_dict(checkpoint(cfg['detector_weights'])['model'],strict=True)
        self.detector.to(self.device).eval()
        self.postprocess = boxes.postprocess
        osnet_file = Path(cfg['reid_source'])
        repository_state(osnet_file.parents[2],REID_SHA)
        osnet = load_file('_b2p_osnet',osnet_file)
        self.reid = osnet.osnet_x1_0(num_classes=1,pretrained=False)
        state = checkpoint(cfg['reid_weights'])
        state = state.get('state_dict',state.get('model',state))
        state = {k.removeprefix('module.'):v for k,v in state.items()}
        own = self.reid.state_dict()
        usable = {k:v for k,v in state.items() if k in own and own[k].shape==v.shape and not k.startswith('classifier.')}
        required = {k for k in own if not k.startswith('classifier.') and not k.endswith('num_batches_tracked')}
        if required-set(usable):
            raise ValueError('OSNet checkpoint is incompatible; missing layers: '+str(sorted(required-set(usable))[:10]))
        self.reid.load_state_dict(usable,strict=False)
        self.reid.to(self.device).eval()
        self.transform = T.Compose([T.Resize((256,128)),T.ToTensor(),
                                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.jersey = ClipAttributes(root,cfg['clip_weights'],self.device) if cfg.get('jersey')=='clip' else None

    def infer(self,frame):
        import cv2
        from PIL import Image
        torch = self.torch
        h,w = frame.shape[:2]
        ratio = min(800/h,1440/w)
        resized = cv2.resize(frame,(int(w*ratio),int(h*ratio))).astype(np.float32)
        padded = np.full((800,1440,3),114.,dtype=np.float32)
        padded[:len(resized),:resized.shape[1]] = resized
        rgb = padded[:,:,::-1]/255.
        rgb = (rgb-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1),dtype=np.float32))[None].to(self.device)
        with torch.inference_mode():
            pred = self.postprocess(self.detector(tensor),1,self.cfg.get('detection_threshold',.1),self.cfg.get('nms_threshold',.7))[0]
        if pred is None:
            return np.empty((0,5)),np.empty((0,512)),[]
        raw = pred.cpu().numpy()
        # Only coordinate columns are resized; probabilities remain unchanged.
        raw[:,:4] /= ratio
        raw[:,[0,2]] = np.clip(raw[:,[0,2]],0,w)
        raw[:,[1,3]] = np.clip(raw[:,[1,3]],0,h)
        raw = raw[(raw[:,2]-raw[:,0]>1)&(raw[:,3]-raw[:,1]>1)]
        detections = np.c_[raw[:,:4],raw[:,4]*raw[:,5]]
        crops = [Image.fromarray(cv2.cvtColor(frame[int(y1):int(y2),int(x1):int(x2)],cv2.COLOR_BGR2RGB))
                 for x1,y1,x2,y2,_ in detections]
        features = []
        with torch.inference_mode():
            for start in range(0,len(crops),32):
                x = torch.stack([self.transform(im) for im in crops[start:start+32]]).to(self.device)
                features.append(torch.nn.functional.normalize(self.reid(x),p=2,dim=1).cpu().numpy())
        feats = np.concatenate(features) if features else np.empty((0,512))
        attrs = self.jersey.predict(crops) if self.jersey else [dict(role='person',jersey=None,color=None)]*len(crops)
        return detections,feats,attrs


class ClipAttributes:
    def __init__(self,root,weights,device):
        import clip
        import torch
        from clip.clip import _transform
        state = checkpoint(weights)
        backbone = {k.removeprefix('clip_model.'):v for k,v in state.items() if k.startswith('clip_model.')}
        if not backbone:
            raise ValueError('CLIP checkpoint must include clip_model.* backbone weights')
        model = clip.model.build_model(backbone).float().to(device)
        original = clip.load
        # Original constructor normally downloads a redundant CLIP backbone.
        clip.load = lambda *a,**k:(model,_transform(model.visual.input_resolution))
        try:
            module = load_file('_b2p_clip_heads',root/'jersey_model/CLIPFinetune.py')
            self.model = module.CLIPFinetune().float()
        finally:
            clip.load = original
        self.model.load_state_dict(state,strict=True)
        self.model.to(device).eval()
        self.device = device
        self.colors = ['red','blue','green','yellow','white','black','orange','purple','pink','brown','grey','charcoal','neon yellow']
        with torch.inference_mode():
            self.colors_encoded = torch.nn.functional.normalize(self.model.clip_model.encode_text(clip.tokenize(self.colors).to(device)).float(),dim=1)

    def predict(self,images):
        import torch
        result = []
        roles = ['Player','Goalkeeper','Referee','Ball','Other']
        with torch.inference_mode():
            for start in range(0,len(images),16):
                batch = torch.stack([self.model.preprocess(im) for im in images[start:start+16]]).to(self.device)
                out = self.model(batch)
                rp = out['role_logits'].softmax(1)
                d1,d2 = out['digit1_logits'].argmax(1),out['digit2_logits'].argmax(1)
                sims = torch.nn.functional.normalize(out['color_embedding'],dim=1)@self.colors_encoded.T
                for i in range(len(batch)):
                    a,b = int(d1[i]),int(d2[i])
                    jersey = None if b==10 else (b if a==10 else 10*a+b)
                    result.append(dict(role=roles[int(rp[i].argmax())],jersey=jersey,
                        color=self.colors[int(sims[i].argmax())],role_score=float(rp[i].max()),
                        color_similarity=float(sims[i].max())))
        return result
