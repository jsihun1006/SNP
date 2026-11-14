"""NumPy/SciPy association backend; replaces cython_bbox and lap extensions.

Expanded-box rule is the upstream rule. LAP is solved with explicit unmatched
assignments. Tie breaking can differ from LAPJV; this is an adapter variant.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def linear_assignment(cost_matrix, thresh):
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n,m = cost.shape
    if not n or not m:
        return np.empty((0,2),dtype=int), tuple(range(n)), tuple(range(m))
    if not np.isfinite(cost).all():
        raise ValueError('Association cost contains NaN/Inf')
    big = (n+m+1)*(max(1., abs(thresh), float(np.abs(cost).max()))+1.)
    aug = np.full((n+m,n+m),big)
    aug[:n,:m] = np.where(cost <= thresh, cost, big)
    aug[np.arange(n),m+np.arange(n)] = thresh/2+1e-10
    aug[n+np.arange(m),np.arange(m)] = thresh/2+1e-10
    aug[n:,m:] = 0
    rr,cc = linear_sum_assignment(aug)
    matches = np.array([(r,c) for r,c in zip(rr,cc) if r<n and c<m and cost[r,c]<=thresh],dtype=int).reshape(-1,2)
    return matches, tuple(sorted(set(range(n))-set(matches[:,0]))), tuple(sorted(set(range(m))-set(matches[:,1])))


def bbox_ious(a,b):
    a,b = np.asarray(a,dtype=float).reshape(-1,4), np.asarray(b,dtype=float).reshape(-1,4)
    lo = np.maximum(a[:,None,:2],b[None,:,:2])
    hi = np.minimum(a[:,None,2:],b[None,:,2:])
    inter = np.maximum(hi-lo+1,0).prod(axis=2)
    aa = np.maximum(a[:,2:]-a[:,:2]+1,0).prod(axis=1)
    bb = np.maximum(b[:,2:]-b[:,:2]+1,0).prod(axis=1)
    return inter / np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)


def expand(tlbr,e):
    # Keep the original implementation's expansion convention.
    t,l,b,r = tlbr
    w,h = r-l,b-t
    return [t-(2*h*e+h)//2,l-(2*w*e+w)//2,b+(2*h*e+h)//2,r+(2*w*e+w)//2]


def _boxes(tracks,last=False):
    return [t if isinstance(t,np.ndarray) else (t.last_tlbr if last else t.tlbr) for t in tracks]


def iou_distance(a,b):
    return 1-bbox_ious(_boxes(a),_boxes(b))


def eiou_distance(a,b,expand):
    from .matching import expand as expand_box
    return 1-bbox_ious([expand_box(t,expand) for t in _boxes(a,True)],
                      [expand_box(t,expand) for t in _boxes(b,True)])


def embedding_distance(tracks,detections,metric='cosine'):
    if not tracks or not detections:
        return np.zeros((len(tracks),len(detections)))
    return np.maximum(0.,cdist([t.smooth_feat for t in tracks],[d.curr_feat for d in detections],metric))
