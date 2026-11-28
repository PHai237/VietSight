"""V2-only mask geometry. Pixel-edge rings, preserved holes/components, decoder gate."""
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mu

from vslib.geometry import _bridge


def decode_polygon(points, height, width):
    return mu.decode(mu.merge(mu.frPyObjects([(points * [width, height]).reshape(-1).tolist()], height, width))).astype(bool)


def rings(mask):
    """Trace oriented boundaries on cell edges; foreground pixels cover [x,x+1]x[y,y+1]."""
    m=np.pad(mask.astype(bool),1)
    center=m[1:-1,1:-1]
    graph=defaultdict(set)
    for active,dx0,dy0,dx1,dy1 in [
        (center & ~m[:-2,1:-1],0,0,1,0),
        (center & ~m[1:-1,2:],1,0,1,1),
        (center & ~m[2:,1:-1],1,1,0,1),
        (center & ~m[1:-1,:-2],0,1,0,0)]:
        ys,xs=np.nonzero(active)
        for x,y in zip(xs.tolist(),ys.tolist()):graph[(x+dx0,y+dy0)].add((x+dx1,y+dy1))
    directions={(1,0):0,(0,1):1,(-1,0):2,(0,-1):3}
    result=[]
    while graph:
        start=min(graph)
        current=start;last_direction=None;points=[]
        while True:
            points.append(current)
            choices=graph[current]
            if last_direction is None:nxt=min(choices)
            else:
                def priority(p):
                    direction=directions[(p[0]-current[0],p[1]-current[1])]
                    turn=(direction-last_direction)%4
                    return ({1:0,0:1,3:2,2:3}[turn],p)
                nxt=min(choices,key=priority)
            last_direction=directions[(nxt[0]-current[0],nxt[1]-current[1])]
            choices.remove(nxt)
            if not choices:del graph[current]
            current=nxt
            if current==start:break
        p=np.asarray(points,dtype=np.float64)
        a=p-np.roll(p,1,axis=0);b=np.roll(p,-1,axis=0)-p
        keep=(a[:,0]*b[:,1]-a[:,1]*b[:,0])!=0
        p=p[keep]
        if len(p)>=3:result.append(p)
    return result


def encode_checked(mask, nb, min_iou=.995, max_points=400):
    """Never largest-part fallback. Return (serialized-precision polygon, metrics, issues)."""
    h,w=mask.shape
    parts=rings(mask)
    if not parts:return None,{},['empty_mask']
    if len(parts)>80:
        return None,{'rings':len(parts),'original_pixels':int(mask.sum())},['too_many_components_for_400_point_gate']
    best=None
    for epsilon in [0,.125,.25,.375,.5,.625,.75,.875,1,1.125,1.25,1.375,1.5]:
        simplified=[]
        for p in parts:
            candidate=cv2.approxPolyDP(p.astype(np.float32),epsilon,True).reshape(-1,2).astype(float) if epsilon else p
            # A thin visible fragment must survive even when approximation
            # collapses it to a line. Simplify the other rings independently.
            simplified.append(candidate if len(candidate)>=3 else p)
        polygon=_bridge(simplified) if len(simplified)>1 else simplified[0]
        polygon=np.round(np.clip(polygon/[w,h],0,1),6)
        if len(polygon)>max_points:continue
        problem=nb['polygon_problem'](polygon,w,h)
        if problem:continue
        restored=decode_polygon(polygon,h,w)
        union=int((mask|restored).sum());intersection=int((mask&restored).sum())
        metrics={'roundtrip_iou':intersection/max(1,union),'lost_pixels':int((mask&~restored).sum()),
                 'added_pixels':int((~mask&restored).sum()),'rings':len(parts),'vertices':len(polygon),
                 'original_pixels':int(mask.sum()),'decoded_pixels':int(restored.sum())}
        if best is None or metrics['roundtrip_iou']>best[1]['roundtrip_iou']:best=(polygon,metrics)
        if metrics['roundtrip_iou']>=min_iou:return polygon,metrics,[]
    if best:return best[0],best[1],['roundtrip_iou_below_gate']
    return None,{'rings':len(parts),'original_pixels':int(mask.sum())},['unrepresentable_polygon_without_pixel_loss']
