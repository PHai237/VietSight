"""Compare stored V1 Sign masks to reconstructed native Mapillary visible masks.

Reference uses the existing converter's rasterizer and painter order; this tests
conversion fidelity, not whether Mapillary itself is correct BTC ground truth.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from vslib.geometry import rasterize,area_at_640
from vslib.mapping import load_source_config,role_of
from vslib.notebook_lib import load
from masks import encode_checked,decode_polygon


def compare(root,reference,out):
    if out.exists() or out.resolve().is_relative_to(root.resolve()) or out.resolve().is_relative_to(reference.resolve()):
        raise ValueError('new output outside inputs required')
    cfg=json.loads(next(reference.rglob('config_v2.0.json')).read_text())
    readable={r['name']:r['readable'].lower() for r in cfg['labels']}
    source_cfg=load_source_config('mapillary_vistas');nb=load();records=[];images=0;missed=[]
    out.mkdir(parents=True);(out/'crops').mkdir()
    for annotation in sorted(reference.glob('*/training/v2.0/polygons/*.json')):
        matches=list((root/'images/train').glob(f'mapillary_vistas__{annotation.stem}__*'))
        if len(matches)!=1:continue
        image_path=matches[0]
        with Image.open(image_path) as im:original=im.convert('RGB')
        w,h=original.size;ann=json.loads(annotation.read_text());scale=min(1,1024/max(ann['height'],ann['width']))
        occupied=np.zeros((h,w),bool);refs=[]
        for obj in reversed(ann['objects']):
            mask=rasterize([np.asarray(obj['polygon'],float)*scale],h,w)
            visible=mask&~occupied;occupied|=mask
            label=readable.get(obj['label'],obj['label']);role=role_of(source_cfg,label)
            if role.get('target')=='Sign' and area_at_640(visible.sum(),h,w)>=64:
                refs.append((str(obj['id']),label,visible))
        rows,_=nb['parse_label_file'](root/'labels/train'/image_path.with_suffix('.txt').name,11)
        stored=[(i+1,decode_polygon(p,h,w)) for i,(c,p) in enumerate(rows) if c==9]
        images+=1;used=set()
        for iid,label,source_mask in refs:
            candidates=[]
            for index,(line,stored_mask) in enumerate(stored):
                if index in used:continue
                inter=int((source_mask&stored_mask).sum());union=int((source_mask|stored_mask).sum())
                candidates.append((inter/max(1,union),index,line,stored_mask))
            if not candidates or max(v[0] for v in candidates)<.1:
                missed.append({'image':image_path.name,'source_instance':iid,'label':label});continue
            iou,index,line,old=max(candidates,key=lambda r:r[0]);used.add(index)
            polygon,metrics,issues=encode_checked(source_mask,nb)
            new=decode_polygon(polygon,h,w) if polygon is not None else np.zeros_like(source_mask)
            components=cv2.connectedComponents(source_mask.astype(np.uint8))[0]-1
            record={'image':image_path.name,'source_instance':iid,'source_label':label,'v1_line':line,
                    'source_visible_pixels':int(source_mask.sum()),'v1_iou':iou,'v1_added_pixels':int((old&~source_mask).sum()),
                    'v1_lost_pixels':int((source_mask&~old).sum()),'source_components':components,
                    'v2':metrics,'v2_issues':issues}
            records.append(record)
            ys,xs=np.nonzero(source_mask|old)
            x0=max(0,int(xs.min())-10);x1=min(w,int(xs.max())+11);y0=max(0,int(ys.min())-10);y1=min(h,int(ys.max())+11)
            tiles=[]
            for name,mask in [('Original',None),('Source visible',source_mask),('V1',old),('V2',new)]:
                base=np.asarray(original).copy()
                if mask is not None:base[mask]=(base[mask]*.5+np.array([0,220,255])*.5).astype(np.uint8)
                tile=ImageOps.contain(Image.fromarray(base).crop((x0,y0,x1,y1)),(220,220),Image.Resampling.NEAREST)
                panel=Image.new('RGB',(220,245),'#202020');panel.paste(tile,(0,25));ImageDraw.Draw(panel).text((5,5),name,fill='white');tiles.append(panel)
            sheet=Image.new('RGB',(880,245),'#202020')
            for k,panel in enumerate(tiles):sheet.paste(panel,(k*220,0))
            sheet.save(out/'crops'/f'{annotation.stem}_{iid}.png')
    values=[r['v1_iou'] for r in records]
    summary={'reference_images':images,'matched_signs':len(records),'unmatched_source_signs':missed,
             'v1_iou_median':float(np.median(values)) if values else None,
             'v1_below_iou_095':sum(v<.95 for v in values),'v1_below_iou_075':sum(v<.75 for v in values),
             'v2_passed_gate':sum(not r['v2_issues'] for r in records),'records':records,
             'limits':'First 12 sorted sample images with Sign. Matched by best available IoU, not native persistent ID. Reference is existing raster/painter reconstruction, not verified BTC truth. No AP attribution measured.'}
    (out/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--sample',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();compare(a.sample,a.reference,a.out)
