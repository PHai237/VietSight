"""V2-only strict conversion of existing COCO/Supervisely masks; no model/download.

Use a new *_v2 source name and new output folder. Coverage/ontology decisions are
explicit reviewer records keyed by original file name and source instance ID.
V1 scripts, configs, dataset, frozen val and notebook are never changed.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from pycocotools import mask as mu

from convert_supervisely import object_mask
from vslib import CLASSES
from vslib.geometry import yolo_line
from vslib.mapping import load_source_config,role_of
from vslib.notebook_lib import load
from vslib.writer import SourceWriter,read_image,safe_name
from masks import encode_checked
from policy import resolve_v2


class V2Writer(SourceWriter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.nb=load()

    def add_v2(self,image,raw,orig,group,coverage_reviewed=False,reviewer=None,initial_flags=()):
        self.n_in+=1;h,w=image.shape[:2]
        instances,flags=resolve_v2(raw,h,w)
        flags=list(initial_flags)+flags
        coverage_reviewed=coverage_reviewed is True
        reviewer=reviewer.strip() if isinstance(reviewer,str) else None
        if not coverage_reviewed or not reviewer:flags.append('coverage_review_required_all_11_classes')
        lines=[];counts=Counter();geometry=[];decoded=[]
        for cid,mask,iid in instances:
            poly,metrics,issues=encode_checked(mask,self.nb)
            geometry.append(dict(instance_id=iid,class_id=cid,class_name=CLASSES[cid],**metrics,issues=issues))
            flags.extend(f'{issue}:{iid}' for issue in issues)
            if poly is None:continue
            short=float(np.ptp(poly*[w,h],axis=0).min()/max(h,w))
            geometry[-1]['short_side_ratio']=short
            if short<.0125:flags.append(f'short_side_guideline_review:{iid}')
            # Keep scorer filtering separate; never silently drop visible data.
            if metrics['decoded_pixels']*(640/max(h,w))**2<64:flags.append(f'scorer_area_under_64:{iid}')
            lines.append(yolo_line(cid,poly));counts[CLASSES[cid]]+=1
            from masks import decode_polygon
            decoded.append((iid,decode_polygon(poly,h,w)))
        for i,(aid,a) in enumerate(decoded):
            for bid,b in decoded[i+1:]:
                if (a&b).any():flags.append(f'decoded_overlap:{aid}:{bid}')
        flags=sorted(set(flags));status='review' if flags else 'ok'
        sub='review/' if flags else '';name=safe_name(self.source,orig)
        if not cv2.imwrite(str(self.out/f'{sub}images'/f'{name}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,self.q]):
            raise OSError('failed to write image')
        (self.out/f'{sub}labels'/f'{name}.txt').write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        if flags:self.n_review+=1;self.review_counts.update(counts);self.flags.update(flags)
        else:self.n_ok+=1;self.counts.update(counts)
        record={'file':name+'.jpg','orig':orig,'group':group,'status':status,'flags':flags,'w':w,'h':h,
                'instances':dict(counts),'geometry':geometry,'v2_policy':1,'reviewer':reviewer,
                'coverage_reviewed':bool(coverage_reviewed and reviewer)}
        self.meta.write(json.dumps(record,ensure_ascii=False)+'\n')
        return record


def checked_child(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError(f'input path escapes image root:{name}')
    return path


def coco_records(annotation,images):
    data=json.loads(annotation.read_text(encoding='utf-8'))
    cats={c['id']:c['name'] for c in data['categories']};by={}
    for a in data['annotations']:by.setdefault(a['image_id'],[]).append(a)
    for im in sorted(data['images'],key=lambda x:x['file_name']):
        yield im['file_name'],checked_child(images,im['file_name']),im.get('height'),im.get('width'),[
            dict(a,label=cats[a['category_id']]) for a in by.get(im['id'],[])]


def sly_records(root):
    for annotation in sorted(root.glob('*/ann/*.json')):
        data=json.loads(annotation.read_text(encoding='utf-8'))
        name=annotation.name[:-5];split=annotation.parent.parent.name
        yield f'{split}/{name}',annotation.parent.parent/'img'/name,data['size']['height'],data['size']['width'],[
            dict(obj,label=obj['classTitle'],id=obj.get('id',f'object:{i}')) for i,obj in enumerate(data.get('objects',[]))]


def mapillary_records(root,images,config):
    names={r['name']:r['readable'].lower() for r in json.loads(config.read_text(encoding='utf-8'))['labels']}
    for annotation in sorted(root.glob('*.json')):
        data=json.loads(annotation.read_text(encoding='utf-8'));stem=annotation.stem
        matches=list(images.glob(stem+'.jpg')) or list(images.glob('mapillary_vistas__'+stem+'__*.jpg'))
        if not matches:continue
        if len(matches)!=1:raise ValueError('ambiguous Mapillary image ID')
        objects=[{'id':o['id'],'label':names.get(o['label'],o['label']),'geometryType':'polygon',
                  'points':{'exterior':o['polygon'],'interior':[]}} for o in data['objects']]
        yield 'training/'+stem+'.jpg',matches[0],data['height'],data['width'],objects


def convert(args):
    cfg=load_source_config(args.config)
    if not args.source.endswith('_v2'):raise ValueError('source name must end in _v2')
    out=(args.out/args.source).resolve()
    protected=[p.resolve() for p in [args.images,args.root,args.ann,args.decisions] if p]
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        raise ValueError('output must be NEW and outside all inputs')
    decisions=json.loads(args.decisions.read_text(encoding='utf-8')) if args.decisions else {}
    if args.format=='coco':items=coco_records(args.ann,args.images)
    elif args.format=='mapillary':items=mapillary_records(args.root,args.images,args.mapillary_config)
    else:items=sly_records(args.root)
    writer=V2Writer(args.source,args.out,max_side=1024,clean=False)
    unmapped=Counter()
    try:
        for index,(orig,path,h,w,objects) in enumerate(items):
            if args.limit and index>=args.limit:break
            image=read_image(path)
            stored_h,stored_w=image.shape[:2]
            if abs(stored_w/stored_h-w/h)>.01:raise ValueError(f'image/annotation aspect mismatch:{orig}')
            scale=stored_w/w
            decision=decisions.get(orig,{})
            reviewer=decision.get('reviewer')
            raw=[];flags=[];unmapped_before=sum(unmapped.values())
            # Decode at original size. No JPEG resize before quality measurement.
            for obj in objects:
                label=obj['label'];role=role_of(cfg,label,unmapped);iid=str(obj['id'])
                if args.format=='coco':
                    seg=obj.get('segmentation')
                    if not seg:
                        if role['role']!='ignore':flags.append(f'missing_instance_mask:{iid}')
                        continue
                    if isinstance(seg,list):rles=mu.frPyObjects(seg,h,w);mask=mu.decode(mu.merge(rles)).astype(bool)
                    else:
                        rle=mu.frPyObjects(seg,h,w) if isinstance(seg.get('counts'),list) else seg
                        mask=mu.decode(rle).astype(bool)
                    if mask.shape!=(stored_h,stored_w):mask=cv2.resize(mask.astype(np.uint8),(stored_w,stored_h),interpolation=cv2.INTER_NEAREST).astype(bool)
                    if obj.get('iscrowd') and role['role']!='ignore':flags.append(f'crowd_requires_instances:{iid}')
                else:
                    mask=object_mask(obj,h,w,scale,stored_h,stored_w)
                    if mask is None:
                        if role['role']!='ignore':flags.append(f'unsupported_geometry:{iid}')
                        continue
                override=decision.get('instances',{}).get(iid,{})
                if override and not reviewer:raise ValueError('instance decisions require named reviewer')
                if override.get('target'):
                    if override['target']=='ignore':role={'role':'ignore'}
                    elif override['target'] in CLASSES:role={'role':'direct','target':override['target']}
                    else:raise ValueError(f'invalid target:{override["target"]}')
                raw.append(dict(role,mask=mask,id=iid,label=label,**{k:v for k,v in override.items() if k!='target'}))
            if args.format in ('supervisely','mapillary') and cfg.get('z_order')=='painter':
                occupied=np.zeros((stored_h,stored_w),bool)
                for item in reversed(raw):
                    original=item['mask'];item['mask']=original&~occupied;occupied|=original
            elif args.format=='supervisely' and cfg.get('z_order')=='bottom':
                flags.append('bottom_z_order_requires_visual_review')
            if sum(unmapped.values())>unmapped_before and not (decision.get('coverage_reviewed') is True and reviewer):
                flags.append('unmapped_source_labels_require_review')
            writer.add_v2(image,raw,orig,decision.get('group') or orig,
                          decision.get('coverage_reviewed',False),reviewer,flags)
        return writer.close({'script':'v2/convert.py','config':str(args.config),'unmapped_labels':dict(unmapped),
                             'note':'V2 only; masks measured at source resolution; no V1 files rewritten'})
    finally:
        if not writer.meta.closed:writer.meta.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--format',choices=['coco','supervisely','mapillary'],required=True)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--source',required=True)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--ann',type=Path)
    ap.add_argument('--images',type=Path)
    ap.add_argument('--root',type=Path)
    ap.add_argument('--mapillary-config',type=Path)
    ap.add_argument('--decisions',type=Path)
    ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args()
    if (a.format=='coco' and not(a.ann and a.images)) or (a.format=='supervisely' and not a.root) or (a.format=='mapillary' and not(a.root and a.images and a.mapillary_config)) or a.limit<0:
        ap.error('supply --ann/--images for COCO or --root for Supervisely; nonnegative limit')
    print(json.dumps(convert(a),ensure_ascii=False,indent=2))
