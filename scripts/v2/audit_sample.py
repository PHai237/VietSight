"""Measure downloaded V1 train labels and bounded sampled images; no inference."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qa'))
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from vslib import CLASSES
from vslib.notebook_lib import load
from label_audit import read_labels


def analyze(root,out):
    if out.exists() or out.resolve().is_relative_to(root.resolve()):raise ValueError('new output outside input required')
    nb=load();counts=Counter();invalid=Counter();parse_problems=[];labels_by_image={}
    for path in sorted((root/'labels/train').glob('*.txt')):
        rows,issues=nb['parse_label_file'](path,11)
        parse_problems.extend(issues)
        for cid,points in rows:
            counts[CLASSES[cid]]+=1
            if nb['polygon_self_intersects'](points):invalid[CLASSES[cid]]+=1
        labels_by_image[path.stem]=rows
    selected=sorted((root/'images/train').glob('*'))
    records=[];signs=[];scales=Counter();allflag=Counter();perclass=Counter()
    out.mkdir(parents=True);(out/'evidence').mkdir()
    tiles=[]
    for p in selected:
        with Image.open(p) as im:original=ImageOps.exif_transpose(im).convert('RGB')
        w,h=original.size
        _,issues,instances=read_labels(root/'labels/train'/f'{p.stem}.txt',w,h,nb)
        rows=labels_by_image[p.stem]
        for item in instances:
            perclass[item['class_name']]+=1
            if item['class_id']==9:
                short=item['short_side_at_640'];scale='under8' if short<8 else ('8to16' if short<16 else ('16to32' if short<32 else '32plus'))
                scales[scale]+=1;signs.append(dict(file=p.name,**item))
        for issue in issues:allflag[issue['reason'].split(':')[0]]+=1
        record={'file':p.name,'width':w,'height':h,'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'instances':len(instances),'signs':sum(i['class_id']==9 for i in instances),'issues':issues}
        records.append(record)
        texts=[f'{cid} {CLASSES[cid]} L{k+1}' for k,(cid,_) in enumerate(rows)]
        overlay=nb['draw_polygons'](original,rows,CLASSES,texts=texts)
        # Both full overview and per-sign crops are source annotations, not GT corrections.
        original.thumbnail((480,320));overlay.thumbnail((480,320))
        panel=Image.new('RGB',(960,350),'#202020');panel.paste(original,(0,30));panel.paste(overlay,(480,30))
        ImageDraw.Draw(panel).text((5,5),p.name,fill='white')
        panel.save(out/'evidence'/f'{p.stem}.png')
        thumb=panel.copy();thumb.thumbnail((480,175));tiles.append(thumb)
    for start in range(0,len(tiles),12):
        batch=tiles[start:start+12];sheet=Image.new('RGB',(1440,700),'#202020')
        for k,tile in enumerate(batch):sheet.paste(tile,((k%3)*480,(k//3)*175))
        sheet.save(out/f'sheet_{start//12+1:02}.png')
    report={'scope':'V1 train only; all downloaded labels, 80 targeted images; no val modification',
            'train_label_files':len(labels_by_image),'train_instances_by_class':{c:counts[c] for c in CLASSES},
            'self_crossing_counts_all_train_labels':dict(invalid),'parse_problems':parse_problems,
            'sample_images':len(records),'sample_instances_by_class':dict(perclass),
            'sample_signs':len(signs),'sample_sign_short_side_at640_bins':dict(scales),'sample_flags':dict(allflag),
            'sample_sign_metrics':signs,'records':records,
            'limits':'Targeted sample, not whole-dataset error rate. No source Mapillary mask comparison yet. Visual conclusions logged separately.'}
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['train_label_files','train_instances_by_class','sample_images','sample_signs','sample_sign_short_side_at640_bins','self_crossing_counts_all_train_labels']},ensure_ascii=False))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();analyze(a.root,a.out)
