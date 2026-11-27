"""CPU diagnostics for class mapping and BTC guideline gaps. No source edits.

Semantic tags in fixtures describe the intended case; the production resolver
currently ignores those tags. Outputs are observations, not production fixes.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pycocotools import mask as mu

from vslib import CLASSES, ROOT
from vslib.geometry import normalize, mask_to_polygon
from vslib.instances import resolve
from vslib.mapping import load_source_config,role_of
from vslib.notebook_lib import check_classes,load


def rect(x0,y0,x1,y1):
    m=np.zeros((200,200),bool)
    m[y0:y1,x0:x1]=True
    return m


def run():
    check_classes(CLASSES)
    nb=load()
    report={'pipeline_commit':subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip(),
            'guideline':'User supplied BTC excerpt + follow-up: 9-15 seats = Car',
            'configs':{},'probes':{}}
    for path in sorted((ROOT/'configs/sources').glob('*.yaml')):
        cfg=load_source_config(path)
        entries=[]
        for label in cfg['mapping']:
            role=role_of(cfg,label)
            potential=[]
            if role['role']=='direct':potential=[CLASSES.index(role['target'])]
            if role['role']=='vehicle2':potential=[4,5] if role['sub']=='motorcycle' else [6,7]
            if role['role']=='person':potential=[5,7,8]
            if role['role']=='rider':
                potential=[5] if role.get('sub')=='motorcycle' else ([7] if role.get('sub')=='bicycle' else [5,7])
            entries.append({'source_label':label,'role':role,'potential_btc_ids':potential})
        report['configs'][cfg['name']]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'mapping':entries,'unmapped':cfg.get('unmapped'),
            'missing_classes':cfg.get('missing_classes',[]),'incomplete_classes':cfg.get('incomplete_classes',[])}
        for label,tags in [('truck',{'vehicle_type':'pickup'}),('bus',{'seat_count':9}),
                           ('bus',{'seat_count':12}),('bus',{'seat_count':15})]:
            if label not in cfg['mapping']:continue
            raw={**role_of(cfg,label),'label':label,'mask':rect(20,30,160,150),**tags}
            rows,flags,_=resolve([raw],200,200)
            report['probes'][f"{cfg['name']}:{label}:{tags}"]={'classes':[CLASSES[c] for c,_ in rows],'flags':flags,
                'semantic_tags':tags,'note':'Synthetic tagged source instance; no claim of dataset-wide prevalence'}
    raw=[{'role':'direct','target':'Truck','label':'truck','mask':rect(10,30,180,160),
          'vehicle_type':'tractor_and_semitrailer'}]
    rows,flags,_=resolve(raw,200,200)
    report['probes']['combined_tractor_trailer']={'classes':[CLASSES[c] for c,_ in rows],'instances':len(rows),'flags':flags}
    raw=[{'role':'direct','target':'Car','label':'car','mask':rect(10,10,120,120)},
         {'role':'direct','target':'Truck','label':'truck','mask':rect(60,60,160,160)}]
    rows,flags,_=resolve(raw,200,200)
    report['probes']['direct_instance_overlap']={'overlap_pixels':int((rows[0][1]&rows[1][1]).sum()),'flags':flags}
    # Positive control: two known riders on one motorcycle are unioned.
    vehicle=rect(30,90,150,170)
    a=rect(65,40,105,145);b=rect(35,40,60,130)
    rows,flags,_=resolve([{'role':'vehicle2','sub':'motorcycle','mask':vehicle,'label':'motorcycle'},
        {'role':'rider','sub':'motorcycle','mask':a,'label':'driver'},
        {'role':'rider','sub':'motorcycle','mask':b,'label':'passenger'}],200,200)
    report['probes']['known_passenger_union']={'classes':[CLASSES[c] for c,_ in rows],'flags':flags,
                                               'all_visible_parts_preserved':len(rows)==1 and np.array_equal(rows[0][1],vehicle|a|b)}
    # Equal-size disconnected visible components: a positive bridge control.
    original=rect(20,20,70,100)|rect(110,20,160,100)
    polygon=normalize(mask_to_polygon(original),200,200)
    restored=mu.decode(mu.merge(mu.frPyObjects([(polygon*[200,200]).reshape(-1).tolist()],200,200))).astype(bool)
    report['probes']['two_large_components_bridge']={'polygon_problem':nb['polygon_problem'](polygon,200,200),
        'retains_pixels_in_both_parts':bool(restored[30:80,30:60].any() and restored[30:80,120:150].any()),
        'iou':float((original&restored).sum()/(original|restored).sum())}
    thin=np.array([[10,10],[110,10],[110,14],[10,14]],float)/640
    sparse=np.array([[20,20],[40,20],[40,21],[21,21],[21,40],[20,40]],float)/640
    for name,points in [('thin_100x4',thin),('sparse_L_20x20',sparse)]:
        ratio=float(np.ptp(points,axis=0).min())
        area=nb['area_at_imgsz'](points,640,640,640)
        raster=mu.decode(mu.merge(mu.frPyObjects([(points*[640,640]).reshape(-1).tolist()],640,640)))
        report['probes'][name]={'short_side_ratio':ratio,'below_guideline_short_side':ratio<.0125,
            'notebook_shoelace_area_640':area,'scorer_decoder_pixel_area_640':int(raster.sum()),
            'below_scorer_area_64':int(raster.sum())<64}
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to(ROOT/'docs/qa/evidence'):
        ap.error('choose a NEW JSON output inside docs/qa/evidence/')
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['probes'],ensure_ascii=False,indent=2))
