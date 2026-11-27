"""Bounded CPU reproductions against the reviewed commit; never touch shared data.

These are diagnostic reproductions of undesirable baseline behaviour, not tests
claiming it is correct. A patched pipeline should change the recorded outcomes.
Run with --out work/qa-repro-<unique>. No training/inference/network.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mu

import build_version
import convert_yolo
import prelabel
import review_queue
from vslib import CLASSES
from vslib.geometry import mask_to_polygon, normalize, rasterize
from vslib.instances import resolve
from vslib.notebook_lib import load
from vslib.writer import SourceWriter


def rect(x0, y0, x1, y1, size=200):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def item(role, mask, **kwargs):
    return dict(role=role, mask=mask, label=role, **kwargs)


def run(out):
    out = out.resolve()
    # This command can invoke converters that rebuild their output: isolate it.
    repo = Path(__file__).resolve().parents[2]
    if not out.is_relative_to(repo / 'work') or out.exists():
        raise ValueError('use a NEW directory inside this worktree work/')
    out.mkdir(parents=True)
    nb = load()
    results = {}

    # Two plausible motorcycles: arbitrary argmax picks one without a flag.
    vehicle = rect(30, 90, 150, 170)
    human = rect(65, 40, 105, 145)
    raw = [item('vehicle2', vehicle.copy(), sub='motorcycle') for _ in range(2)]
    raw.append(item('rider', human, sub='motorcycle'))
    instances, flags, _ = resolve(raw, 200, 200)
    results['ambiguous_pair'] = {'classes': [CLASSES[c] for c, _ in instances], 'flags': flags}

    # A known rider is attached; the second COCO person weakly overlaps the now
    # occupied bike. The weak-person review gate only checks FREE vehicles.
    raw = [item('vehicle2', vehicle.copy(), sub='motorcycle'),
           item('rider', human, sub='motorcycle'), item('person', rect(35, 20, 60, 100))]
    instances, flags, _ = resolve(raw, 200, 200)
    results['weak_passenger'] = {'classes': [CLASSES[c] for c, _ in instances], 'flags': flags}

    # Geometry alone cannot distinguish a person walking beside a bike from a
    # seated rider with this projection; no human-review flag is added.
    raw = [item('vehicle2', vehicle.copy(), sub='bicycle'), item('person', human.copy())]
    instances, flags, _ = resolve(raw, 200, 200)
    results['person_beside_bike'] = {'classes': [CLASSES[c] for c, _ in instances], 'flags': flags,
                                   'semantic_truth': 'synthetic geometry only; not a real-image error count'}

    # Holed masks and secondary components have visible pixels that must survive.
    for name, mask in [('hole', rect(20, 20, 180, 180) & ~rect(60, 60, 140, 140)),
                       ('small_component', rect(10, 10, 110, 110) | rect(160, 160, 170, 170))]:
        writer = SourceWriter(name, out / 'processed')
        status = writer.add(np.zeros((200, 200, 3), np.uint8),
                            [item('direct', mask, target='Car')], orig=name)
        summary = writer.close()
        lp = next((out / 'processed' / name / 'labels').glob('*.txt'))
        rows, _ = nb['parse_label_file'](lp, len(CLASSES))
        decoded = np.zeros_like(mask)
        for _, points in rows:
            decoded |= mu.decode(mu.merge(mu.frPyObjects([(points * [200, 200]).reshape(-1).tolist()], 200, 200))).astype(bool)
        intersection = int((mask & decoded).sum())
        union = int((mask | decoded).sum())
        results[name] = {'status': status, 'original_pixels': int(mask.sum()),
                         'decoded_pixels': int(decoded.sum()), 'roundtrip_iou': intersection / union,
                         'lost_pixels': int((mask & ~decoded).sum()), 'added_pixels': int((~mask & decoded).sum()),
                         'drops': summary['dropped']}

    # COCO's polygon list is a union; cv2.fillPoly over all contours is even-odd.
    a = np.array([[20,20],[100,20],[100,100],[20,100]])
    b = a + 40
    combined = rasterize([a,b], 200, 200)
    union = rasterize([a],200,200) | rasterize([b],200,200)
    results['overlapping_coco_parts'] = {'lost_union_pixels': int((union & ~combined).sum())}

    # Real cmd_accept: parser accepts a self-crossing bow tie; polygon_problem
    # would reject it. No source or shared queue is involved.
    source = out / 'queue'
    for d in ['images','labels','review/images','review/labels']:
        (source / d).mkdir(parents=True)
    Image.new('RGB', (200,200)).save(source / 'review/images/bow.jpg')
    (source / 'review/labels/bow.txt').write_text('',encoding='utf-8')
    (source / 'meta.jsonl').write_text(json.dumps({'file':'bow.jpg','status':'review','w':200,'h':200})+'\n',encoding='utf-8')
    exported = out / 'cvat'
    exported.mkdir()
    (exported / 'bow.txt').write_text('0 .1 .1 .9 .9 .1 .9 .9 .1\n',encoding='utf-8')
    (exported / 'data.yaml').write_text(yaml.safe_dump({'names':CLASSES}),encoding='utf-8')
    review_queue.cmd_accept(source, SimpleNamespace(**{'from':str(exported),'reviewer':'qa-synthetic'}))
    rows, probs = nb['parse_label_file'](source / 'labels/bow.txt', len(CLASSES))
    results['accept_invalid_polygon'] = {'status':json.loads((source/'meta.jsonl').read_text())['status'],
                                        'parse_problems':probs,'polygon_problem':nb['polygon_problem'](rows[0][1],200,200)}

    # A source missing both Sign/TL: no predictions -> fill never sets review.
    source = out / 'fill'
    (source / 'images').mkdir(parents=True)
    (source / 'labels').mkdir()
    Image.new('RGB',(200,200)).save(source/'images/scene.jpg')
    (source/'labels/scene.txt').write_text('0 .1 .1 .9 .1 .9 .9 .1 .9\n',encoding='utf-8')
    (source/'meta.jsonl').write_text(json.dumps({'file':'scene.jpg','status':'ok','flags':[]})+'\n',encoding='utf-8')
    with patch.object(prelabel,'predict_masks',return_value=[]):
        prelabel.cmd_fill(SimpleNamespace(out=str(out),source='fill',classes=['Sign','Traffic light'],conf=.35,model='stub'),None)
    results['fill_zero_predictions'] = json.loads((source/'meta.jsonl').read_text())

    # YOLO converter loses an annotated object but still approves the image.
    source = out/'bad_yolo'
    (source/'images').mkdir(parents=True)
    (source/'labels').mkdir()
    Image.new('RGB',(200,200)).save(source/'images/scene.jpg')
    (source/'labels/scene.txt').write_text('0 .5 .5 .2 .2\n',encoding='utf-8')
    convert_yolo.main(['--root',str(source),'--source','bad','--out',str(out/'converted')])
    results['malformed_yolo_approved'] = json.loads((out/'converted/bad/meta.jsonl').read_text())

    # Actual build_version on tiny independent data; no stubbing the validator.
    source = out/'build_sources'/'fixture'
    (source/'images').mkdir(parents=True)
    (source/'labels').mkdir()
    meta = []
    for i in range(6):
        pixels=np.random.default_rng(55+i).integers(0,256,(100,100,3),dtype=np.uint8)
        Image.fromarray(pixels).save(source/f'images/img{i}.png')
        (source/f'labels/img{i}.txt').write_text('0 .1 .1 .9 .1 .9 .9 .1 .9\n',encoding='utf-8')
        meta.append({'file':f'img{i}.png','status':'ok','group':f'g{i}','orig':f'img{i}.png'})
    (source/'meta.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in meta),encoding='utf-8')
    frozen = out/'frozen.txt'
    frozen.write_text('fixture/img0.png\nfixture/absent.png\n',encoding='utf-8')
    config = out/'fixture.yaml'
    config.write_text(yaml.safe_dump({'name':'fixture','sources':[{'name':'fixture'}],
                                      'val':{'frozen_list':str(frozen)},'dedup':{'hamming':2}}),encoding='utf-8')
    argv = [str(config),'--processed',str(out/'build_sources'),'--out-root',str(out/'versions')]
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):
        build_version.main(argv)
    (out/'build.log').write_text(buffer.getvalue(),encoding='utf-8')
    stats=json.loads((out/'versions/fixture_report/stats.json').read_text(encoding='utf-8'))
    results['missing_frozen_key']={'build_completed':True,'requested_val':2,'actual_val':stats['images']['val'],
                                   'warning_seen':'frozen val images are not in this build' in buffer.getvalue()}
    # Demonstrate mutable hard-linked validation labels, only in this fixture.
    destination=out/'versions/fixture/labels/val/img0.txt'
    before=destination.read_text()
    (source/'labels/img0.txt').write_text('1 .1 .1 .9 .1 .9 .9 .1 .9\n',encoding='utf-8')
    results['hardlink_label_mutation']={'built_val_changed_without_rebuild':destination.read_text()!=before,
                                      'same_frozen_key':True}
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out)
