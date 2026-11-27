"""CPU regression tests for opt-in V2; no notebook/validator/model changes."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image

from masks import encode_checked,decode_polygon
from policy import resolve_v2
from convert import convert
from build import build as build_v2
from vslib.notebook_lib import load


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.nb=load()

    def tearDown(self):self.tmp.cleanup()

    def test_holes_small_fragments_and_diagonal_components_roundtrip(self):
        masks=[]
        m=np.zeros((200,200),bool);m[20:180,20:180]=1;m[60:140,60:140]=0;masks.append(m)
        m=np.zeros((200,200),bool);m[10:110,10:110]=1;m[160:170,160:170]=1;masks.append(m)
        m=np.zeros((30,30),bool);m[3:10,3:10]=1;m[10:20,10:20]=1;masks.append(m)
        for mask in masks:
            polygon,metrics,issues=encode_checked(mask,self.nb)
            self.assertFalse(issues,metrics)
            self.assertIsNone(self.nb['polygon_problem'](polygon,*mask.shape[::-1]))
            self.assertTrue(np.array_equal(mask,decode_polygon(polygon,*mask.shape)))

    def test_ambiguous_person_and_overlap_require_review(self):
        a=np.zeros((200,200),bool);a[60:180,20:160]=1
        b=np.zeros_like(a);b[20:150,40:100]=1
        out,flags=resolve_v2([{'id':'v','role':'vehicle2','sub':'motorcycle','mask':a},
                             {'id':'p','role':'person','mask':b}],200,200)
        self.assertTrue(any('ambiguous_association' in v for v in flags))
        self.assertTrue(any('overlapping_instances' in v for v in flags))

    def test_reviewed_walking_person_and_seat_override(self):
        m=np.ones((50,50),bool)
        rows,flags=resolve_v2([{'id':'bus12','role':'direct','target':'Bus','mask':m,
                               'seats':12,'semantic_reviewed':True}],50,50)
        self.assertEqual(rows[0][0],0);self.assertFalse(flags)

    def fixture(self):
        import yaml
        images=self.root/'images';images.mkdir()
        Image.new('RGB',(200,200),'white').save(images/'scene.jpg')
        ann=self.root/'ann.json'
        ann.write_text(json.dumps({'images':[{'id':1,'file_name':'scene.jpg','height':200,'width':200}],
            'categories':[{'id':1,'name':'sign'}],
            'annotations':[{'id':10,'image_id':1,'category_id':1,'segmentation':[[20,20,100,20,100,100,20,100]]}]}),encoding='utf-8')
        cfg=self.root/'source.yaml';cfg.write_text(yaml.safe_dump({'name':'test','mapping':{'sign':'Sign'}}),encoding='utf-8')
        return argparse.Namespace(config=cfg,source='test_v2',out=self.root/'output',format='coco',
                                   ann=ann,images=images,root=None,decisions=None,limit=0)

    def test_unreviewed_coverage_cannot_enter_training(self):
        args=self.fixture();result=convert(args)
        self.assertEqual(result['images_ok'],0);self.assertEqual(result['images_review'],1)

    def test_coverage_review_cannot_bypass_overlap(self):
        args=self.fixture();data=json.loads(args.ann.read_text())
        data['annotations'].append(dict(data['annotations'][0],id=11))
        args.ann.write_text(json.dumps(data),encoding='utf-8')
        decisions=self.root/'decisions.json';decisions.write_text(json.dumps({'scene.jpg':{
            'coverage_reviewed':True,'reviewer':'qa-test'}}),encoding='utf-8');args.decisions=decisions
        result=convert(args)
        self.assertEqual(result['images_ok'],0)
        self.assertTrue(any('overlap' in flag for flag in result['review_flags']))

    def test_native_mapillary_annotation_matches_preshrunk_image(self):
        args=self.fixture();args.format='mapillary';args.root=self.root/'native';args.root.mkdir()
        args.ann=None
        (args.root/'scene.json').write_text(json.dumps({'width':400,'height':400,'objects':[
            {'id':10,'label':'traffic-sign-front','polygon':[[40,40],[200,40],[200,200],[40,200]]}]}),encoding='utf-8')
        args.mapillary_config=self.root/'native_config.json'
        args.mapillary_config.write_text(json.dumps({'labels':[{'name':'traffic-sign-front','readable':'Sign'}]}),encoding='utf-8')
        decisions=self.root/'decisions.json';decisions.write_text(json.dumps({'training/scene.jpg':{
            'coverage_reviewed':True,'reviewer':'qa-test'}}),encoding='utf-8');args.decisions=decisions
        result=convert(args)
        self.assertEqual(result['images_ok'],1)
        record=json.loads((args.out/'test_v2/meta.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(record['geometry'][0]['roundtrip_iou'],1)

    def test_reviewed_mask_accepted_and_inputs_unchanged(self):
        import hashlib
        args=self.fixture();before=hashlib.sha256(args.ann.read_bytes()).hexdigest()
        decisions=self.root/'decisions.json';decisions.write_text(json.dumps({'scene.jpg':{'coverage_reviewed':True,
            'reviewer':'qa-test','instances':{}}}),encoding='utf-8');args.decisions=decisions
        result=convert(args)
        self.assertEqual(result['images_ok'],1)
        self.assertEqual(before,hashlib.sha256(args.ann.read_bytes()).hexdigest())
        rows=json.loads((args.out/'test_v2/meta.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(rows['geometry'][0]['roundtrip_iou'],1)
        with self.assertRaises(ValueError):convert(args)

    def test_build_preserves_frozen_v1_and_does_not_link_mutable_sources(self):
        args=self.fixture()
        decisions=self.root/'decisions.json';decisions.write_text(json.dumps({'scene.jpg':{
            'coverage_reviewed':True,'reviewer':'qa-test'}}),encoding='utf-8');args.decisions=decisions
        convert(args)
        baseline=self.root/'baseline'
        for split,names in [('train',['a.png','b.png']),('val',['v.png'])]:
            (baseline/'images'/split).mkdir(parents=True);(baseline/'labels'/split).mkdir(parents=True)
            for seed,name in enumerate(names,start=100 if split=='val' else 55):
                Image.fromarray(np.random.default_rng(seed).integers(0,256,(100,100,3),dtype=np.uint8)).save(baseline/'images'/split/name)
                (baseline/'labels'/split/Path(name).with_suffix('.txt')).write_text('0 .1 .1 .9 .1 .9 .9 .1 .9\n',encoding='utf-8')
        frozen=self.root/'frozen.txt';frozen.write_text('mapillary_vistas/v.png\n',encoding='utf-8')
        build_args=argparse.Namespace(out=self.root/'build',baseline=baseline,frozen=frozen,
            sources=[args.out/args.source],name='fixture_v2',max_additions=1)
        release=build_v2(build_args)
        self.assertTrue(release['frozen_val_matches_v1'])
        self.assertEqual(release['selected_addition_images'],1)
        before=(build_args.out/'versions/fixture_v2/labels/val/v.txt').read_bytes()
        (baseline/'labels/val/v.txt').write_text('1 .1 .1 .9 .1 .9 .9 .1 .9\n',encoding='utf-8')
        self.assertEqual((build_args.out/'versions/fixture_v2/labels/val/v.txt').read_bytes(),before)
        self.assertEqual(frozen.read_text(),'mapillary_vistas/v.png\n')


if __name__=='__main__':unittest.main()
