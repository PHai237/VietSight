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

    def provenance(self):
        # Synthetic fixture declarations only; never used for real imagery.
        return {'country':'VN','location_verified':True,'location_evidence':'fixture://location',
                'license_verified':True,'license_evidence':'fixture://license','group':'fixture-camera-01'}

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
        cfg=self.root/'source.yaml';cfg.write_text(yaml.safe_dump({'name':'test','url':'fixture://source',
            'license':'synthetic test only','mapping':{'sign':'Sign'}}),encoding='utf-8')
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
            **self.provenance(),'coverage_reviewed':True,'reviewer':'qa-test'}}),encoding='utf-8');args.decisions=decisions
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
            **self.provenance(),'coverage_reviewed':True,'reviewer':'qa-test'}}),encoding='utf-8');args.decisions=decisions
        result=convert(args)
        self.assertEqual(result['images_ok'],1)
        record=json.loads((args.out/'test_v2/meta.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(record['geometry'][0]['roundtrip_iou'],1)

    def test_reviewed_mask_accepted_and_inputs_unchanged(self):
        import hashlib
        args=self.fixture();before=hashlib.sha256(args.ann.read_bytes()).hexdigest()
        decisions=self.root/'decisions.json';decisions.write_text(json.dumps({'scene.jpg':{'coverage_reviewed':True,
            **self.provenance(),'reviewer':'qa-test','instances':{}}}),encoding='utf-8');args.decisions=decisions
        result=convert(args)
        self.assertEqual(result['images_ok'],1)
        self.assertEqual(before,hashlib.sha256(args.ann.read_bytes()).hexdigest())
        rows=json.loads((args.out/'test_v2/meta.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(rows['geometry'][0]['roundtrip_iou'],1)
        with self.assertRaises(ValueError):convert(args)

    def test_unknown_or_foreign_location_cannot_be_accepted(self):
        args=self.fixture();decisions=self.root/'decisions.json'
        for i,country in enumerate([None,'US','VN']):
            evidence=self.provenance();evidence['country']=country
            if country=='VN':evidence['location_evidence']=''
            decisions.write_text(json.dumps({'scene.jpg':{**evidence,'coverage_reviewed':True,'reviewer':'qa-test'}}),encoding='utf-8')
            args.decisions=decisions;args.source=f'geo_{i}_v2'
            result=convert(args)
            self.assertEqual(result['images_ok'],0)
            self.assertIn('vietnam_location_evidence_required',result['review_flags'])

    def test_build_vietnam_only_and_separate_frozen_v1(self):
        args=self.fixture()
        decisions=self.root/'decisions.json';decisions.write_text(json.dumps({'scene.jpg':{
            **self.provenance(),'coverage_reviewed':True,'reviewer':'qa-test'}}),encoding='utf-8');args.decisions=decisions
        convert(args)
        source=args.out/args.source
        record=json.loads((source/'meta.jsonl').read_text(encoding='utf-8'));records=[]
        label=(source/'labels'/Path(record['file']).with_suffix('.txt')).read_text()
        for i in range(3):
            r=dict(record,file=f'vn{i}.jpg',orig=f'camera{i}/scene.jpg',
                   provenance=dict(record['provenance'],capture_group=f'vn-camera-{i}'))
            Image.fromarray(np.random.default_rng(i+5).integers(0,256,(200,200,3),dtype=np.uint8)).save(source/'images'/r['file'])
            (source/'labels'/f'vn{i}.txt').write_text(label,encoding='utf-8');records.append(r)
        (source/'meta.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
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
        self.assertTrue(release['v1_validation_kept_separate'])
        self.assertEqual(release['selected_train_images'],1)
        self.assertEqual(release['v1_train_images_imported'],0)
        self.assertEqual(release['vietnam_validation_images'],1)
        dataset=build_args.out/'versions/fixture_v2'
        self.assertFalse((dataset/'images/train/a.png').exists())
        self.assertFalse((dataset/'images/val/v.png').exists())
        val_name=next((dataset/'images/val').iterdir()).name
        val_record=next(r for r in records if r['file']==val_name)
        other=self.root/'other_v2'
        for sub in ['images','labels']:(other/sub).mkdir(parents=True)
        # The same capture group exported under another source must stay out of train.
        import shutil
        shutil.copyfile(source/'images'/val_name,other/'images/reexport.jpg')
        (other/'labels/reexport.txt').write_text(label,encoding='utf-8')
        (other/'meta.jsonl').write_text(json.dumps(dict(val_record,file='reexport.jpg'))+'\n',encoding='utf-8')
        # Reuse freezes image+label bytes, not only a list of file names.
        build_args.out=self.root/'reuse';build_args.vn_val_list=dataset.parents[1]/'frozen_vn.txt'
        build_args.vn_val_reference=dataset;build_args.sources.append(other);build_args.max_additions=10
        self.assertEqual(build_v2(build_args)['status'],'prepared_for_review')
        self.assertFalse((build_args.out/'versions/fixture_v2/images/train/reexport.jpg').exists())
        vn_val=next((dataset/'labels/val').glob('*.txt'));before=vn_val.read_bytes()
        (source/'labels'/vn_val.name).write_text('',encoding='utf-8')
        self.assertEqual(vn_val.read_bytes(),before)
        build_args.out=self.root/'mutated'
        with self.assertRaisesRegex(ValueError,'leakage gate'):
            build_v2(build_args)
        (baseline/'labels/val/v.txt').write_text('1 .1 .1 .9 .1 .9 .9 .1 .9\n',encoding='utf-8')
        self.assertEqual(vn_val.read_bytes(),before)
        self.assertEqual(frozen.read_text(),'mapillary_vistas/v.png\n')

    def test_coverage_review_cannot_bypass_missing_license(self):
        args=self.fixture();evidence=self.provenance();evidence['license_verified']=False
        decisions=self.root/'decisions.json';decisions.write_text(json.dumps({'scene.jpg':{
            **evidence,'coverage_reviewed':True,'reviewer':'qa-test'}}),encoding='utf-8');args.decisions=decisions
        result=convert(args)
        self.assertEqual(result['images_ok'],0)
        self.assertIn('source_license_evidence_required',result['review_flags'])


if __name__=='__main__':unittest.main()
