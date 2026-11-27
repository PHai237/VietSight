"""Small CPU checks for the independent QA tools; no training or shared writes."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
from label_audit import digest, read_labels, run, safe_output
from compare_validation import compare
from vslib import CLASSES
from vslib.notebook_lib import load


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.nb=load()

    def tearDown(self):
        self.tmp.cleanup()

    def test_invalid_geometry_and_bbox_are_review_candidates(self):
        p=self.root/'labels.txt'
        p.write_text('0 .1 .1 .9 .9 .1 .9 .9 .1\n1 .5 .5 .2 .2\n',encoding='utf-8')
        rows,issues,instances=read_labels(p,100,100,self.nb)
        self.assertEqual(len(rows),1)
        self.assertTrue(any('invalid_polygon' in x['reason'] for x in issues))
        self.assertTrue(any(x['line']==2 for x in issues))
        self.assertEqual(instances[0]['line'],1)

    def test_nan_id_is_flagged_without_crash(self):
        p=self.root/'labels.txt'
        p.write_text('nan .1 .1 .9 .1 .9 .9 .1 .9\n',encoding='utf-8')
        self.assertIn('official_parser_error',read_labels(p,100,100,self.nb)[1][0]['reason'])

    def test_overlap_and_short_side_candidates(self):
        p=self.root/'labels.txt'
        p.write_text('0 .1 .1 .9 .1 .9 .9 .1 .9\n1 .2 .2 .8 .2 .8 .8 .2 .8\n9 .2 .2 .201 .2 .201 .9 .2 .9\n',encoding='utf-8')
        _,issues,_=read_labels(p,1000,1000,self.nb)
        self.assertTrue(any('overlap_with_L1' in x['reason'] for x in issues))
        self.assertTrue(any('short_side_under' in x['reason'] for x in issues))

    def test_short_side_and_area_thresholds_are_independent(self):
        from vslib.geometry import yolo_line
        import numpy as np
        p=self.root/'labels.txt'
        thin=np.array([[10,10],[110,10],[110,14],[10,14]],float)/640
        sparse=np.array([[20,20],[40,20],[40,21],[21,21],[21,40],[20,40]],float)/640
        p.write_text(yolo_line(9,thin)+'\n'+yolo_line(0,sparse)+'\n',encoding='utf-8')
        _,issues,instances=read_labels(p,640,640,self.nb)
        one=[v['reason'] for v in issues if v['line']==1]
        two=[v['reason'] for v in issues if v['line']==2]
        self.assertTrue(any('short_side_under' in v for v in one))
        self.assertFalse(any('tiny_at_640' in v for v in one))
        self.assertTrue(any('tiny_at_640' in v for v in two))
        self.assertFalse(any('short_side_under' in v for v in two))
        self.assertGreater(instances[0]['area_at_640'],64)
        self.assertLess(instances[1]['area_at_640'],64)

    def test_output_cannot_modify_inputs_or_ancestors(self):
        source=self.root/'source'
        source.mkdir()
        for out in [source,source/'reports',self.root]:
            with self.assertRaises(ValueError):
                safe_output(out,[source])

    def dataset(self):
        source=self.root/'dữ liệu'
        (source/'images').mkdir(parents=True)
        (source/'labels').mkdir()
        Image.new('RGB',(100,100),'red').save(source/'images/ảnh.png')
        (source/'labels/ảnh.txt').write_text('0 .1 .1 .8 .1 .8 .8 .1 .8\n',encoding='utf-8')
        return source

    def arguments(self,source,out):
        return argparse.Namespace(root=source,out=out,split=None,labels=None,after_labels=None,
                                  names_from=None,seed=55,per_class=1,limit=0,max_previews=3,side=100)

    def test_evidence_sampling_and_readonly_input(self):
        source=self.dataset()
        before={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
        corrected=self.root/'corrected';corrected.mkdir()
        (corrected/'ảnh.txt').write_text('1 .1 .1 .8 .1 .8 .8 .1 .8\n',encoding='utf-8')
        a=self.arguments(source,self.root/'out1');a.after_labels=corrected
        with contextlib.redirect_stdout(io.StringIO()):
            one=run(a)
            two=run(self.arguments(source,self.root/'out2'))
        self.assertEqual(one['samples_by_class'],two['samples_by_class'])
        self.assertEqual(one['instances_by_class']['Car'],1)
        png=a.out/one['records'][0]['evidence']
        with Image.open(png) as im:self.assertEqual(im.width,300)
        self.assertTrue(one['records'][0]['after_label_sha256'])
        after={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertTrue((a.out/'review.csv').is_file())

    def test_wrong_class_order_refused(self):
        source=self.dataset()
        import yaml
        (source/'data.yaml').write_text(yaml.safe_dump({'names':list(reversed(CLASSES))}),encoding='utf-8')
        with self.assertRaises(ValueError):run(self.arguments(source,self.root/'out'))
        self.assertFalse((self.root/'out').exists())

    def test_frozen_labels_and_cross_split_duplicates(self):
        import shutil
        root=self.root/'built'
        for split in ['train','val']:
            (root/'images'/split).mkdir(parents=True)
            (root/'labels'/split).mkdir(parents=True)
            Image.new('RGB',(100,100),'red').save(root/'images'/split/'a.png')
            (root/'labels'/split/'a.txt').write_text('0 .1 .1 .9 .1 .9 .9\n',encoding='utf-8')
        ref=self.root/'reference';shutil.copytree(root,ref)
        (root/'labels/val/a.txt').write_text('1 .1 .1 .9 .1 .9 .9\n',encoding='utf-8')
        result=compare(root,ref)
        self.assertFalse(result['ok'])
        self.assertEqual({p['kind'] for p in result['problems']},{'val_identity_changed','exact_train_val_leak'})


if __name__=='__main__':
    unittest.main()
