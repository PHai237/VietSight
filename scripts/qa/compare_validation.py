"""Read-only val identity and train/val exact-leak gate for built YOLO versions.

Compare image AND label bytes with a preserved reference directory. File names
alone cannot freeze GT. Optional near-image check reuses notebook dhash helpers;
same-video leakage still needs provenance/group inspection. No model is loaded.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vslib import IMG_EXTS
from vslib.notebook_lib import load


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def scan(root, split):
    image_root=root/'images'/split
    if not image_root.is_dir():
        raise ValueError(f'missing split: {image_root}')
    entries={}
    for p in sorted(image_root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        rel=p.relative_to(image_root)
        label=root/'labels'/split/rel.with_suffix('.txt')
        entries[rel.as_posix()]={'image_sha256':sha(p),'label_sha256':sha(label) if label.is_file() else None}
    return entries


def compare(root, reference=None, near_hamming=None):
    val,train=scan(root,'val'),scan(root,'train')
    result={'root':str(root.resolve()),'val':val,'train_images':len(train),'problems':[]}
    if not val:
        result['problems'].append({'kind':'empty_val'})
    if not train:
        result['problems'].append({'kind':'empty_train'})
    for split,records in [('val',val),('train',train)]:
        for key,item in records.items():
            if item['label_sha256'] is None:
                result['problems'].append({'kind':'missing_label','split':split,'file':key})
    seen=defaultdict(list)
    for name,item in val.items():
        seen[item['image_sha256']].append(name)
    for name,item in train.items():
        for other in seen[item['image_sha256']]:
            result['problems'].append({'kind':'exact_train_val_leak','train':name,'val':other})
    if reference:
        ref=scan(reference,'val')
        result['reference']=str(reference.resolve())
        for name in sorted(set(val)|set(ref)):
            if val.get(name)!=ref.get(name):
                result['problems'].append({'kind':'val_identity_changed','file':name,
                                           'current':val.get(name),'reference':ref.get(name)})
    if near_hamming is not None:
        nb=load()
        names=[('val',n) for n in val]+[('train',n) for n in train]
        hashes=[nb['dhash64'](root/'images'/s/n) for s,n in names]
        for i,j,distance in nb['find_duplicate_pairs'](hashes,near_hamming):
            if names[i][0]!=names[j][0]:
                result['problems'].append({'kind':'near_train_val_candidate','a':names[i],'b':names[j],
                                           'distance':distance,'note':'human confirmation required; dhash is not proof'})
    result['ok']=not result['problems']
    result['limits']='Exact hashes do not detect video/sequence leakage. Use canonical cross-source groups. No semantic label approval.'
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--reference',type=Path)
    ap.add_argument('--near-hamming',type=int,choices=range(0,65))
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    for p in [a.root,a.reference]:
        if p and (a.out.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(a.out.resolve())):
            ap.error('output must be outside all input directories')
    if a.out.exists():
        ap.error('choose a new output; evidence is never overwritten')
    report=compare(a.root,a.reference,a.near_hamming)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'ok':report['ok'],'val_images':len(report['val']),'problems':len(report['problems'])}))
    sys.exit(0 if report['ok'] else 1)
