"""Bounded read-only Kaggle output sample; labels + at most 100 train images.

No notebook executions, GPU, dataset writes, or bulk Mapillary download. Signed
URLs/credentials stay in memory and are never stored in the public manifest.
"""
import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import requests


PREFIX='versions/vietsight_v1/'
PRIORITY={4:'Motorcycle',3:'Other vehicle',9:'Sign',6:'Bicycle'}


def retrieve(url,path,max_bytes,resume=False):
    if path.exists():
        if not resume:raise ValueError('refusing to overwrite sample')
        payload=path.read_bytes()
        if len(payload)>max_bytes:raise ValueError('cached file exceeds limit')
        return {'file':path.name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        for attempt in range(4):
            try:
                with requests.get(url,stream=True,timeout=90) as response:
                    response.raise_for_status()
                    payload=bytearray()
                    for chunk in response.iter_content(65536):
                        payload.extend(chunk)
                        if len(payload)>max_bytes:raise ValueError('per-file byte limit exceeded')
                break
            except requests.RequestException:
                if attempt==3:raise
                time.sleep(1+attempt*2)
        path.write_bytes(payload)
        return {'file':path.name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    except Exception as exc:
        # Never print signed URL from a requests exception.
        raise RuntimeError(f'fetch failed for {path.name}: {type(exc).__name__}') from None


def fetch(out,n=80,resume=False):
    from kaggle.api.kaggle_api_extended import KaggleApi
    if not 60<=n<=100 or (out.exists() and not resume):raise ValueError('new output directory (or --resume); 60..100 images required')
    repo=Path(__file__).resolve().parents[2]
    if not out.resolve().is_relative_to(repo/'data/raw'):
        raise ValueError('store dataset only in this worktree data/raw/ (gitignored)')
    api=KaggleApi();api.authenticate()
    # Use the same SDK request as the installed CLI's kernels_output, but fetch
    # only allowed files and do not write its notebook log or signed URLs.
    Request=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest']
    urls={};token=None
    with api.build_kaggle_client() as client:
        for page in range(100):
            req=Request();req.user_name='mangashounen';req.kernel_slug='vietsight-prep-vietsight-v1';req.page_size=200
            if token:req.page_token=token
            response=client.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in response.files:
                if item.file_name.startswith(PREFIX):urls[item.file_name[len(PREFIX):]]=item.url
            token=response.next_page_token
            if page%5==0:print(f'listing page {page+1}, version files {len(urls)}',flush=True)
            if not token:break
        else:raise RuntimeError('pagination limit exceeded')
    labels=sorted(name for name in urls if name.startswith('labels/train/') and name.endswith('.txt'))
    if not 1<=len(labels)<=2000:raise ValueError(f'unexpected label file count:{len(labels)}')
    out.mkdir(parents=True,exist_ok=resume)
    index=[]
    def one(name):return name,retrieve(urls[name],out/name,1_000_000,resume)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i,(name,record) in enumerate(executor.map(one,labels),1):
            index.append(dict(record,path=name))
            if i%100==0:print(f'labels {i}/{len(labels)}',flush=True)
    pool={cid:[] for cid in PRIORITY}
    for name in labels:
        classes={int(float(line.split()[0])) for line in (out/name).read_text(encoding='utf-8').splitlines() if line.strip()}
        for cid in pool:
            if cid in classes:pool[cid].append(Path(name).stem)
    selected=[];reasons={};rng=random.Random(55)
    # Round-robin targets, no broad all-image expansion. Include a mix of sign
    # area scales within the Sign bucket by deterministic sampling.
    for cid in pool:rng.shuffle(pool[cid])
    while len(selected)<n and any(pool.values()):
        for cid in pool:
            while pool[cid] and pool[cid][-1] in reasons:pool[cid].pop()
            if pool[cid] and len(selected)<n:
                stem=pool[cid].pop();selected.append(stem);reasons[stem]=PRIORITY[cid]
    if len(selected)<n:raise ValueError('not enough images in target classes')
    image_paths=[]
    for stem in selected:
        matches=[name for name in urls if name.startswith('images/train/') and Path(name).stem==stem]
        if len(matches)!=1:raise ValueError(f'missing/ambiguous image:{stem}')
        image_paths.append(matches[0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(retrieve,urls[name],out/name,8_000_000,resume) for name in image_paths]
        for i,(name,future) in enumerate(zip(image_paths,futures),1):
            record=future.result();index.append(dict(record,path=name,selected_for=reasons[Path(name).stem]))
            if i%10==0:print(f'images {i}/{n}',flush=True)
    manifest={'source':'mangashounen/vietsight-prep-vietsight-v1','output_prefix':PREFIX,
              'labels_train':len(labels),'images_train_sample':n,'seed':55,'priority':PRIORITY,'files':index,
              'limits':'Targeted sample, not a random error-rate estimator. No val images or frozen val modified.'}
    (out/'sample_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'labels':len(labels),'images':n,'bytes':sum(r['bytes'] for r in index)}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--images',type=int,default=80);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();fetch(args.out,args.images,args.resume)
