"""Read-only bounded inventory of local source folders, including partial downloads.

File counts are point-in-time observations. Objects are annotation records by
geometry, NOT unique physical instances (mirrors can duplicate bbox + polygon).
No files, images, annotations or archives are downloaded or modified.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def inventory(root, max_annotations=60):
    record={'root':str(root.resolve()),'started_utc':datetime.now(timezone.utc).isoformat(),
            'sample_limit_per_source':max_annotations,'sources':[]}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files=sorted(p for p in folder.rglob('*') if p.is_file())
        anns=[p for p in files if p.parent.name=='ann' and p.suffix=='.json']
        images=[p for p in files if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
        source={'name':folder.name,'file_counts_by_suffix':dict(Counter(p.suffix.lower() for p in files)),
                'image_files':len(images),'annotation_files':len(anns),'splits':{},'sample':[]}
        for split in sorted({p.parent.parent.name for p in anns}):
            sa=[p for p in anns if p.parent.parent.name==split]
            source['splits'][split]={'annotation_files':len(sa),
                                     'paired_image_paths':sum((p.parent.parent/'img'/p.name[:-5]).is_file() for p in sa)}
        # Prefer paired annotations; include evenly spread positions, not just
        # the first archive entries (which may all be unlabelled test frames).
        pool=sorted(anns,key=lambda p:(not (p.parent.parent/'img'/p.name[:-5]).is_file(),p.as_posix()))
        paired=[p for p in pool if (p.parent.parent/'img'/p.name[:-5]).is_file()]
        pool=paired or pool
        n=min(max_annotations,len(pool))
        selected=[pool[round(i*(len(pool)-1)/max(1,n-1))] for i in range(n)]
        counts=Counter()
        for path in selected:
            entry={'file':path.relative_to(folder).as_posix()}
            try:
                payload=path.read_bytes();obj=json.loads(payload)
                entry.update(sha256=hashlib.sha256(payload).hexdigest(),objects=len(obj.get('objects',[])))
                for item in obj.get('objects',[]):
                    counts[(item.get('classTitle','unknown'),item.get('geometryType','unknown'))]+=1
                entry['paired_image_exists']=(path.parent.parent/'img'/path.name[:-5]).is_file()
            except (OSError,ValueError) as exc:
                entry['error']=str(exc)
            source['sample'].append(entry)
        source['sample_objects_by_class_geometry']=[{'class':c,'geometry':g,'records':v} for (c,g),v in sorted(counts.items())]
        for name in ['meta.json','LICENSE.md']:
            p=folder/name
            if p.is_file():
                payload=p.read_bytes();source[name+'_sha256']=hashlib.sha256(payload).hexdigest()
        record['sources'].append(source)
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    return record


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--max-annotations',type=int,default=60)
    a=ap.parse_args()
    if a.max_annotations<1 or a.out.exists() or a.out.resolve().is_relative_to(a.root.resolve()):
        ap.error('positive sample limit; new output outside the input required')
    result=inventory(a.root,a.max_annotations)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:s[k] for k in ['name','image_files','annotation_files','splits']} for s in result['sources']],indent=2))
