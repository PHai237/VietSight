"""Build Vietnam-only V2, independent VN holdout, read-only V1 val reference.

Reuses the existing builder on private copies. Never imports V1 train images.
No training/GPU. Every image needs location, license and coverage evidence.
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qa'))
import yaml
import build_version
from compare_validation import scan,compare
from vslib import ROOT,CLASSES
from vslib.notebook_lib import load

PRIORITIES=['Motorcycle','Other vehicle','Sign','Bicycle']


def build(args):
    out=args.out.resolve();baseline=args.baseline
    inputs=[args.frozen.resolve()]+[p.resolve() for p in args.sources]
    if baseline:inputs.append(baseline.resolve())
    vn_reference=getattr(args,'vn_val_reference',None)
    if vn_reference:inputs.append(vn_reference.resolve())
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError('use new output outside every input')
    if not args.name.endswith('_v2'):raise ValueError('version name must end in _v2')
    frozen_bytes=args.frozen.read_bytes()
    v1_keys={s.strip() for s in frozen_bytes.decode('utf-8').splitlines() if s.strip() and not s.startswith('#')}
    v1_stems={Path(k).stem for k in v1_keys}
    val_before=scan(baseline,'val') if baseline else None
    if baseline and v1_keys!={'mapillary_vistas/'+name for name in val_before}:
        raise ValueError('reference V1 val does not match frozen V1 list')
    if baseline and any(r['label_sha256'] is None for r in val_before.values()):
        raise ValueError('V1 comparison reference requires original validation labels as well as images')
    nb=load();candidates=[];seen_files=set()
    if len({p.name for p in args.sources})!=len(args.sources):raise ValueError('source names must be unique')
    for source in args.sources:
        if not source.name.endswith('_v2'):raise ValueError('source must end in _v2')
        for line in (source/'meta.jsonl').read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['status']!='ok':continue
            p=r.get('provenance',{})
            if not (r.get('v2_policy')==2 and r.get('coverage_reviewed') is True and r.get('reviewer')) or r.get('flags'):
                raise ValueError('ok image lacks current V2 coverage approval; reconvert old records')
            if not (p.get('country')=='VN' and p.get('location_verified') is True and p.get('location_evidence')
                    and p.get('license_verified') is True and p.get('license_evidence')
                    and p.get('source_url') and p.get('license') and p.get('capture_group')):
                raise ValueError('Vietnam location, license and capture group evidence required per image')
            name=r['file'];key=source.name+'/'+name
            if Path(name).name!=name or name in seen_files:raise ValueError('unsafe or duplicate output image name')
            seen_files.add(name)
            if Path(name).stem in v1_stems or Path(r['orig']).stem in v1_stems:
                raise ValueError('V1 frozen validation image cannot enter VN V2')
            image=source/'images'/name;label=source/'labels'/Path(name).with_suffix('.txt')
            if not image.is_file() or not label.is_file():raise ValueError('missing approved image/label')
            rows,problems=nb['parse_label_file'](label,11)
            if problems:raise ValueError('approved label has parse errors')
            for cid,points in rows:
                if len(points)>400 or nb['polygon_problem'](points,r['w'],r['h']):
                    raise ValueError('approved label has invalid geometry')
            r=dict(r,group=p['capture_group'])
            candidates.append((source,r,{CLASSES[c] for c,_ in rows},key))
    if not candidates:raise ValueError('no reviewed Vietnam images; no trainable dataset created')
    val_list=getattr(args,'vn_val_list',None)
    if val_list and val_list.resolve()==args.frozen.resolve():raise ValueError('VN holdout must not use V1 frozen list')
    if val_list and not val_list.is_file():raise ValueError('supplied VN frozen list must already exist')
    if val_list:
        if not vn_reference:raise ValueError('reuse VN frozen list with --vn-val-reference to freeze image AND label bytes')
        val_bytes=val_list.read_bytes()
        val_keys={s for s in val_bytes.decode('utf-8').splitlines() if s and not s.startswith('#')}
        if not val_keys or not val_keys<={x[3] for x in candidates}:raise ValueError('frozen VN val keys missing')
        val_groups={r['provenance']['capture_group'] for _,r,_,k in candidates if k in val_keys}
    else:
        val_bytes=None;groups=sorted({r['provenance']['capture_group'] for _,r,_,_ in candidates})
        if len(groups)<2:raise ValueError('at least two independent capture groups needed for VN holdout')
        random.Random(55).shuffle(groups);val_groups=set();target=max(1,round(len(candidates)*.2))
        for group in groups[:-1]:
            val_groups.add(group)
            if sum(r['provenance']['capture_group'] in val_groups for _,r,_,_ in candidates)>=target:break
        val_keys={k for _,r,_,k in candidates if r['provenance']['capture_group'] in val_groups}
    val_items=[x for x in candidates if x[3] in val_keys]
    # Canonical camera/video/trip groups are global across export source names.
    pool=[x for x in candidates if x[1]['provenance']['capture_group'] not in val_groups and x[2].intersection(PRIORITIES)]
    random.Random(55).shuffle(pool);selected=[];seen=set()
    queues={c:[i for i,x in enumerate(pool) if c in x[2]] for c in PRIORITIES}
    while len(selected)<args.max_additions and any(queues.values()):
        for c in PRIORITIES:
            while queues[c] and queues[c][-1] in seen:queues[c].pop()
            if queues[c] and len(selected)<args.max_additions:
                i=queues[c].pop();seen.add(i);selected.append(pool[i])
    if not selected or not val_items:raise ValueError('need priority train images and independent VN val')
    out.mkdir(parents=True);processed=out/'private_processed';names=[]
    for source,r,_,key in val_items+selected:
        destination=processed/source.name
        for sub in ['images','labels']:(destination/sub).mkdir(parents=True,exist_ok=True)
        name=r['file'];label=Path(name).with_suffix('.txt').name
        shutil.copyfile(source/'images'/name,destination/'images'/name)
        shutil.copyfile(source/'labels'/label,destination/'labels'/label)
        with (destination/'meta.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        if source.name not in names:names.append(source.name)
    private_frozen=out/'frozen_vn.txt'
    private_frozen.write_text('# VN holdout, seed 55, canonical capture groups\n'+'\n'.join(sorted(val_keys))+'\n',encoding='utf-8')
    config=out/'v2_build.yaml'
    config.write_text(yaml.safe_dump({'name':args.name,'description':'Vietnam-only V2; independent VN holdout; V1 evaluation separate',
        'sources':[{'name':n} for n in names],'val':{'frozen_list':str(private_frozen)},'dedup':{'hamming':2}}),encoding='utf-8')
    build_version.main([str(config),'--processed',str(processed),'--out-root',str(out/'versions'),'--val-list',str(private_frozen)])
    dataset=out/'versions'/args.name;comparison=compare(dataset,reference=vn_reference,near_hamming=2)
    if not comparison['ok']:raise ValueError('VN train/val leakage gate failed')
    if args.frozen.read_bytes()!=frozen_bytes or (baseline and scan(baseline,'val')!=val_before):raise ValueError('V1 reference changed')
    if val_bytes is not None and val_list.read_bytes()!=val_bytes:raise ValueError('VN input holdout changed')
    manifest=[json.loads(s) for s in (out/'versions'/f'{args.name}_report/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if {r['source']+'/'+r['file'] for r in manifest if r['split']=='val'}!=val_keys:
        raise ValueError('dedup changed frozen VN holdout; review duplicate entries before release')
    v1_conflicts=[]
    if baseline:
        refs=sorted((baseline/'images/val').iterdir())
        current=[p for s in ['train','val'] for p in sorted((dataset/'images'/s).iterdir())]
        paths=refs+current;hashes=[nb['dhash64'](p) for p in paths];n=len(refs)
        v1_conflicts=[{'v1':paths[i].name,'vn':paths[j].name,'distance':d}
            for i,j,d in nb['find_duplicate_pairs'](hashes,2) if i<n<=j]
    stats=json.loads((out/'versions'/f'{args.name}_report/stats.json').read_text(encoding='utf-8'))
    status='prepared_for_review' if baseline else 'blocked_v1_val_overlap_not_checked'
    if stats['invalid_polygons'] or v1_conflicts:status='blocked_geometry_or_v1_overlap'
    release={'version':args.name,'policy':'VN_ONLY','priority':PRIORITIES,'selected_train_images':stats['images']['train'],
        'vietnam_validation_images':stats['images']['val'],'v1_train_images_imported':0,'v1_validation_kept_separate':True,
        'v1_reference_checked':bool(baseline),'v1_overlap_candidates':v1_conflicts,'input_frozen_unchanged':True,
        'frozen_v1_sha256':hashlib.sha256(frozen_bytes).hexdigest(),'status':status,'instances':stats['instances'],
        'missing_train_classes':[c for c in CLASSES if not stats['instances']['train'][c]],
        'missing_validation_classes':[c for c in CLASSES if not stats['instances']['val'][c]],
        'invalid_polygons':stats['invalid_polygons'],'training_started':False}
    (out/'release_gate.json').write_text(json.dumps(release,ensure_ascii=False,indent=2),encoding='utf-8')
    return release


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,help='V1 VAL reference ONLY; never imported into V2')
    ap.add_argument('--frozen',type=Path,default=ROOT/'configs/val_frozen.txt')
    ap.add_argument('--vn-val-list',type=Path,help='existing independent VN holdout list, never V1 list')
    ap.add_argument('--vn-val-reference',type=Path,help='prior immutable VN dataset; required when reusing VN val list')
    ap.add_argument('--sources',nargs='+',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--name',default='vietsight_vietnam_v2');ap.add_argument('--max-additions',type=int,default=200)
    args=ap.parse_args()
    if not 1<=args.max_additions<=1000:ap.error('max train images must be 1..1000')
    result=build(args);print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status'].startswith('blocked'):sys.exit(2)
