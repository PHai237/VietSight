"""Build V2 from an immutable built V1 + strictly reviewed, targeted additions.

Uses the existing builder on PRIVATE COPIES, then checks val image+label bytes
against V1. Never rewrites V1, shared processed sources, or configs/val_frozen.txt.
No training/export. Output remains blocked if legacy invalid polygons are present.
"""
import argparse
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
    out=args.out.resolve();inputs=[args.baseline.resolve(),args.frozen.resolve()]+[p.resolve() for p in args.sources]
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError('use new output outside every input')
    if not args.name.endswith('_v2'):raise ValueError('version name must end in _v2')
    frozen_text=args.frozen.read_text(encoding='utf-8')
    frozen={s.strip() for s in frozen_text.splitlines() if s.strip() and not s.startswith('#')}
    val_before=scan(args.baseline,'val')
    expected={'mapillary_vistas/'+name for name in val_before}
    if frozen!=expected:raise ValueError('baseline val keys do not exactly match supplied frozen V1 list')
    nb=load();candidates=[]
    if len({p.name for p in args.sources})!=len(args.sources):raise ValueError('source names must be unique')
    for source in args.sources:
        if not source.name.endswith('_v2'):raise ValueError('addition source must end in _v2')
        names_seen=set()
        for line in (source/'meta.jsonl').read_text(encoding='utf-8').splitlines():
            record=json.loads(line)
            if record['status']!='ok':continue
            if not(record.get('v2_policy')==1 and record.get('coverage_reviewed') is True and record.get('reviewer')) or record.get('flags'):
                raise ValueError('ok addition lacks V2 coverage/quality approval')
            name=record['file']
            if name in names_seen:raise ValueError('duplicate approved image record')
            names_seen.add(name)
            if Path(name).name!=name:raise ValueError('source image path must be a basename')
            image=source/'images'/name;label=source/'labels'/Path(name).with_suffix('.txt')
            if not image.is_file() or not label.is_file():raise ValueError('missing approved image/label')
            rows,problems=nb['parse_label_file'](label,11)
            if problems:raise ValueError('approved label has parse errors')
            for cid,points in rows:
                if len(points)>400 or nb['polygon_problem'](points,record['w'],record['h']):
                    raise ValueError('approved label has invalid/unchecked geometry')
            classes={CLASSES[c] for c,_ in rows}
            if classes.intersection(PRIORITIES):candidates.append((source,record,classes))
    if not candidates:raise ValueError('no strictly reviewed additions containing priority classes; no V2 built')
    rng=random.Random(55);rng.shuffle(candidates)
    queues={c:[i for i,(_,_,classes) in enumerate(candidates) if c in classes] for c in PRIORITIES}
    selected=[];seen=set()
    while len(selected)<args.max_additions and any(queues.values()):
        for c in PRIORITIES:
            while queues[c] and queues[c][-1] in seen:queues[c].pop()
            if queues[c] and len(selected)<args.max_additions:
                index=queues[c].pop();seen.add(index);selected.append(candidates[index])
    out.mkdir(parents=True)
    processed=out/'private_processed';baseline=processed/'mapillary_vistas'
    (baseline/'images').mkdir(parents=True);(baseline/'labels').mkdir()
    base_meta=[]
    for split in ['train','val']:
        for image in sorted((args.baseline/'images'/split).iterdir()):
            if image.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp'}:continue
            label=args.baseline/'labels'/split/image.with_suffix('.txt').name
            if not label.is_file():raise ValueError('baseline label missing')
            if (baseline/'images'/image.name).exists():raise ValueError('baseline duplicate file name across splits')
            shutil.copyfile(image,baseline/'images'/image.name)
            shutil.copyfile(label,baseline/'labels'/label.name)
            base_meta.append({'file':image.name,'orig':split+'/'+image.name,'group':image.name,'status':'ok'})
    (baseline/'meta.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in base_meta),encoding='utf-8')
    source_names=[]
    for source,record,_ in selected:
        destination=processed/source.name
        for sub in ['images','labels']:(destination/sub).mkdir(parents=True,exist_ok=True)
        name=record['file'];label=Path(name).with_suffix('.txt').name
        shutil.copyfile(source/'images'/name,destination/'images'/name)
        shutil.copyfile(source/'labels'/label,destination/'labels'/label)
        with (destination/'meta.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        if source.name not in source_names:source_names.append(source.name)
    private_frozen=out/'frozen_v1_copy.txt';private_frozen.write_text(frozen_text,encoding='utf-8')
    config=out/'v2_build.yaml'
    config.write_text(yaml.safe_dump({'name':args.name,'description':'V2 targeted reviewed additions, exact V1 val identity',
        'sources':[{'name':'mapillary_vistas'}]+[{'name':n} for n in source_names],
        'val':{'frozen_list':str(private_frozen)},'dedup':{'hamming':2}}),encoding='utf-8')
    build_version.main([str(config),'--processed',str(processed),'--out-root',str(out/'versions'),'--val-list',str(private_frozen)])
    dataset=out/'versions'/args.name
    comparison=compare(dataset,args.baseline)
    if not comparison['ok'] or scan(args.baseline,'val')!=val_before or args.frozen.read_text(encoding='utf-8')!=frozen_text:
        raise ValueError('V2 blocked: val identity/leakage/input mutation gate failed')
    stats=json.loads((out/'versions'/f'{args.name}_report/stats.json').read_text(encoding='utf-8'))
    train_invalid=[v for v in stats['invalid_polygons'] if v.startswith('train/')]
    val_invalid=[v for v in stats['invalid_polygons'] if v.startswith('val/')]
    release={'version':args.name,'priority':PRIORITIES,'selected_addition_images':len(selected),
             'frozen_val_matches_v1':True,'input_frozen_unchanged':True,
             'status':'blocked_invalid_train_polygons' if train_invalid else 'prepared_for_review',
             'invalid_train_polygons':train_invalid,'unchanged_legacy_val_warnings':val_invalid,'training_started':False,
             'note':'Copies isolate builder hardlinks from mutable input; no automatic training or publication.'}
    (out/'release_gate.json').write_text(json.dumps(release,ensure_ascii=False,indent=2),encoding='utf-8')
    return release


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',required=True,type=Path);ap.add_argument('--frozen',type=Path,default=ROOT/'configs/val_frozen.txt')
    ap.add_argument('--sources',nargs='+',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--name',default='vietsight_targeted_v2');ap.add_argument('--max-additions',type=int,default=200)
    a=ap.parse_args()
    if not 1<=a.max_additions<=1000:ap.error('max additions must be 1..1000; bounded intake only')
    result=build(a);print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status'].startswith('blocked'):sys.exit(2)
