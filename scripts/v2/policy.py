"""Conservative association and ontology gates for V2, without changing V1 helpers."""
from vslib import CLASS_ID
from vslib.geometry import bbox
from vslib.instances import _ride_score, _touch, VEH_CLASS


def resolve_v2(raw,h,w):
    flags=[];out=[]
    visible=[dict(r) for r in raw if r['mask'].any() and r['role']!='ignore']
    vehicles=[r for r in visible if r['role']=='vehicle2']
    humans=[r for r in visible if r['role'] in ('rider','person')]
    owner={};k=max(2,round(.006*max(h,w)))
    for human in humans:
        explicit=human.get('vehicle_id')
        candidates=[]
        for i,v in enumerate(vehicles):
            if human.get('sub') not in (None,v['sub']):continue
            score,ratio=_ride_score(bbox(human['mask']),bbox(v['mask']),human['mask'],v['mask'],k)
            if ratio>0 and _touch(human['mask'],v['mask'],k):candidates.append((i,score,ratio))
        if explicit is not None:
            matches=[i for i,v in enumerate(vehicles) if str(v['id'])==str(explicit)]
            if len(matches)==1:owner.setdefault(matches[0],[]).append(human);continue
            flags.append(f"invalid_vehicle_assignment:{human['id']}")
        elif human.get('walking_bike'):
            out.append((8,human['mask'],str(human['id'])));continue
        elif human['role']=='rider' and len(candidates)==1:
            owner.setdefault(candidates[0][0],[]).append(human);continue
        elif candidates:
            flags.append(f"ambiguous_association:{human['id']}")
        elif human['role']=='rider':flags.append(f"rider_without_vehicle:{human['id']}")
        # Preserve the draft pixels, but an ambiguous image cannot enter train.
        out.append((8,human['mask'],str(human['id'])))
    for i,v in enumerate(vehicles):
        name=VEH_CLASS[v['sub']][bool(owner.get(i))]
        mask=v['mask'].copy()
        for human in owner.get(i,[]):mask|=human['mask']
        out.append((CLASS_ID[name],mask,str(v['id'])))
    for r in visible:
        if r['role']=='direct':
            target=r['target'];seats=r.get('seats')
            if r.get('vehicle_type') in ('tractor_trailer','tractor_and_semitrailer'):
                flags.append(f"split_tractor_and_trailer:{r['id']}")
            if seats is not None and 9<=int(seats)<=15 and target in ('Car','Bus'):
                target='Car'
            if target in ('Truck','Bus','Other vehicle') and r.get('semantic_reviewed') is not True:
                flags.append(f"ontology_review_required:{r['id']}:{target}")
            out.append((CLASS_ID[target],r['mask'],str(r['id'])))
        elif r['role'] in ('drop','blackout'):
            flags.append(f"unresolved_crowd_or_group:{r['id']}")
    for i,(_,a,aid) in enumerate(out):
        for _,b,bid in out[i+1:]:
            overlap=int((a&b).sum())
            if overlap:flags.append(f"overlapping_instances:{aid}:{bid}:{overlap}px")
    return out,flags
