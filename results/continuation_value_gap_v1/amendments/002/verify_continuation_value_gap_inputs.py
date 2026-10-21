"""Supplemental independent provenance/resource/head verification, no main imports."""
from __future__ import annotations
from collections import Counter,defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/continuation_value_gap_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def main():
    result=dict(passed=True,checks=[])
    a=read(OUT/'amendments/mechanical_001.json')
    assert a['original_seal_sha256']==sha(OUT/'design_seal.json')
    assert a['unchanged_UCF_outcome_seal_sha256']==sha(OUT/'ucf/outcome_seal.json')
    for n,v in a['changed_files'].items():
        assert sha(ROOT/n)==v['amended_sha256'] and sha(ROOT/v['preserved_original'])==v['original_sha256']
    result['checks'].append('Mechanical amendment originals/current code and unchanged UCF table seal')
    prior=ROOT/'results/causal_observability_population_v2'
    hashes=read(prior/'SHA256SUMS.json')
    for n in ['screening_evidence.jsonl.gz','source_family_audit.json','archive_member_inventory.json','screening_identity.json','allocation_manifest.json']:
        p=prior/n;assert sha(p)==hashes[p.relative_to(ROOT).as_posix()]
    inv=read(prior/'archive_member_inventory.json')
    with gzip.open(prior/'screening_evidence.jsonl.gz','rt') as f:screen={r['key']:r for r in map(json.loads,f)}
    originals=ROOT/'data/source/causal_observability_v1';labels={}
    for p in (originals/'splits').rglob('*_test_split1.txt'):
        rel=p.relative_to(originals/'splits').as_posix();assert sha(p)==inv['annotations'][rel]['sha256']
        cl=p.name.removesuffix('_test_split1.txt')
        for line in p.read_text().splitlines():
            fn,flag=line.rsplit(None,1);labels[cl+'/'+fn]=(cl,int(flag))
    assert set(labels)==set(screen) and len(labels)==6766
    graph=read(prior/'source_family_audit.json');neighbors={k:set() for k in labels}
    for e in graph['edges']:
        x,y=e['first'],e['second'];neighbors[x].add(y);neighbors[y].add(x)
        if e['kind']=='exact':assert screen[x]['sha256']==screen[y]['sha256']==e['evidence']
        elif e['kind']=='title':assert screen[x]['title_family']==screen[y]['title_family']==e['evidence']
        elif e['kind']=='near':
            distances=[(int(a,2)^int(b,2)).bit_count() for a,b in zip(screen[x]['fingerprints'],screen[y]['fingerprints'])]
            assert max(distances)<=4 and distances==e['hamming']
            assert min(screen[x]['pts_span'],screen[y]['pts_span'])/max(screen[x]['pts_span'],screen[y]['pts_span'])>=.9
        else:raise AssertionError('Unknown graph edge')
    mapping={};unseen=set(labels)
    while unseen:
        stack=[min(unseen)];members=set(stack)
        while stack:
            for k in neighbors[stack.pop()]:
                if k not in members:members.add(k);stack.append(k)
        unseen-=members;family='family_'+digest(sorted(members))
        mapping.update({k:family for k in members})
    expected_graph={k:c['source_family'] for c in graph['components'] for k in c['members']}
    assert mapping==expected_graph
    testfamilies={mapping[k] for k,(c,f) in labels.items() if f==2}
    expected={k:('EVAL' if f==2 else 'HEAD_TRAIN') for k,(c,f) in labels.items()
              if screen[k]['new_eligible'] and (f==2 or (f==1 and mapping[k] not in testfamilies))}
    pop=read(OUT/'hmdb_population.json');actual={m['clip']:m for m in pop['videos']}
    assert set(actual)==set(expected)
    assert Counter(expected.values())=={'HEAD_TRAIN':2754,'EVAL':1530}
    for k,m in actual.items():
        assert m['role']==expected[k] and m['label']==labels[k][0] and m['official_flag']==labels[k][1]
        assert m['family']==mapping[k] and m['grid_sha256']==digest(screen[k]['grid']) and m['input_sha256']==screen[k]['sha256']
    result['checks'].append('Independent all-clip annotations, conservative BFS global graph and exact two-role reconstruction')
    result['HEAD_TRAIN']=2754;result['EVAL']=1530
    # Input/grid identities for all retained clips; actual decode is checked by collector.
    for m in pop['videos']:
        r=screen[m['clip']];ids=r['presentation_identities'];pts=[float(Fraction(v['pts'])*Fraction(v['time_base'])) for v in ids]
        assert len(pts)>=19 and all(b>a for a,b in zip(pts,pts[1:]))
        ep=[(45+t*(len(pts)-4))//15 for t in range(16)]
        assert ep==r['grid']['endpoint_indices']
        for t,e in enumerate(ep):
            lt=next(j for j,p in enumerate(pts) if p>=pts[e]-47/30);left=min(lt,e-3)
            indices=[(left*3+j*(e-left))//3 for j in range(4)]
            assert indices==r['grid']['frame_indices'][t] and len(set(indices))==4
            assert r['grid']['frame_identities'][t]==[ids[j] for j in indices]
    result['checks'].append('Independent rational PTS/order and integer trailing-grid replay, all4284 clips')
    if (OUT/'hmdb_eval_collection.json').exists():
        modelsha=sha(OUT/'hmdb_encoder.json');head=read(OUT/'hmdb_head.json');h={k:np.array(v) for k,v in head['parameters'].items()}
        collections={**read(OUT/'hmdb_train_collection.json'),**read(OUT/'hmdb_eval_collection.json')}
        assert set(collections)==set(actual)
        means=[];trainmeta=[]
        for m in pop['videos']:
            r=collections[m['clip']];p=ROOT/r['path']
            assert sha(p)==r['feature_sha256'] and r['input_sha256']==m['input_sha256'] and r['grid_sha256']==m['grid_sha256'] and r['encoder_sha256']==modelsha
            assert r['forward_starts']==r['forward_completions']==16
            with np.load(p) as z:f=z['features'].astype(np.float64)
            assert f.shape==(16,2048) and np.isfinite(f).all()
            if m['role']=='HEAD_TRAIN':means.append(f.mean(0));trainmeta.append(m)
        f=np.array(means);C=51
        nc={c:len({m['family'] for m in trainmeta if m['label']==c}) for c in pop['classes']};nclip=Counter((m['label'],m['family']) for m in trainmeta)
        w=np.array([1/(C*nc[m['label']]*nclip[m['label'],m['family']]) for m in trainmeta]);assert abs(w.sum()-1)<1e-12
        mu=w@f;sd=np.sqrt(w@np.square(f-mu));scale=np.where(sd<=1e-12,1.,sd)
        np.testing.assert_allclose(mu,h['means'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(scale,h['scales'],atol=1e-12,rtol=1e-12)
        x=(f-mu)/scale;y=np.array([m['y'] for m in trainmeta]);z=x@h['weights'].T+h['bias'];lse=logsumexp(z,axis=1)
        prob=np.exp(z-lse[:,None]);prob[np.arange(len(y)),y]-=1;err=w[:,None]*prob
        gw=err.T@x+.01*h['weights'];gb=err.sum(0)
        objective=float(w@(lse-z[np.arange(len(y)),y])+.01/2*np.square(h['weights']).sum());gradient=float(max(abs(gw).max(),abs(gb).max()))
        fit=read(OUT/'hmdb_head_fit.json');assert abs(objective-fit['objective'])<1e-10 and gradient<=1e-6
        for n,v in read(OUT/'hmdb_evaluation_seal.json')['sha256'].items():assert sha(ROOT/n)==v
        assert read(OUT/'hmdb_first_test_exposure.json')['seal_sha256']==sha(OUT/'hmdb_evaluation_seal.json')
        resources=read(OUT/'collection_resources.json');events=[json.loads(l) for l in (OUT/'exposure_resource_ledger.jsonl').read_text().splitlines()]
        starts=sum(e['kind']=='HMDB_forward_start' for e in events);completions=sum(e['kind']=='HMDB_forward_complete' for e in events)
        assert starts==completions==68544
        assert abs(resources['cumulative_seconds']-sum(a['seconds'] for a in resources['attempts']))<1e-8
        assert all(a['status']=='complete' for a in resources['attempts']) and len(resources['attempts'])==4284
        assert resources['cumulative_seconds']<=28800
        result.update(head_objective=objective,head_gradient_max_abs=gradient,encoder_starts=starts,encoder_completions=completions,
                      collection_seconds=resources['cumulative_seconds'],cache_reuses=resources['reuse'])
        result['checks'].append('All genuine2048D feature hashes, complete membership, train-only scaler/objective/gradient, model seal, first test exposure and every forward counter')
    else:result['feature_checks']='PENDING_COLLECTION'
    (OUT/'supplemental_input_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
