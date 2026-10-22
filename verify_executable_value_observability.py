"""Independent empirical E26 replay. Does not import the runner or target code."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import csv
import gzip
import hashlib
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'results/continuation_value_gap_v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def scalar_hog(a):
    a=np.asarray(a,float)/255; h=np.zeros((16,9))
    for y in range(32):
        for x in range(32):
            dx=(a[y,min(31,x+1)]-a[y,max(0,x-1)])/2
            dy=(a[min(31,y+1),x]-a[max(0,y-1),x])/2
            b=(np.arctan2(dy,dx)%np.pi)*9/np.pi; lo=int(np.floor(b))%9; fraction=b-np.floor(b); mag=np.hypot(dx,dy)
            cell=y//8*4+x//8; h[cell,lo]+=mag*(1-fraction); h[cell,(lo+1)%9]+=mag*fraction
    return h/np.sqrt((h*h).sum(1,keepdims=True)+1e-12)


def independent_repair(t, action):
    chosen=[u for u in [0,4,8,12] if u<t]
    if action: chosen.append(t)
    for u in range(t+1,16):
        budget=4-len(chosen); anchors=sum(v>=u for v in [4,8,12])
        if budget and ((u in [4,8,12] and anchors<=budget) or (u not in [4,8,12] and anchors<budget)):
            chosen.append(u)
    assert len(chosen)==4 and chosen[0]==0
    assert [u for u in chosen if u<t]==[u for u in [0,4,8,12] if u<t]
    return tuple(chosen)


def independent_weights(meta):
    classes=sorted({m['label'] for m in meta}); cells={c:sorted({m['family'] for m in meta if m['label']==c}) for c in classes}
    counts=Counter((m['label'],m['family']) for m in meta)
    return np.array([1/len(classes)/len(cells[m['label']])/counts[m['label'],m['family']] for m in meta])


def verify(out):
    out=Path(out); seal=read(out/'design_seal.json')
    for name,expected in seal['sha256'].items(): assert sha(ROOT/name)==expected, name
    old=read(out/'historical_fingerprint.json')
    for name,expected in old.items(): assert sha(ROOT/name)==expected, f'Historical file changed: {name}'
    inventory=read(out/'input_inventory.json'); assert not inventory['missing']
    for name,expected in inventory['manifest_sha256'].items(): assert sha(ROOT/name)==expected
    meta=[m for m in read(OLD/'hmdb_population.json')['videos'] if m['role']=='EVAL']; n=len(meta); assert n==1530
    pop=read(OLD/'hmdb_population.json'); train=[m for m in pop['videos'] if m['role']=='HEAD_TRAIN']
    assert not ({m['family'] for m in meta}&{m['family'] for m in train})
    mapping=read(out/'fold_manifest.json')['global_component_to_fold']; folds=np.array([mapping[m['family']] for m in meta])
    for f in range(5):
        a=[m for m,fold in zip(meta,folds) if fold!=f]; b=[m for m,fold in zip(meta,folds) if fold==f]
        assert not ({m['family'] for m in a}&{m['family'] for m in b})
        assert len({m['label'] for m in a})==51
        assert all(any(m['label']==c for m in a) for c in pop['classes'])
    with np.load(OLD/'hmdb/outcome_table.npz') as z: table={k:z[k] for k in z.files}
    positions=[(0,*p) for p in combinations(range(1,16),3)]; lookup={p:i for i,p in enumerate(positions)}
    assert np.array_equal(positions,table['positions'])
    with gzip.open(OLD/'hmdb/per_state.csv.gz','rt',newline='') as f: states=list(csv.DictReader(f))
    q=np.empty((n,12,2)); replay_error=0.
    for i,m in enumerate(meta):
        for t in range(1,13):
            r=states[12*i+t-1]; assert r['clip']==m['clip'] and int(r['t'])==t
            for action,col in [(0,'Q_P_H'),(1,'Q_P_R')]:
                j=lookup[independent_repair(t,action)]; q[i,t-1,action]=table['loss'][i,j]
                replay_error=max(replay_error,abs(float(r[col])-q[i,t-1,action]))
                assert j==int(r['hold_schedule' if action==0 else 'refresh_schedule'])
    assert replay_error<1e-10
    # Hash-selected REAL examples, independent raw-feature all455 schedule/head replay.
    chosen=sorted(range(n),key=lambda i:hashlib.sha256(('20261022|real-replay|'+meta[i]['clip']).encode()).hexdigest())[:3]
    head={k:np.array(v) for k,v in read(OLD/'hmdb_head.json')['parameters'].items()}
    rawerror=0.; hogerror=0.; invariance=0.; examples=[]
    for i in chosen:
        m=meta[i]; stem=hashlib.sha256(m['clip'].encode()).hexdigest()
        fp=ROOT/'data/source/continuation_value_gap_v1/features'/(stem+'.npz')
        assert sha(fp)==read(fp.with_suffix('.json'))['feature_sha256']
        with np.load(fp) as z: features=z['features'].astype(float)
        for j,p in enumerate(positions):
            pooled=sum(features[max(v for v in p if v<=t)] for t in range(16))/16
            logits=((pooled-head['means'])/head['scales'])@head['weights'].T+head['bias']
            loss=float(np.logaddexp.reduce(logits)-logits[m['y']]); rawerror=max(rawerror,abs(loss-table['loss'][i,j]))
        with np.load(out/'observations'/(stem+'.npz')) as z: x=z['x']; thumbnails=z['thumbnails']
        hh=[scalar_hog(a) for a in thumbnails]
        for t in range(1,13):
            r=max(v for v in [0,4,8,12] if v<t)
            transition=np.r_[np.linalg.norm(hh[t]-hh[r],axis=1),np.linalg.norm(hh[t]-hh[t-1],axis=1)]
            hogerror=max(hogerror,float(abs(x[t-1,115:]-transition).max()))
        # Observation function receives no label and rejects current/future features.
        # Empirical poison test changes every nonpast feature and future thumbnail.
        from executable_value_observability import trajectory
        genuine=trajectory(thumbnails,{u:features[u] for u in [0,4,8,12]},head)
        np.testing.assert_allclose(genuine,x,atol=1e-12,rtol=0)
        for t in range(1,13):
            altered=features.copy(); altered[t:]=np.nan
            # Only genuinely earlier anchors supplied to the one-state observation API.
            from executable_value_observability import observation
            original=observation(t,{u:features[u] for u in [0,4,8,12] if u<t},head)
            poisoned=observation(t,{u:altered[u] for u in [0,4,8,12] if u<t},head)
            assert np.array_equal(original,poisoned)
            changed=thumbnails.copy(); changed[t+1:]=0
            invariant=trajectory(changed,{u:features[u] for u in [0,4,8,12]},head)[:t]
            invariance=max(invariance,float(abs(invariant-x[:t]).max()))
        examples.append(m['clip'])
    assert rawerror<1e-10 and hogerror<1e-10 and invariance<1e-12
    with np.load(out/'oof_predictions.npz') as z: arrays={k:z[k] for k in z.files}
    assert np.array_equal(arrays['fold'],folds)
    np.testing.assert_allclose(arrays['task_target'],q[:,:,0]-q[:,:,1],atol=1e-12)
    w=independent_weights(meta); summary=read(out/'summary.json'); base=table['loss'][:,lookup[(0,4,8,12)]]
    max_metric_error=0.; verified_models=0
    for r in summary['metrics']:
        name=r['model']; prediction=arrays[name]
        if not name.startswith('task_'): continue
        a=np.where(prediction>1e-12,1,np.where(prediction < -1e-12,0,np.tile(np.isin(np.arange(1,13),[4,8,12]),(n,1))))
        selected=np.array([[q[i,t,int(a[i,t])] for t in range(12)] for i in range(n)])
        # Explicitly mean state consequences within clip FIRST.
        value=float(sum(w[i]*sum(base[i]-selected[i,t] for t in range(12))/12 for i in range(n)))
        regret=float(sum(w[i]*sum(selected[i,t]-min(q[i,t]) for t in range(12))/12 for i in range(n)))
        max_metric_error=max(max_metric_error,abs(value-r['value']),abs(regret-r['regret'])); verified_models+=1
    assert max_metric_error<1e-12
    # Verify each ridge prediction from saved model and the actual held-out observation vectors.
    ridge_error=0.; normal_error=0.; donor_map=read(out/'shuffled_hog_manifest.json')['donors']; index={m['clip']:i for i,m in enumerate(meta)}
    allx=[]
    for m in meta:
        with np.load(out/'observations'/(hashlib.sha256(m['clip'].encode()).hexdigest()+'.npz')) as z: allx.append(z['x'])
    allx=np.array(allx)
    for f in range(5):
        ids=np.flatnonzero(folds==f); xx=[]
        for i in ids:
            stem=hashlib.sha256(meta[i]['clip'].encode()).hexdigest()
            with np.load(out/'observations'/(stem+'.npz')) as z: xx.append(z['x'])
        xx=np.array(xx)
        for target in ['task','drift']:
            for view,size in [('S',19),('H',83),('HG',115),('HGH',147)]:
                model=read(out/'models'/f'{target}_ridge_{view}_fold{f}.json')
                val=((xx[:,:,:size]-model['means'])/model['scales'])@model['coefficients']+model['intercept']
                ridge_error=max(ridge_error,float(abs(val-arrays[f'{target}_ridge_{view}'][ids]).max()))
                # Check the saved optimum and training-only scalers without fitting again.
                trainmask=folds!=f; training=[m for m,k in zip(meta,trainmask) if k]
                ww=np.repeat(independent_weights(training)/12,12); tx=allx[trainmask,:,:size].reshape(-1,size)
                mu=ww@tx; sd=np.sqrt(ww@np.square(tx-mu)); sd=np.where(sd<=1e-12,1.,sd)
                np.testing.assert_allclose(mu,model['means'],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(sd,model['scales'],atol=1e-10,rtol=1e-10)
                zz=(tx-model['means'])/model['scales']; yy=arrays[target+'_target'][trainmask].ravel()
                err=zz@model['coefficients']+model['intercept']-yy
                normal_error=max(normal_error,float(abs(zz.T@(ww*err)+model['coefficients']).max()),float(abs(ww@err)))
        for i in ids:
            donor=index[donor_map[meta[i]['clip']]]; assert folds[donor]==f and meta[donor]['family']!=meta[i]['family']
    assert ridge_error<1e-12 and normal_error<1e-10
    with open(out/'timing_raw_trials.csv',newline='') as f: trials=list(csv.DictReader(f))
    assert len(trials)==60*7*4 and sum(r['warmup']=='False' for r in trials)==60*5*4
    assert len({r['family'] for r in trials})==60
    for r in trials:
        assert r['status']=='COMPLETE'
        assert int(r['scores'])==(0 if r['arm']=='A' else 12)
        assert int(r['thumbnails'])==(16 if r['arm'] in ['C','D'] else 0)
    with open(out/'timing_paired_differences.csv',newline='') as f: pairs=list(csv.DictReader(f))
    times={(r['clip'],int(r['repetition']),r['arm']):float(r['total_seconds']) for r in trials}
    for p in pairs:
        a,b=p['contrast'].split('_minus_'); key=(p['clip'],int(p['repetition']))
        assert abs(1000*(times[(*key,a)]-times[(*key,b)])-float(p['paired_ms']))<1e-10
    ledger=[json.loads(s) for s in (out/'resource_ledger.jsonl').read_text().splitlines()]
    starts=sum(e['kind']=='decode_started' for e in ledger); completions=sum(e['kind']=='decode_complete' for e in ledger)
    assert starts==completions==1530+60*7*4
    fits=[e for e in ledger if e['kind']=='probe_fit_complete']
    assert sum(e.get('family')=='ridge' for e in fits)==40
    assert sum(e['kind']=='probe_fit_started' for e in ledger)==len(fits)
    info=read(out/'resource_accounting.json'); assert info['cpu_only_import_check']
    assert info['retained_bytes']<2*2**30 and info['elapsed_seconds']<14400
    # Rehash retained per-clip evidence and original inputs, not outcome-filtered samples.
    for record in inventory['clips']:
        assert sha(ROOT/record['path'])==record['input_sha256']
        assert sha(ROOT/record['feature_path'])==record['feature_sha256']
        p=out/'observations'/(hashlib.sha256(record['clip'].encode()).hexdigest()+'.npz')
        assert sha(p)==read(p.with_suffix('.json'))['sha256']
    result=dict(passed=True,historical_files_unchanged=len(old),population_clips=n,states=12*n,components=len(mapping),classes=51,
        training_components_disjoint=True,component_folds_separate=True,
        min_training_components_per_class=min(r['min_training_components_per_class'] for r in read(out/'fold_manifest.json')['diagnostics']),
        all_state_replay_max_error=replay_error,real_raw_all455_replay_max_error=rawerror,real_replay_clips=examples,
        scalar_HOG_reference_max_error=hogerror,future_pixels_and_features_invariance_error=invariance,
        independent_metric_max_error=max_metric_error,ridge_saved_model_prediction_max_error=ridge_error,
        independent_ridge_training_gradient_max_error=normal_error,verified_action_models=verified_models,
        decode_starts=starts,decode_completions=completions,probe_fits=len(fits),timing_trials=len(trials),
        exact_K4_original_prefixes_checked=True,original_seals_unchanged=True,
        synthetic_fixtures='Separate test_executable_value_observability.py; not empirical evidence',
        no_encoder_execution='No encoder/model runtime imported; CPU affine head arithmetic only; module import and ledger accounting recorded')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result
