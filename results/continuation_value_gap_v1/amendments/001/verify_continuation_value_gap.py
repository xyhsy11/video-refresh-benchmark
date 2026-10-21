"""Independent Experiment 25 verifier. Does not import endpoint implementation.

Synthetic fixtures are explicitly separate from empirical evidence.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/continuation_value_gap_v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def matrix_cr2(rows):
    """Full class-dummy projection with global-family cluster adjustment."""
    classes=sorted({r['label'] for r in rows}); groups=sorted({r['family'] for r in rows})
    X=np.array([[int(r['label']==c) for c in classes] for r in rows],dtype=float)
    y=np.array([r['value'] for r in rows]); bread=np.linalg.inv(X.T@X)
    H=X@bread@X.T; residual=np.eye(len(y))-H
    beta=bread@X.T@y; contrast=np.ones(len(classes))/len(classes)
    B=[]
    for g in groups:
        ix=np.array([i for i,r in enumerate(rows) if r['family']==g])
        ev,vec=np.linalg.eigh(np.eye(len(ix))-H[np.ix_(ix,ix)])
        assert ev.min()>0
        adj=(vec/np.sqrt(ev))@vec.T
        B.append(contrast@bread@X[ix].T@adj@residual[ix])
    B=np.array(B);K=B@B.T
    return float(contrast@beta),float(np.square(B@y).sum()),float(np.trace(K)**2/np.square(K).sum())


def special_cr2(rows):
    classes=sorted({r['label'] for r in rows});groups=sorted({r['family'] for r in rows}); C=len(classes)
    scores=dict.fromkeys(groups,0.);K=np.zeros((len(groups),len(groups)))
    means=[]
    for c in classes:
        rr=[r for r in rows if r['label']==c];n=len(rr);mean=sum(r['value'] for r in rr)/n;means.append(mean)
        assert n>=2
        k=1/(C*n*math.sqrt(1-1/n));ix=[groups.index(r['family']) for r in rr]
        for r in rr:scores[r['family']]+=k*(r['value']-mean)
        for a in range(n):
            for b in range(n):K[ix[a],ix[b]]+=k*k*((a==b)-1/n)
    return sum(means)/C,sum(v*v for v in scores.values()),float(np.trace(K)**2/(K*K).sum())


def synthetic():
    # Synthetic mathematical fixtures only; no video result is constructed here.
    pos=[(0,*s) for s in combinations(range(1,16),3)];assert len(pos)==455 and len(set(pos))==455
    P=(0,4,8,12)
    import continuation_value_gap as implementation  # Only synthetic path imports main.
    # Independent verifier's empirical path never imports it.
    for t in range(1,13):
        prefix=[v for v in P if v<t]
        for a in [0,1]:
            suffix=list(combinations(range(t+1,16),4-len(prefix)-a))
            expected=[tuple(prefix+([t] if a else [])+list(s)) for s in suffix]
            main=[tuple(pos[j]) for j in implementation.branch(t,a)]
            assert main==expected and len(main)>0
            assert implementation.repair(t,a)==implementation.sequential_repair(t,a)
            assert implementation.repair(t,a) in main
        assert implementation.repair(t,int(t in P))==P
    assert implementation.repair(2,1)==(0,2,8,12)
    assert implementation.repair(4,0)==(0,5,8,12)
    # Exact rational sequential remaining-budget/positions recursion.
    def uniform(n,b):
        if n==0:return {():Fraction(1)} if b==0 else {}
        out={}
        for a in [0,1]:
            if not 0<=b-a<=n-1:continue
            p=Fraction(b,n) if a else Fraction(n-b,n)
            for tail,v in uniform(n-1,b-a).items():out[(a,*tail)]=p*v
        return out
    for n in range(1,8):
        for b in range(n+1):
            dist=uniform(n,b);assert sum(dist.values())==1
            assert set(dist.values())=={Fraction(1,math.comb(n,b))}
            losses={s:sum((j+1)**2*a for j,a in enumerate(s)) for s in dist}
            assert sum(dist[s]*losses[s] for s in dist)==Fraction(sum(losses.values()),len(losses))
    fixtures=[
        [dict(label='a',family=str(j),value=float(v)) for j,v in enumerate([1,2,4,8,16])],
        [dict(label=c,family=g,value=v) for c,g,v in [('a','x',1.),('a','y',3.),('a','z',4.),('b','x',7.),('b','p',1.),('b','q',5.),('b','z',3.)]]]
    for rows in fixtures:
        np.testing.assert_allclose(special_cr2(rows),matrix_cr2(rows),atol=1e-12,rtol=1e-12)
        m=implementation.cr2(rows); np.testing.assert_allclose([m['mean'],m['variance'],m['df']],matrix_cr2(rows),atol=1e-12)
        perm=[dict(label='class_'+r['label'],family='family_'+r['family'],value=r['value']) for r in reversed(rows)]
        np.testing.assert_allclose(special_cr2(rows),special_cr2(perm),atol=1e-12)
    n=len(fixtures[0]);v=np.array([r['value'] for r in fixtures[0]])
    np.testing.assert_allclose(special_cr2(fixtures[0])[1:], [v.var(ddof=1)/n,n-1],atol=1e-12)
    # Gap and regret identities with deliberately adverse fixed tails.
    for qs,qp in [(np.array([1.,2.]),np.array([5.,2.])),(np.array([3.,1.]),np.array([4.,6.]))]:
        r=qp-qs;assert (r>=0).all();assert (qp[0]-qp[1])-(qs[0]-qs[1])==r[0]-r[1]
        a=int(qs[0]>qs[1]);baseline=3.;assert (qp[a]-qp.min())==(baseline-qp.min())-(baseline-qp[a])
    return dict(passed=True,kind='SYNTHETIC_ONLY',checks=['455 schedules','entire prefix including holds',
        'both actions legal at t1..12','repair examples/direct-sequential/baseline','uniform exact rational recursion',
        'branch minima/means','state reset','gap/regret identities','CR2 full projection and shared-family fixture',
        'one-class sample SE/df','label permutation','hierarchical estimate'])


def empirical(dataset):
    dst=OUT/dataset; meta=read(dst/'population.json')
    seal=read(dst/'outcome_seal.json')
    for n,h in seal['sha256'].items():assert sha(ROOT/n)==h
    with np.load(dst/'outcome_table.npz') as z:
        losses=z['loss'];pred=z['prediction'];dc=z['dense_ce'];dpred=z['dense_prediction'];positions=z['positions'];logits=z['candidate_logits']
    expected=[(0,*s) for s in combinations(range(1,16),3)]
    assert positions.tolist()==[list(s) for s in expected]
    assert losses.shape==(len(meta),455) and np.isfinite(losses).all()
    lookup={s:j for j,s in enumerate(expected)};pid=lookup[(0,4,8,12)]
    state=pd.read_csv(dst/'per_state.csv.gz',float_precision='round_trip')
    clip=pd.read_csv(dst/'per_clip.csv',float_precision='round_trip'); estimates=read(dst/'estimates.json')
    assert len(state)==12*len(meta) and len(clip)==len(meta) and list(clip.clip)==[m['clip'] for m in meta]
    maxerr=0.;checked=0; e22err=0.; teacher_errors=0
    old={}
    if dataset=='ucf':
        with gzip.open(ROOT/'results/task_utility_observability_v1/states_execution_periodic.json.gz','rt') as f:
            old={(r['clip'],r['t']):r for r in json.load(f) if 1<=r['t']<=12}
    recomputed=[]
    for i,m in enumerate(meta):
        rows=state.iloc[i*12:(i+1)*12];vals=[]
        assert list(rows.t)==list(range(1,13)) and set(rows.clip)=={m['clip']}
        L=losses[i];y=m['y'];base=L[pid]
        for _,r in rows.iterrows():
            t=int(r.t);prefix=[v for v in (0,4,8,12) if v<t];qs=[];qu=[];qp=[];rid=[];branchids=[]
            for a in [0,1]:
                ss=[tuple(prefix+([t] if a else [])+list(s)) for s in combinations(range(t+1,16),4-len(prefix)-a)]
                ids=[lookup[s] for s in ss];branchids+=ids
                ll=[float(L[j]) for j in ids];qs.append(min(ll));qu.append(math.fsum(ll)/len(ll))
                assert int(r['refresh_count' if a else 'hold_count'])==len(ss)
                b=4-len(prefix)-a; tail=[x for x in [4,8,12] if x>t]
                while len(tail)>b:tail.pop(0)
                for u in range(t+1,16):
                    if len(tail)==b:break
                    if u not in tail:tail.append(u)
                sc=tuple(prefix+([t] if a else [])+sorted(tail));j=lookup[sc];rid.append(j);qp.append(float(L[j]))
                assert j==int(r['refresh_schedule' if a else 'hold_schedule'])
            ds=qs[0]-qs[1];dpp=qp[0]-qp[1];du=qu[0]-qu[1]
            a=int(t in [0,4,8,12]) if abs(ds)<=1e-10 else int(ds>0)
            ap=int(t in [0,4,8,12]) if abs(dpp)<=1e-10 else int(dpp>0)
            q=Fraction(4-len(prefix),16-t);vu=(1-float(q))*qu[0]+float(q)*qu[1]
            want=dict(Q_star_H=qs[0],Q_star_R=qs[1],Q_P_H=qp[0],Q_P_R=qp[1],Q_U_H=qu[0],Q_U_R=qu[1],
                D_star=ds,D_P=dpp,D_U=du,V_U=vu,G_star=base-qp[a],G_matched=base-min(qp),R_star=qp[a]-min(qp),
                G_star_U=vu-qu[a],G_matched_U=vu-min(qu),R_star_U=qu[a]-min(qu),
                accuracy_star=int(pred[i,rid[a]]==y),accuracy_matched=int(pred[i,rid[ap]]==y),
                correct_to_wrong=int(pred[i,pid]==y and pred[i,rid[a]]!=y),wrong_to_correct=int(pred[i,pid]!=y and pred[i,rid[a]]==y))
            assert a==int(r.a_star); assert abs((dpp-ds)-((qp[0]-qs[0])-(qp[1]-qs[1])))<1e-10
            assert min(np.array(qp)-qs)>=-1e-10
            for k,v in want.items():maxerr=max(maxerr,abs(float(r[k])-v))
            assert maxerr<1e-10
            if old:
                o=old[m['clip'],t];assert o['history']==prefix
                e22err=max(e22err,abs(ds-o['gain_budget']))
            if len(logits):
                z=logits[i];zc=z-z.mean(1,keepdims=True);obj=[]
                for j in sorted(branchids):
                    sc=expected[j];cache=[max(x for x in sc if x<=u) for u in range(16)]
                    v=np.mean(zc[cache],axis=0)-zc.mean(0);obj.append((j,float(np.mean(v*v))))
                best=min(v for j,v in obj);j=next(j for j,v in obj if v<=best+1e-12);at=int(t in expected[j])
                teacher_errors+=int(at!=int(r.a_Z_pool));want['G_Z_pool']=base-qp[at]
                assert abs(r.G_Z_pool-want['G_Z_pool'])<1e-10
            vals.append(want);checked+=1
        row=dict(**m)
        for k in vals[0]:row[k]=math.fsum(v[k] for v in vals)/12
        for k,v in row.items():
            if isinstance(v,(int,float)):assert abs(float(clip.iloc[i][k])-v)<1e-10
        recomputed.append(row)
    assert e22err<1e-10 and teacher_errors==0
    # Independent explicit hierarchy; no main weights/aggregation imported.
    def hierarchy(key):
        by=defaultdict(lambda:defaultdict(list))
        for r in recomputed:by[r['label']][r['family']].append(r[key])
        return math.fsum(math.fsum(math.fsum(v)/len(v) for v in g.values())/len(g) for g in by.values())/len(by)
    for key in recomputed[0]:
        if key in estimates['estimates']:
            assert abs(hierarchy(key)-estimates['estimates'][key]['mean'])<1e-10
    cf=pd.read_csv(dst/'class_family.csv',float_precision='round_trip');cf=cf[cf.endpoint=='G_star']
    rr=[dict(label=r.label,family=r.family,value=r.value) for _,r in cf.iterrows()]
    counts=cf.groupby('label').size()
    if counts.min()<2:
        assert estimates['primary']['interval'] is None
    else:
        result=matrix_cr2(rr);np.testing.assert_allclose(result,special_cr2(rr),atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(result,[estimates['primary'][k] for k in ['mean','variance','df']],atol=1e-12,rtol=1e-10)
    # Separate scalar affine/raw-feature replay on hash-selected genuine clips, all schedules.
    rawerr=0.; rawcount=0
    ranked=sorted(range(len(meta)),key=lambda i:hashlib.sha256(('20261026|'+meta[i]['clip']).encode()).hexdigest())[:3]
    hmdb_inputs={m['clip']:m for m in read(OUT/'hmdb_population.json')['videos']} if dataset=='hmdb' else None
    if dataset=='ucf':
        cohort=read(OUT/'ucf_manifest.json');source={m['clip']:m for m in cohort['clips']}
        hp=read(ROOT/read(ROOT/'results/ucf101_causal_development_v1/recognition_head_reuse.json')['path'])['parameters']
    else:hp=read(OUT/'hmdb_head.json')['parameters']
    head={k:np.array(v) for k,v in hp.items()}
    for i in ranked:
        if dataset=='ucf':
            m=source[meta[i]['clip']];p=ROOT/m['feature']['path']
            if not p.exists():continue
            assert sha(p)==m['feature']['sha256']
            with np.load(p) as z:f=z['x3d_xs'].astype(float)
        else:
            m=hmdb_inputs[meta[i]['clip']];p=ROOT/'data/source/continuation_value_gap_v1/features'/(hashlib.sha256(m['clip'].encode()).hexdigest()+'.npz')
            with np.load(p) as z:f=z['features'].astype(float)
        for j,sc in enumerate(expected):
            pooled=np.mean([f[max(x for x in sc if x<=u)] for u in range(16)],axis=0)
            zz=head['weights']@((pooled-head['means'])/head['scales'])+head['bias']
            a=float(zz.max());ce=a+math.log(math.fsum(math.exp(float(v)-a) for v in zz))-float(zz[meta[i]['y']])
            rawerr=max(rawerr,abs(ce-losses[i,j]));assert int(zz.argmax())==pred[i,j]
        rawcount+=1
    assert rawerr<1e-10
    if dataset=='hmdb':assert rawcount==3
    return dict(passed=True,clips=len(meta),states=checked,scalar_state_max_error=maxerr,E22_D_star_max_error=e22err,
                raw_feature_clips=rawcount,raw_feature_all455_CE_max_error=rawerr,teacher_action_errors=teacher_errors,
                primary_interval=estimates['primary'],independent_branch_enumerator='itertools.combinations',
                independent_aggregation='explicit class/family/clip arithmetic',independent_CR2='full dummy projection')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--synthetic',action='store_true');args=parser.parse_args()
    if args.synthetic:
        result=synthetic();p=OUT/'synthetic_verification.json'
    else:
        start=time.perf_counter()
        for n,h in read(OUT/'design_seal.json')['sha256'].items():assert sha(ROOT/n)==h
        for n,h in read(OUT/'input_manifest.json').items():assert sha(ROOT/n)==h
        historical=read(OUT/'historical_git_blobs.json')['lines']
        current=subprocess.check_output(['git','ls-files','-s','results'],cwd=ROOT,text=True).splitlines()
        assert set(historical).issubset(current), 'Historical staged blob changed'
        # Also check working-tree content against original blobs without parsing protected contents.
        for line in historical:
            left,name=line.split('\t',1);blob=left.split()[1]
            actual=subprocess.check_output(['git','hash-object','--',name],cwd=ROOT,text=True).strip()
            assert actual==blob, 'Historical working file changed: '+name
        result=dict(passed=True,ucf=empirical('ucf'))
        if (OUT/'hmdb/estimates.json').exists():result['hmdb']=empirical('hmdb')
        result['historical_result_files_unchanged']=len(historical);result['seconds']=time.perf_counter()-start
        p=OUT/'independent_verification.json'
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
