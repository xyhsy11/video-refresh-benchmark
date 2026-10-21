"""Explicit, isolated HMDB two-role adapter for Experiment 25.

Existing model factories/preprocessing are read-only; no legacy runner or OUT
mutation. No network/download/package installation exists in this adapter.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
from scipy.special import logsumexp
from continuation_value_gap import (ROOT,OUT,DATA,SEED,POS,now,sha,digest,read,save,event,
    seal_check,verify_hashes,weights,pool_weights,analyze_tables,write_csv,cr2)

REVISION='f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0'
CHECKPOINT_SHA='6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86'
CAP_SECONDS=8*3600
CAP_BYTES=20*2**30


class Gate(RuntimeError):
    pass


def prepare():
    seal_check()
    if (OUT/'hmdb_population_seal.json').exists():
        verify_hashes(read(OUT/'hmdb_population_seal.json')['sha256']);return
    from causal_observability_population_v2 import graph_replay,availability,scalar_validate_frames
    prior=ROOT/'results/causal_observability_population_v2'
    with gzip.open(prior/'screening_evidence.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    audit=read(prior/'source_family_audit.json');mapping=graph_replay(rows,audit)
    train,test,_=availability(rows,mapping)
    classes=sorted({r['class_name'] for r in rows})
    assert len(classes)==51 and len(rows)==6766
    assert len(train)==2754 and len(test)==1530, 'Reconcile current counts before exposure; no subset adjustment'
    assert not ({mapping[r['key']] for r in train}&{mapping[r['key']] for r in test})
    oldevents=[json.loads(l) for l in (ROOT/'results/causal_observability_v1/encoder_events.jsonl').read_text().splitlines()]
    exposed={r['clip'] for r in oldevents};assert all(next(v for v in rows if v['key']==k)['official_flag']==1 for k in exposed)
    originals=ROOT/'data/source/causal_observability_v1/videos'
    # Authenticate all originals, retained input/graph evidence and metadata before training.
    missing=[];changed=[]
    for j,r in enumerate(rows):
        p=originals/r['key']
        if not p.is_file():missing.append(r['key'])
        elif sha(p)!=r['sha256']:changed.append(r['key'])
        if j%1000==0:print(f'HMDB authenticated originals {j}/{len(rows)}',flush=True)
    save(OUT/'hmdb_original_authentication.json',dict(at_utc=now(),expected=len(rows),missing=missing,changed=changed,
        verified=len(rows)-len(missing)-len(changed),historical_smoke_clips=sorted(exposed),historical_test_calls=0))
    if missing or changed:raise Gate(f'BLOCKED_INPUT: {len(missing)} missing / {len(changed)} changed original videos')
    videos=[];spans=[];gaps=[]
    for role,subset in [('HEAD_TRAIN',train),('EVAL',test)]:
        for r in sorted(subset,key=lambda r:r['key']):
            scalar_validate_frames(r)
            videos.append(dict(clip=r['key'],label=r['class_name'],family=mapping[r['key']],
                y=classes.index(r['class_name']),role=role,official_flag=r['official_flag'],input_sha256=r['sha256'],
                grid_sha256=digest(r['grid']),decoded_frames=r['decoded_frames'],bytes=r['bytes'],
                path=(originals/r['key']).relative_to(ROOT).as_posix()))
            spans.extend(r['grid']['history_window_seconds']);gaps.extend(np.diff(r['grid']['endpoint_timestamps']).tolist())
    save(OUT/'hmdb_population.json',dict(classes=classes,videos=videos,counts={'HEAD_TRAIN':len(train),'EVAL':len(test)},
        rule='adaptive_history_index_uniform_v2',historical_smoke_clips=sorted(exposed)))
    # Retained real metadata has all PTS/order/identities; freeze the exact source and compact grid digests.
    names=['results/causal_observability_population_v2/'+n for n in ['screening_evidence.jsonl.gz','source_family_audit.json',
        'allocation_manifest.json','per_class_eligibility.csv','archive_member_inventory.json','screening_identity.json']]
    names+=['results/causal_observability_v1/encoder_events.jsonl','results/causal_observability_v1/input_manifest.json']
    names=[n for n in names if (ROOT/n).exists()]
    inv={n:sha(ROOT/n) for n in names}
    inv['data/source/temporal_refresh_generalization_v1/X3D_XS.pyth']=sha(ROOT/'data/source/temporal_refresh_generalization_v1/X3D_XS.pyth')
    assert inv['data/source/temporal_refresh_generalization_v1/X3D_XS.pyth']==CHECKPOINT_SHA
    setup=read(ROOT/'results/temporal_refresh_generalization_v1/dependency_setup.json')
    archive=ROOT/'data/source/temporal_refresh_generalization_v1/pytorchvideo.zip'
    assert sha(archive)==setup['upstream_zip_sha256'];inv[archive.relative_to(ROOT).as_posix()]=sha(archive)
    with zipfile.ZipFile(archive) as z:
        members=[n for n in z.namelist() if '/pytorchvideo/' in n and n.endswith('.py')]
        for n in members:
            p=archive.parent/n
            import hashlib
            assert sha(p)==hashlib.sha256(z.read(n)).hexdigest()
            inv[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'hmdb_input_manifest.json',inv)
    trainmeta=[v for v in videos if v['role']=='HEAD_TRAIN'];evalmeta=[v for v in videos if v['role']=='EVAL']
    summaries={}
    for role,rr in [('HEAD_TRAIN',trainmeta),('EVAL',evalmeta)]:
        w=weights(rr);fm=defaultdict(float)
        for r,wi in zip(rr,w):fm[r['family']]+=float(wi)
        families={r['family'] for r in rr}
        fc=[len({r['family'] for r in rr if r['label']==c}) for c in classes]
        counts=Counter(r['family'] for r in rr)
        summaries[role]=dict(clips=len(rr),global_components=len(families),sum_class_specific_families=sum(fc),
            min_families_per_class=min(fc),max_families_per_class=max(fc),maximum_component_size=max(counts.values()),
            cross_class_components=sum(len({r['label'] for r in rr if r['family']==g})>1 for g in families),
            maximum_weight=max(fm.values()),effective_components=1/sum(v*v for v in fm.values()),
            per_class_clips=dict(Counter(r['label'] for r in rr)))
    save(OUT/'hmdb_population_diagnostics.json',dict(roles=summaries,all_graph_components=len(audit['components']),
        all_graph_cross_class_components=sum(c['n_classes']>1 for c in audit['components']),
        all_graph_maximum_component_size=max(c['size'] for c in audit['components']),
        snippet_duration_seconds_quantiles=np.quantile(spans,[0,.01,.05,.5,.95,.99,1]).tolist(),
        candidate_gap_seconds_quantiles=np.quantile(gaps,[0,.01,.05,.5,.95,.99,1]).tolist(),
        caveat='HMDB adaptive frame-index recipe differs from UCF; physical temporal support differs, not solely semantic domain'))
    # Exploratory variance-transfer projection uses only the 99 UCF two-family classes.
    import pandas as pd
    uc=pd.read_csv(OUT/'ucf/class_family.csv');uc=uc[uc.endpoint=='G_star'];nc=uc.groupby('label').size()
    eligible=nc[nc==2].index;v=float(uc[uc.label.isin(eligible)].groupby('label').value.var(ddof=1).mean())
    projection=[dict(label=r['label'],family=r['family'],value=0.) for r in evalmeta]
    cf={(r['label'],r['family']) for r in projection};C=51
    # Expected variance under transferable independent class-family cell variability, including global cluster design df.
    projected_variance=sum(v/(C*C*len({g for c,g in cf if c==cl})) for cl in classes)
    df=cr2(projection)['df']
    from scipy.stats import t as student_t
    save(OUT/'exploratory_precision_projection.json',dict(ucf_classes=99,excluded_singleton_classes=['BenchPress','JavelinThrow'],
        transfer_cell_variance=v,projected_SE=float(np.sqrt(projected_variance)),projected_halfwidth=float(student_t.ppf(.975,df)*np.sqrt(projected_variance)),
        design_df=df,assumptions='Exploratory independent-cell variance transfer from UCF 99 two-group classes; not a full-cohort UCF variance, power guarantee, or stopping rule; cross-class outcome covariance unknown',
        full_frozen_HMDB_population_always_used=True,at_utc=now()))
    files=['results/continuation_value_gap_v1/'+n for n in ['hmdb_population.json','hmdb_input_manifest.json','hmdb_population_diagnostics.json','exploratory_precision_projection.json']]
    save(OUT/'hmdb_population_seal.json',dict(at_utc=now(),before_training=True,sha256={n:sha(ROOT/n) for n in files}))
    event('HMDB_membership_input_seal',sha256=sha(OUT/'hmdb_population_seal.json'),HEAD_TRAIN=len(train),EVAL=len(test))


class Encoder:
    def __init__(self):
        start=time.perf_counter()
        import torch,pytorchvideo
        from pytorchvideo.models.hub.x3d import x3d_xs
        from benchmark import configure_numerics
        self.torch=torch;self.numerics=configure_numerics(20261026)
        if not torch.cuda.is_available():raise Gate('BLOCKED_INPUT: CUDA unavailable in existing temporal environment')
        source=ROOT/f'data/source/temporal_refresh_generalization_v1/pytorchvideo-{REVISION}'
        assert Path(pytorchvideo.__file__).resolve().is_relative_to(source.resolve())
        checkpoint=source.parent/'X3D_XS.pyth';assert sha(checkpoint)==CHECKPOINT_SHA
        self.model=x3d_xs(pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'],strict=True)
        self.model.eval().to(device='cuda',dtype=torch.float32)
        self.model.blocks[-1].proj.register_forward_hook(self.capture)
        spec=importlib.util.spec_from_file_location('experiment25_pinned_functional',source/'pytorchvideo/transforms/functional.py')
        self.functional=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.functional)
        self.starts=self.completions=0
        self.identity=dict(revision=REVISION,checkpoint_sha256=CHECKPOINT_SHA,numerics_seed=20261026,numerics=self.numerics,
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),pytorchvideo_path=str(pytorchvideo.__file__),
            actual_preprocessing='4 distinct RGB frames /255; mean .45 std .225; pinned bilinear shortside182; center crop182',
            feature='real 2048D preprojection input, averaged across spatial/temporal dimensions',
            native='unchanged X3D-XS native400 projection/averaging executes; activation=None; float32 inference_mode; batch1')
        if (OUT/'hmdb_encoder.json').exists():assert read(OUT/'hmdb_encoder.json')==self.identity
        else:save(OUT/'hmdb_encoder.json',self.identity)
        event('HMDB_encoder_setup',seconds=time.perf_counter()-start,encoder_sha256=sha(OUT/'hmdb_encoder.json'))

    def capture(self,module,args,output):
        self.projection_input=args[0];self.projection_output=output

    def feature(self,rgbs,clip,t,role):
        # Reuse the authenticated pure preprocessing method only; never its initializer/runners.
        from ucf101_proxy_inference import X3D
        tensor=X3D.prepare(self,rgbs).to('cuda',non_blocking=True)
        self.starts+=1;event('HMDB_forward_start',clip=clip,t=t,role=role)
        with self.torch.inference_mode():
            native=self.model(tensor)
            f=self.projection_input.mean(dim=tuple(range(1,self.projection_input.ndim-1)))
            assert f.shape==(1,2048) and self.model.blocks[-1].activation is None
            result=f.cpu().numpy()[0].copy()
        self.torch.cuda.synchronize();self.completions+=1
        event('HMDB_forward_complete',clip=clip,t=t,role=role)
        return result,native

    def native_check(self,native):
        torch=self.torch;h=self.model.blocks[-1]
        f=self.projection_input.mean(dim=tuple(range(1,self.projection_input.ndim-1)))
        projected=self.projection_output.mean(dim=(1,2,3))
        with torch.inference_mode():affine=torch.nn.functional.linear(f,h.proj.weight,h.proj.bias)
        torch.testing.assert_close(affine,projected,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(native,projected,atol=2e-5,rtol=2e-5)
        return dict(native_mean_max_difference=float((native-projected).abs().max().cpu()),
                    affine_mean_max_difference=float((affine-projected).abs().max().cpu()))


def cachepath(m):
    import hashlib
    return DATA/'features'/(hashlib.sha256(m['clip'].encode()).hexdigest()+'.npz')


def resources():
    p=OUT/'collection_resources.json'
    if p.exists():return read(p)
    return dict(cap_seconds=CAP_SECONDS,cap_bytes=CAP_BYTES,cumulative_seconds=0.,attempts=[],reuse=0)


def used_bytes():
    return sum(p.stat().st_size for d in [DATA,OUT] if d.exists() for p in d.rglob('*') if p.is_file())


def collect_one(m,screen,encoder,smoke=False):
    dst=cachepath(m);identity=dst.with_suffix('.json');r=resources()
    if dst.exists() and identity.exists():
        old=read(identity)
        assert old['feature_sha256']==sha(dst) and old['input_sha256']==m['input_sha256']
        assert old['grid_sha256']==m['grid_sha256'] and old['encoder_sha256']==sha(OUT/'hmdb_encoder.json')
        with np.load(dst) as z:assert z['features'].shape==(16,2048) and np.isfinite(z['features']).all()
        r['reuse']+=1;save(OUT/'collection_resources.json',r);return old
    if r['cumulative_seconds']>=CAP_SECONDS:raise Gate('INCOMPLETE_RESOURCE: cumulative collection wall cap reached')
    if used_bytes()>CAP_BYTES:raise Gate('INCOMPLETE_RESOURCE: additional disk cap reached')
    start=time.perf_counter();starts=encoder.starts;completions=encoder.completions
    attempt=dict(clip=m['clip'],role=m['role'],at_utc=now(),status='inflight')
    r['attempts'].append(attempt);save(OUT/'collection_resources.json',r)
    event('HMDB_extract_or_resume',clip=m['clip'],role=m['role'],smoke=smoke)
    try:
        import av
        from research_utils import configure_decoder
        from ucf101_proxy_data import presentation_order
        from causal_observability_population_v2 import adaptive_grid
        path=ROOT/m['path'];assert sha(path)==m['input_sha256']
        encoded=path.read_bytes();frames=[];pts=[];ids=[]
        with av.open(io.BytesIO(encoded)) as container:
            stream=container.streams.video[0];configure_decoder(stream)
            for i,f in enumerate(container.decode(stream)):
                assert f.pts is not None and f.time_base is not None
                frames.append(f);pts.append(float(f.pts*f.time_base));ids.append(dict(index=i,pts=int(f.pts),time_base=str(f.time_base)))
            frames,pts,ids,_=presentation_order(frames,pts,ids)
            grid=adaptive_grid(pts,ids)
            assert ids==screen['presentation_identities'] and grid==screen['grid']
            assert digest(grid)==m['grid_sha256'] and len(frames)==m['decoded_frames']
            features=[];checks=[]
            for t,indices in enumerate(grid['frame_indices']):
                if r['cumulative_seconds']+time.perf_counter()-start>=CAP_SECONDS:
                    raise Gate('INCOMPLETE_RESOURCE: wall cap during clip')
                f,native=encoder.feature([frames[j].to_ndarray(format='rgb24') for j in indices],m['clip'],t,m['role'])
                features.append(f)
                if smoke and t==0:checks.append(encoder.native_check(native))
        f=np.asarray(features,dtype=np.float32);assert f.shape==(16,2048) and np.isfinite(f).all()
        dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.writing')
        with tmp.open('wb') as stream:np.savez_compressed(stream,features=f)
        tmp.replace(dst)
        old=dict(clip=m['clip'],role=m['role'],input_sha256=m['input_sha256'],grid_sha256=m['grid_sha256'],
            encoder_sha256=sha(OUT/'hmdb_encoder.json'),feature_sha256=sha(dst),path=dst.relative_to(ROOT).as_posix(),
            at_utc=now(),forward_starts=encoder.starts-starts,forward_completions=encoder.completions-completions,
            native_checks=checks,peak_cuda_allocated_bytes=encoder.torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=encoder.torch.cuda.max_memory_reserved())
        assert old['forward_starts']==old['forward_completions']==16
        save(identity,old);attempt['status']='complete'
        return old
    except Exception as e:
        attempt['status']='failed';attempt['error']=repr(e);raise
    finally:
        duration=time.perf_counter()-start
        attempt.update(seconds=duration,starts=encoder.starts-starts,completions=encoder.completions-completions)
        r['cumulative_seconds']+=duration
        save(OUT/'collection_resources.json',r)
        event('HMDB_collection_attempt_end',**attempt,cumulative_seconds=r['cumulative_seconds'],additional_bytes=used_bytes())


def recover_inflight():
    # Charge elapsed wall time on interrupted process conservatively; a restart never resets the cap.
    from datetime import datetime
    r=resources()
    for a in r['attempts']:
        if a['status']=='inflight':
            elapsed=(datetime.fromisoformat(now())-datetime.fromisoformat(a['at_utc'])).total_seconds()
            a.update(status='interrupted_charged',seconds=elapsed)
            r['cumulative_seconds']+=elapsed
            event('interrupted_collection_charged',clip=a['clip'],seconds=elapsed)
    save(OUT/'collection_resources.json',r)


def train_head(train,classes):
    if (OUT/'hmdb_head.json').exists():
        assert read(OUT/'hmdb_head_fit.json')['accepted'];return
    import torch
    torch.manual_seed(20261026);torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    means_f=[]
    for m in train:
        p=cachepath(m);identity=read(p.with_suffix('.json'));assert sha(p)==identity['feature_sha256']
        with np.load(p) as a:means_f.append(a['features'].astype(np.float64).mean(0))
    f=np.array(means_f);w=weights(train);means=w@f;sd=np.sqrt(w@np.square(f-means));scales=np.where(sd<=1e-12,1.,sd)
    x=torch.tensor((f-means)/scales,dtype=torch.float64);y=torch.tensor([m['y'] for m in train],dtype=torch.int64)
    sw=torch.tensor(w,dtype=torch.float64);head=torch.nn.Linear(2048,51,dtype=torch.float64)
    opt=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=2000,max_eval=4000,tolerance_grad=1e-9,
        tolerance_change=1e-12,history_size=50,line_search_fn='strong_wolfe')
    count=0;start=time.perf_counter();event('HMDB_head_training_start',seed=20261026,attempt=1)
    def closure():
        nonlocal count
        opt.zero_grad();loss=(torch.nn.functional.cross_entropy(head(x),y,reduction='none')*sw).sum()+.01/2*head.weight.square().sum()
        loss.backward();count+=1;return loss
    opt.step(closure);objective=float(closure().detach());grad=max(float(p.grad.abs().max()) for p in head.parameters())
    W=head.weight.detach().numpy();b=head.bias.detach().numpy()
    # Independent NumPy objective and analytic gradient, not the optimization closure.
    xx=x.numpy();yy=y.numpy();z=xx@W.T+b;lognorm=logsumexp(z,axis=1);prob=np.exp(z-lognorm[:,None]);prob[np.arange(len(y)),yy]-=1
    err=w[:,None]*prob;gw=err.T@xx+.01*W;gb=err.sum(0)
    independent_obj=float(w@(lognorm-z[np.arange(len(y)),yy])+.01/2*np.square(W).sum())
    independent_grad=float(max(abs(gw).max(),abs(gb).max()))
    assert abs(independent_obj-objective)<1e-10 and abs(independent_grad-grad)<1e-10
    accepted=bool(np.isfinite(f).all() and np.isfinite(W).all() and np.isfinite(b).all() and np.isfinite(objective) and independent_grad<=1e-6)
    fit=dict(accepted=accepted,seed=20261026,alpha=.01,attempts=1,iterations=int(opt.state[head.weight]['n_iter']),closure_calls=count,
        objective=objective,gradient_max_abs=grad,independent_objective=independent_obj,independent_gradient_max_abs=independent_grad,
        at_iteration_cap=int(opt.state[head.weight]['n_iter'])>=2000,seconds=time.perf_counter()-start,
        constant_columns=int((sd<=1e-12).sum()),dtype='float64',solver='torch LBFGS lr1 max_iter2000 max_eval4000 grad1e-9 change1e-12 history50 strong_wolfe')
    save(OUT/'hmdb_head_fit.json',fit)
    save(OUT/'hmdb_head.json',dict(classes=classes,parameters=dict(means=means.tolist(),scales=scales.tolist(),weights=W.tolist(),bias=b.tolist())))
    event('HMDB_head_training_complete',**fit)
    if not accepted:raise Gate('VERIFICATION_FAILED: fixed one-attempt head gradient criterion failed; no EVAL calls authorized')


def evaluate(test):
    dst=OUT/'hmdb';dst.mkdir(parents=True,exist_ok=True)
    if (dst/'estimates.json').exists():return
    hp=read(OUT/'hmdb_head.json');h={k:np.array(v) for k,v in hp['parameters'].items()}
    A=pool_weights();L=[];pred=[];dc=[];dp=[];zs=[]
    for m in test:
        p=cachepath(m);assert sha(p)==read(p.with_suffix('.json'))['feature_sha256']
        with np.load(p) as a:f=a['features'].astype(np.float64)
        z=((f-h['means'])/h['scales'])@h['weights'].T+h['bias'];pooled=A@z
        ce=logsumexp(pooled,axis=1)-pooled[:,m['y']];dense=z.mean(0)
        L.append(ce);pred.append(pooled.argmax(1));dc.append(float(logsumexp(dense)-dense[m['y']]));dp.append(int(dense.argmax()));zs.append(z)
    meta=[{k:m[k] for k in ['clip','label','family','y']} for m in test]
    np.savez_compressed(dst/'outcome_table.npz',loss=L,prediction=pred,dense_ce=dc,dense_prediction=dp,positions=POS,candidate_logits=zs)
    save(dst/'population.json',meta)
    save(dst/'outcome_seal.json',dict(at_utc=now(),sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [dst/'outcome_table.npz',dst/'population.json']}))
    result=analyze_tables('hmdb',meta,np.array(L),np.array(pred),dc,dp,np.array(zs))
    event('HMDB_evaluation_complete',primary=result['primary'],clips=len(test))


def run():
    start=time.perf_counter();stage='prepare'
    try:
        prepare();recover_inflight();seal_check();verify_hashes(read(OUT/'hmdb_population_seal.json')['sha256'])
        verify_hashes(read(OUT/'hmdb_input_manifest.json'))
        population=read(OUT/'hmdb_population.json');train=[m for m in population['videos'] if m['role']=='HEAD_TRAIN'];test=[m for m in population['videos'] if m['role']=='EVAL']
        with gzip.open(ROOT/'results/causal_observability_population_v2/screening_evidence.jsonl.gz','rt') as f:
            screens={r['key']:r for r in map(json.loads,f)}
        stage='encoder_setup';encoder=Encoder()
        exposed=set(population['historical_smoke_clips'])
        import hashlib
        smoke=sorted([m for m in train if m['clip'] not in exposed],key=lambda m:hashlib.sha256(f"20261026|{m['input_sha256']}|{m['clip']}".encode()).hexdigest())[:8]
        if not (OUT/'hmdb_smoke_manifest.json').exists():save(OUT/'hmdb_smoke_manifest.json',dict(clips=[m['clip'] for m in smoke],selection='SHA256(20261026|input_sha256|clip) among clean HEAD_TRAIN; excludes old technical smoke clips'))
        assert read(OUT/'hmdb_smoke_manifest.json')['clips']==[m['clip'] for m in smoke]
        stage='training_only_smoke'
        for m in smoke:collect_one(m,screens[m['clip']],encoder,smoke=True)
        res=resources();attempts=[a for a in res['attempts'] if a['status']=='complete' and a['clip'] in {m['clip'] for m in smoke}]
        durations=[a['seconds'] for a in attempts]
        assert len(durations)>=8
        # Maximum observed smoke time * 1.5, full cohort, not an optimistic mean.
        perclip=max(durations)*1.5;completed=sum(cachepath(m).exists() and cachepath(m).with_suffix('.json').exists() for m in train+test)
        projection=res['cumulative_seconds']+perclip*(len(train)+len(test)-completed)
        cachebytes=sum(cachepath(m).stat().st_size for m in smoke); projected_bytes=int(cachebytes/8*(len(train)+len(test))*1.5+200*2**20)
        available=shutil.disk_usage(ROOT).free
        gate=dict(status='PASS',smoke_clips=8,smoke_seconds=durations,conservative_seconds_per_clip=perclip,
            projected_total_collection_seconds=projection,wall_cap_seconds=CAP_SECONDS,
            projected_additional_bytes=projected_bytes,disk_cap_bytes=CAP_BYTES,available_disk_bytes=available,
            expected_new_snippet_forwards=(len(train)+len(test))*16,smoke_forwards_reused=128,
            measured_cumulative_seconds=res['cumulative_seconds'],at_utc=now())
        if projection>CAP_SECONDS or projected_bytes>CAP_BYTES or projected_bytes>available:
            gate['status']='INCOMPLETE_RESOURCE';gate['blocker']='Conservative smoke-based full-cohort collection projection exceeds cap or available disk'
            save(OUT/'hmdb_gate.json',gate);event('HMDB_resource_gate_stopped',**gate);return
        save(OUT/'hmdb_gate.json',gate);stage='HEAD_TRAIN_collection'
        for j,m in enumerate(train):
            collect_one(m,screens[m['clip']],encoder)
            if j%25==0:print(f'HEAD_TRAIN {j+1}/{len(train)}; cumulative {resources()["cumulative_seconds"]:.1f}s',flush=True)
        stage='head_fit';train_head(train,population['classes'])
        assert read(OUT/'hmdb_head_fit.json')['accepted']
        # Seal final head/scalers, every input/grid/model/design before ANY official-test model call.
        sealfile=OUT/'hmdb_evaluation_seal.json'
        train_cache={m['clip']:read(cachepath(m).with_suffix('.json')) for m in train}
        if not sealfile.exists():
            save(OUT/'hmdb_train_collection.json',train_cache)
            names=['design_seal.json','hmdb_population_seal.json','hmdb_population.json','hmdb_input_manifest.json','hmdb_encoder.json',
                   'hmdb_head.json','hmdb_head_fit.json','hmdb_train_collection.json']
            save(sealfile,dict(at_utc=now(),prior_EVAL_calls=0,sha256={str((OUT/n).relative_to(ROOT).as_posix()):sha(OUT/n) for n in names}))
        verify_hashes(read(sealfile)['sha256']);verify_hashes(read(OUT/'hmdb_input_manifest.json'))
        if not (OUT/'hmdb_first_test_exposure.json').exists():
            save(OUT/'hmdb_first_test_exposure.json',dict(at_utc=now(),first_clip=test[0]['clip'],seal_sha256=sha(sealfile),kind='before first EVAL extraction/model call'))
            event('HMDB_first_official_test_exposure',clip=test[0]['clip'],seal_sha256=sha(sealfile))
        else:assert read(OUT/'hmdb_first_test_exposure.json')['seal_sha256']==sha(sealfile)
        stage='EVAL_collection'
        for j,m in enumerate(test):
            collect_one(m,screens[m['clip']],encoder)
            if j%25==0:print(f'EVAL {j+1}/{len(test)}; cumulative {resources()["cumulative_seconds"]:.1f}s',flush=True)
        save(OUT/'hmdb_eval_collection.json',{m['clip']:read(cachepath(m).with_suffix('.json')) for m in test})
        stage='schedule_analysis';evaluate(test)
        gate.update(status='COMPLETE',actual_collection_seconds=resources()['cumulative_seconds'],additional_bytes=used_bytes(),completed_at_utc=now())
        save(OUT/'hmdb_gate.json',gate)
    except Exception as exc:
        status='INCOMPLETE_RESOURCE' if str(exc).startswith('INCOMPLETE_RESOURCE') else 'VERIFICATION_FAILED' if isinstance(exc,AssertionError) or str(exc).startswith('VERIFICATION_FAILED') else 'BLOCKED_INPUT'
        save(OUT/'hmdb_gate.json',dict(status=status,stage=stage,blocker=repr(exc),at_utc=now(),resume_command='& .\\.venv_temporal\\Scripts\\python.exe .\\continuation_value_gap_hmdb.py --resume'))
        event('HMDB_gate_failure',stage=stage,status=status,error=repr(exc))
        print(f'{status} at {stage}: {exc}',flush=True)
        raise
    finally:event('HMDB_adapter_invocation_end',seconds=time.perf_counter()-start,stage=stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.parse_args();run()
