"""Independent fitting-only checker. No native forwards, fitting or EVAL reads."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from native_head_value_confirmation import read,sha,trajectory
from verify_native_head_value_confirmation import scalar_loss,independent_repair

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/native_head_value_confirmation_v1'
CACHE=ROOT/'data/source/native_head_value_confirmation_v1'
VIEWS={'S':19,'H':432,'HG':464,'HGH':496}


def expected_fold_membership(rows):
    result=np.empty(len(rows),dtype=int)
    for c in range(400):
        ids=sorted((i for i,r in enumerate(rows) if r['y']==c),key=lambda i:hashlib.sha256(
            ('native_head_value_confirmation_v1|20261023|fold|'+rows[i]['component']).encode()).hexdigest())
        if len(ids)!=8:raise ValueError('Exactly eight sources/class required')
        for j,i in enumerate(ids):result[i]=j//2
    return result


def precision_independent(rows,effects):
    variances=[]
    for c in range(400):
        values=[float(x) for r,x in zip(rows,effects) if r['y']==c]
        if len(values)!=8 or not all(math.isfinite(x) for x in values):raise ValueError('Eight finite genuine source effects/class required')
        mean=math.fsum(values)/len(values)
        variances.append(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
    table={str(m):{str(a):1.96*a*math.sqrt(math.fsum(variances)/(400*400*m)) for a in (1.,1.5,2.)} for m in range(8,21)}
    selected=next((m for m in range(8,21) if table[str(m)]['1.5']<=.005),None)
    return dict(per_class_variance=variances,projections=table,selected_m=selected)


def verify_fitting():
    errors=[]; checks={}
    def check(ok,message):
        if not ok:errors.append(message)
    rows=read(OUT/'val_manifest.json.gz');fs=expected_fold_membership(rows)
    check(len({r['component'] for r in rows})==3200,'component leakage')
    check(np.array_equal(fs,read(OUT/'fit_folds.json')),'deterministic folds')
    for f in range(4):
        check(Counter(r['y'] for i,r in enumerate(rows) if fs[i]==f)==Counter({c:2 for c in range(400)}),'class balance fold '+str(f))
        checkpoint=read(OUT/'task1/folds'/f'{f}.json')
        check(checkpoint['folds']==fs.tolist(),'checkpoint fold identity')
        for name,digest in checkpoint['sha256'].items():check(sha(ROOT/name)==digest,'fold artifact '+name)
    summaries=read(OUT/'fit_oof_source_summaries.json')
    with np.load(OUT/'fit_oof_state_predictions.npz',allow_pickle=False) as saved:
        predictions={t:{v:saved[t+'_'+v] for v in VIEWS} for t in ('advantage','drift')}
    for views in predictions.values():
        for a in views.values():check(a.shape==(38400,) and np.isfinite(a).all(),'prediction shape/finiteness')
    x=[];targets={'advantage':[],'drift':[]};effects=[];maximum_scalar=0.
    from itertools import combinations
    schedules=[(0,*tail) for tail in combinations(range(1,16),3)]
    for i,row in enumerate(rows):
        key=hashlib.sha256(('native_head_value_confirmation_v1|20261023|feature|'+row['filename']).encode()).hexdigest()
        p=CACHE/'features'/(key+'.npz');receipt=read(p.with_suffix('.json'))
        check(sha(ROOT/row['path'])==row['encoded_sha256'],'input '+row['filename'])
        check(receipt['array_sha256']==sha(p),'array '+row['filename'])
        with np.load(p,allow_pickle=False) as a:features=a['features'].astype(float);logits=a['logits'].astype(float);thumbs=a['thumbnails']
        x.append(trajectory(thumbs,features,logits))
        losses=np.array([scalar_loss(logits,s,row['y'])[0] for s in schedules])
        with np.load(OUT/'fit_schedule_tables'/(key+'.npz'),allow_pickle=False) as a:
            error=float(abs(a['loss']-losses).max());maximum_scalar=max(maximum_scalar,error)
            check(error<=1e-10,'455 scalar schedule losses '+row['filename'])
        branches=np.array([[losses[schedules.index(independent_repair(t,action))] for action in (0,1)] for t in range(1,13)])
        targets['advantage'].extend(branches[:,0]-branches[:,1])
        targets['drift'].extend([1-float(features[t]@features[max(u for u in (0,4,8,12) if u<t)])/
            max(np.linalg.norm(features[t])*np.linalg.norm(features[max(u for u in (0,4,8,12) if u<t)]),1e-12) for t in range(1,13)])
        periodic=losses[schedules.index((0,4,8,12))];values={}
        check(summaries[i]['component']==row['component'] and summaries[i]['y']==row['y'],'summary source identity')
        for v in VIEWS:
            pred=predictions['advantage'][v][i*12:(i+1)*12]
            actions=np.array([1 if q>1e-12 else 0 if q< -1e-12 else int(t in (0,4,8,12)) for t,q in enumerate(pred,1)])
            values[v]=float(np.mean(periodic-branches[np.arange(12),actions]))
            check(abs(values[v]-summaries[i]['V_'+v])<1e-10,'held-out whole-schedule value')
        effects.append(values['HGH']-values['HG'])
        check(abs(effects[-1]-summaries[i]['primary'])<1e-10,'Delta_HOG source effect')
    x=np.concatenate(x);targets={k:np.array(v) for k,v in targets.items()}
    max_gradient=0.;max_prediction=0.
    for f in range(4):
        tr=np.flatnonzero(np.repeat(fs!=f,12));te=np.flatnonzero(np.repeat(fs==f,12))
        # Six sources/class, twelve states/source: frozen weighting is uniform.
        weights=np.full(len(tr),1/len(tr))
        for target,y in targets.items():
            for view,size in VIEWS.items():
                report=read(OUT/'probes'/f'{f}_{target}_{view}.json');model=report['model']
                # Exact training_identity also includes frozen non-code controls.
                from native_head_confirmation_runtime import identities
                expected_training=hashlib.sha256(('native_head_value_confirmation_v1|20261023|fit|'+json.dumps(dict(
                    train=[rows[i]['encoded_sha256'] for i in np.flatnonzero(fs!=f)],target=target,view=view,fold=f,implementation=identities()),sort_keys=True)).encode()).hexdigest()
                check(report['training_identity']==expected_training,'training identity')
                train=x[tr,:size];mean=np.average(train,axis=0,weights=weights)
                scale=np.sqrt(np.average((train-mean)**2,axis=0,weights=weights));scale[scale<=1e-12]=1
                check(np.allclose(mean,model['means'],rtol=0,atol=1e-10),'training-only mean')
                check(np.allclose(scale,model['scales'],rtol=0,atol=1e-10),'training-only scale')
                beta=np.array(model['coefficients']);intercept=model['intercept']
                standardized=(train-mean)/scale;residual=standardized@beta+intercept-y[tr]
                gradient=np.r_[standardized.T@(weights*residual)+beta,np.dot(weights,residual)]
                g=float(abs(gradient).max());max_gradient=max(max_gradient,g)
                check(g<1e-7,'weighted ridge alpha=1 stationarity')
                pred=(x[te,:size]-mean)/scale@beta+intercept
                e=float(abs(pred-predictions[target][view][te]).max());max_prediction=max(max_prediction,e)
                check(e<1e-9,'genuine held-out prediction identity')
                with np.load(OUT/'task1/folds'/f'{f}.npz',allow_pickle=False) as saved:
                    check(np.array_equal(saved['heldout_rows'],te),'fold checkpoint held-out indices')
                    check(np.array_equal(saved[target+'_'+view],predictions[target][view][te]),'fold checkpoint held-out predictions')
                check(abs(np.mean(abs(residual))-report['train_mae'])<1e-9,'training error')
                check(abs(np.mean(abs(pred-y[te]))-report['heldout_mae'])<1e-9,'held-out error')
                constant=float(np.mean(y[tr]));check(abs(constant-report['constant'])<1e-10,'training constant')
                check(abs(np.mean(abs(constant-y[te]))-report['constant_heldout_mae'])<1e-9,'held-out constant comparison')
    pp=read(OUT/'precision_plan.json');ind=precision_independent(rows,effects)
    check(pp['selected_m']==ind['selected_m'],'smallest feasible m')
    check(np.allclose(pp['per_class_variance'],ind['per_class_variance'],rtol=0,atol=1e-12),'unbiased within-class variances')
    for m in ind['projections']:
        for a,value in ind['projections'][m].items():check(abs(value-pp['projections'][m][a])<1e-12,'precision table/sensitivity')
    checks.update(sources=3200,folds=4,probe_models=32,maximum_scalar_loss_error=maximum_scalar,maximum_ridge_gradient=max_gradient,
                  maximum_heldout_prediction_error=max_prediction,selected_m=ind['selected_m'])
    return dict(passed=not errors,errors=errors,checks=checks,scope='Independent fitting OOF development verification; no fresh evaluation evidence',
                native_forwards=0,probe_fits=0,evaluation_outcomes_accessed=False)
