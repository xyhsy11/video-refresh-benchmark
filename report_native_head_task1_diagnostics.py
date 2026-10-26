"""Descriptive diagnostics of genuine held-out fitting predictions only."""
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from native_head_value_confirmation import read,save,sha,consequences,VIEWS
from native_head_confirmation_runtime import feature_path
from run_native_head_value_confirmation import OUT,run_lock
from native_head_confirmation_state import metered


def report():
    rows=read(OUT/'val_manifest.json.gz')
    fs=np.array(read(OUT/'fit_folds.json'));state_folds=np.repeat(fs,12)
    if len(rows)!=3200 or fs.shape!=(3200,):raise ValueError('Complete genuine fixed-fold population required')
    with np.load(OUT/'fit_oof_state_predictions.npz',allow_pickle=False) as saved:
        predictions={t:{v:saved[t+'_'+v] for v in VIEWS} for t in ('advantage','drift')}
    targets={t:[] for t in predictions}
    for row in rows:
        with np.load(feature_path(row),allow_pickle=False) as a:
            actual=consequences(a['logits'],a['features'],row['y'])
        for target in targets:targets[target].extend(actual[target])
    result=dict(scope='Fitting-only OOF development diagnostics; no fresh evaluation or scientific confirmation',targets={},
        predictions_sha256=sha(OUT/'fit_oof_state_predictions.npz'),source_summaries_sha256=sha(OUT/'fit_oof_source_summaries.json'),
        native_forwards=0,probe_fits=0,evaluation_outcomes_accessed=False)
    for target,actual in targets.items():
        y=np.asarray(actual);result['targets'][target]={}
        for view,p in predictions[target].items():
            if p.shape!=y.shape or not np.isfinite(p).all():raise ValueError('Nonfinite or incomplete prediction; no row removal permitted')
            constant=np.empty_like(y);signed_bins={}
            for fold in range(4):
                fit=read(OUT/'probes'/f'{fold}_{target}_{view}.json')
                mask=state_folds==fold;constant[mask]=fit['constant']
                scale=max(abs(fit['constant']),1e-12)
                edges=np.r_[-np.inf,np.array([-2,-1,-.5,0,.5,1,2])*scale,np.inf]
                signed_bins[str(fold)]=[]
                for lo,hi in zip(edges[:-1],edges[1:]):
                    take=mask & (p>=lo) & (p<hi)
                    signed_bins[str(fold)].append(dict(lower=float(lo) if np.isfinite(lo) else None,
                        upper=float(hi) if np.isfinite(hi) else None,rows=int(take.sum()),
                        predicted=float(p[take].mean()) if take.any() else None,
                        observed=float(y[take].mean()) if take.any() else None))
            correlation=spearmanr(y,p).statistic if np.ptp(y)>0 and np.ptp(p)>0 else None
            result['targets'][target][view]=dict(mae=float(np.mean(abs(y-p))),
                constant_heldout_mae=float(np.mean(abs(y-constant))),
                spearman=float(correlation) if correlation is not None else None,
                auroc=float(roc_auc_score(y>0,p)) if len(set(y>0))==2 else None,
                average_precision=float(average_precision_score(y>0,p)) if len(set(y>0))==2 else None,
                near_zero_predictions=int((abs(p)<=1e-12).sum()),nonfinite_predictions=0,
                residual_quantiles=np.quantile(p-y,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
                signed_bins_by_fold=signed_bins,bin_scope='Each held-out fold uses its training-only target constant scale; bins not selected on held-out performance')
    sources=read(OUT/'fit_oof_source_summaries.json');result['source_summary']={}
    for key in ('primary','drift_contrast','V_S','V_H','V_HG','V_HGH','opportunity'):
        x=np.array([r[key] for r in sources])
        result['source_summary'][key]=dict(equal_class_mean=float(np.mean([x[[i for i,r in enumerate(sources) if r['y']==c]].mean() for c in range(400)])),
            quantiles=np.quantile(x,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),negative=int((x<0).sum()),exact_zero=int((x==0).sum()),positive=int((x>0).sum()))
    result['dense_wrong_sources']=sum(r['dense_correct']==0 for r in sources)
    result['interpretation']='Lower drift MAE does not establish better refresh utility. Task utility uses independently reset complete schedules. No held-out source or unfavorable effect is excluded.'
    save(OUT/'task1/oof_prediction_diagnostics.json',result)
    print(json.dumps(dict(source_summary=result['source_summary']),indent=2))


if __name__=='__main__':
    with run_lock(),threadpool_limits(limits=1),metered('cpu','task1-oof-development-diagnostics'):
        report()
