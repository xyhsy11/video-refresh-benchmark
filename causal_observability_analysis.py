"""Paired source-family inference and figures from genuine executed predictions only."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from causal_observability_common import *


def cluster_plan(entries,repeats=10000):
    families=sorted({r['source_family'] for r in entries});support={f:tuple(sorted({r['label'] for r in entries if r['source_family']==f})) for f in families}
    strata=defaultdict(list)
    for i,f in enumerate(families):strata[support[f]].append(i)
    rng=np.random.Generator(np.random.PCG64(SEED));indices=np.empty((repeats,len(families)),dtype=np.int32)
    offset=0
    for signature,members in sorted(strata.items()):
        indices[:,offset:offset+len(members)]=np.asarray(members)[rng.integers(len(members),size=(repeats,len(members)))];offset+=len(members)
    return families,indices,[dict(classes=list(s),families=[families[i] for i in ids],singleton=len(ids)==1) for s,ids in sorted(strata.items())]


def paired_interval(entries,values,classes,families,indices):
    sums=np.zeros((len(classes),len(families)));counts=np.zeros_like(sums)
    for row,value in zip(entries,values):
        c=classes.index(row['label']);f=families.index(row['source_family']);sums[c,f]+=value;counts[c,f]+=1
    if (counts.sum(1)==0).any():raise Blocker('Class missing from paired estimand')
    draws=np.zeros(len(indices));means=[]
    for c in range(len(classes)):
        numerator=sums[c][indices].sum(1);denominator=counts[c][indices].sum(1)
        if (denominator==0).any():raise Blocker('Cluster bootstrap lost a class')
        draws+=numerator/denominator/len(classes);means.append(sums[c].sum()/counts[c].sum())
    ci=np.percentile(draws,[2.5,97.5],method='linear')
    return dict(mean=float(np.mean(means)),CI95_lower=float(ci[0]),CI95_upper=float(ci[1]),interval_width=float(ci[1]-ci[0]),clusters=len(families),clips=len(entries))


def score(logits,label,dense=None):
    z=np.asarray(logits,dtype=float);lp=z-logsumexp(z)
    result=dict(CE=float(-lp[label]),correct=int(z.argmax()==label))
    if dense is not None:
        d=np.asarray(dense);ld=d-logsumexp(d)
        result.update(dense_agreement=int(d.argmax()==z.argmax()),KL=float(np.exp(ld) @ (ld-lp)))
    return result


def summarize(manifest):
    frame=pd.read_csv(OUT/'per_source_metrics.csv',float_precision='round_trip')
    entries=[v for v in manifest['videos'] if v['role']=='test'];classes=manifest['classes']
    families,idx,strata=cluster_plan(entries)
    np.save(OUT/'bootstrap_cluster_indices.npy',idx,allow_pickle=False)
    save(OUT/'bootstrap_clusters.json',dict(families=families,strata=strata,seed=SEED,repeats=10000,
        conditioning='families resampled within exact class-support signatures; singleton strata have no estimated between-family variance'))
    keys=[v['label']+'/'+v['clip'] for v in entries]
    pivot=frame.pivot(index='key',columns='method',values='CE').loc[keys]
    primary=paired_interval(entries,(pivot.periodic-pivot.history_motion).to_numpy(),classes,families,idx)
    save(OUT/'primary_interval.json',dict(contrast='CE(periodic)-CE(history_motion)',primary=True,**primary))
    contrasts={
        'HplusM_vs_H':pivot.history-pivot.history_motion,
        'observability_gap_HplusM':pivot.history_motion-pivot.Z_pool_offline,
        'offline_advantage_over_periodic':pivot.periodic-pivot.Z_pool_offline,
    }
    secondary=[dict(contrast=k,primary=False,**paired_interval(entries,v.to_numpy(),classes,families,idx)) for k,v in contrasts.items()]
    aggregates=[]
    for method in frame.method.unique():
        f=frame[frame.method==method].set_index('key').loc[keys]
        for metric in ['CE','correct','dense_agreement','KL']:
            if metric in f and f[metric].notna().all():
                aggregates.append(dict(method=method,metric=metric,**paired_interval(entries,f[metric].to_numpy(),classes,families,idx)))
    pd.DataFrame(secondary).to_csv(OUT/'secondary_intervals.csv',index=False,float_format='%.17g')
    pd.DataFrame(aggregates).to_csv(OUT/'method_intervals.csv',index=False,float_format='%.17g')
    if (OUT/'per_source_decisions.csv').exists():
        decision=pd.read_csv(OUT/'per_source_decisions.csv',float_precision='round_trip');decision_intervals=[]
        for method in decision['method'].unique():
            f=decision[decision['method']==method].set_index('key').loc[keys]
            for metric in ['teacher_agreement','helpful','harmful']:
                decision_intervals.append(dict(method=method,metric=metric,secondary='marginal/unadjusted source-level rate',
                    **paired_interval(entries,f[metric].to_numpy(),classes,families,idx)))
        pd.DataFrame(decision_intervals).to_csv(OUT/'decision_intervals.csv',index=False,float_format='%.17g')
    frame.groupby(['label','method'])[['CE','correct','dense_agreement','KL']].mean().reset_index().to_csv(OUT/'per_class_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(dict(key=keys,source_family=[v['source_family'] for v in entries],label=[v['label'] for v in entries],Delta_CE=(pivot.periodic-pivot.history_motion).to_numpy(),HplusM_minus_offline=(pivot.history_motion-pivot.Z_pool_offline).to_numpy())).to_csv(OUT/'paired_methods.csv',index=False,float_format='%.17g')
    return primary


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    intervals=pd.read_csv(OUT/'method_intervals.csv');f=intervals[intervals.metric=='CE']
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for i,(_,r) in enumerate(f.iterrows()):ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.CI95_lower],[r.CI95_upper-r['mean']]],fmt='o',capsize=4)
    ax.set_yticks(range(len(f)),f.method);ax.set(xlabel='Class-balanced CE, nats',title='HMDB51 split1 recognition; source-family paired intervals\nMethod intervals secondary/marginal; offline diagnostics privileged')
    fig.savefig(folder/'CE_comparisons.png',dpi=180);fig.savefig(folder/'CE_comparisons.pdf');plt.close(fig)
    f=pd.read_csv(OUT/'secondary_intervals.csv');f=f[f.contrast.str.startswith('observability_gap')]
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
    for i,(_,r) in enumerate(f.iterrows()):ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.CI95_lower],[r.CI95_upper-r['mean']]],fmt='o',capsize=4)
    ax.axvline(0,color='gray');ax.set_yticks(range(len(f)),['H+M minus offline Z_pool']);ax.set(xlabel='CE gap, nats',title='Privileged hindsight versus executed causal decisions\nSecondary, marginal/unadjusted source-family interval')
    fig.savefig(folder/'observability_gap.png',dpi=180);fig.savefig(folder/'observability_gap.pdf');plt.close(fig)
    timing=pd.read_csv(OUT/'timing_per_video.csv');medians=timing.groupby('method')['median_seconds'].median()
    quality=intervals[(intervals.metric=='CE')&intervals.method.isin(medians.index)].set_index('method')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for method in quality.index:
        ax.scatter(medians[method]*1000,quality.loc[method,'mean']);ax.annotate(method,(medians[method]*1000,quality.loc[method,'mean']),xytext=(4,4),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Median per-video measured latency, ms',ylabel='Class-balanced CE, nats',title='Quality / actual encoded-memory-to-output cost\nNo periodic latency assigned to offline selectors')
    fig.savefig(folder/'quality_latency.png',dpi=180);fig.savefig(folder/'quality_latency.pdf');plt.close(fig)
    frame=pd.read_csv(OUT/'per_class_metrics.csv').pivot(index='label',columns='method',values='CE')
    delta=(frame.periodic-frame.history_motion).sort_values()
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');ax.bar(range(len(delta)),delta);ax.axhline(0,color='gray');ax.set_xticks(range(len(delta)),delta.index,rotation=90,fontsize=7);ax.set(ylabel='Periodic minus H+M CE, nats',title='Per-class descriptive heterogeneity; no selected subgroup claim')
    fig.savefig(folder/'class_heterogeneity.png',dpi=180);fig.savefig(folder/'class_heterogeneity.pdf');plt.close(fig)
