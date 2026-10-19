"""REAL development-only retrospective review; no training, CUDA or video reads.

Default is read-only JSON output. --write retains the descriptive case tables.
These outcome-defined subsets are exploratory, never new confirmation cohorts.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parent


def summarize():
    quality=pd.read_csv(OUT/'development_per_clip_metrics.csv',float_precision='round_trip')
    pivot=quality.pivot(index=['clip','label','group'],columns='method',values='CE')
    cases=pivot[['periodic','history','history_motion','dense16','Z_pool_OFFLINE_privileged']].reset_index()
    cases['HplusM_minus_periodic_CE']=cases['history_motion']-cases['periodic']
    cases['category']=np.where(cases['HplusM_minus_periodic_CE']>1e-12,'higher_CE_than_periodic',
        np.where(cases['HplusM_minus_periodic_CE']< -1e-12,'lower_CE_than_periodic','numerically_tied'))
    decisions=pd.read_csv(OUT/'retrospective_decisions.csv',float_precision='round_trip')
    d=decisions[(decisions['method']=='history_motion')&decisions['discretionary']].merge(cases[['clip','category']],on='clip',validate='many_to_one')
    counts=d.groupby('clip').agg(discretionary_decisions=('t','size'),teacher_agreement=('teacher_agreement','mean'),
        chosen_harmful=('chosen_harmful','sum'),missed_helpful=('missed_helpful','sum'),
        teacher_positive_task_harmful=('teacher_positive_task_harmful','sum'),
        mean_current_MAD=('current_MAD','mean'),mean_imitation_score=('imitation_probability','mean')).reset_index()
    cases=cases.merge(counts,on='clip',validate='one_to_one')
    live=pd.read_csv(OUT/'live_state_distribution.csv',float_precision='round_trip')
    states=live.merge(cases[['clip','category']],on='clip',validate='many_to_one')
    state_means=states.groupby('clip')[['standardized_abs_mean','fraction_abs_gt3']].mean().reset_index()
    cases=cases.merge(state_means,on='clip',validate='one_to_one')
    result=[]
    for category,c in cases.groupby('category',sort=True):
        dd=d[d['category']==category]
        selected=dd[dd['selected']];teacher_positive=dd[dd['teacher_refresh']==1]
        corr=spearmanr(dd['current_MAD'],dd['local_CE_gain']).statistic if dd['current_MAD'].nunique()>1 else None
        result.append(dict(category=category,clips=len(c),groups=len(c[['label','group']].drop_duplicates()),classes=c['label'].nunique(),
            mean_clip_CE_difference=float(c['HplusM_minus_periodic_CE'].mean()),
            discretionary_decisions=len(dd),teacher_agreement=float(dd['teacher_agreement'].mean()),
            selected_harmful_count=int(selected['harmful'].sum()),
            selected_harmful_fraction=float(selected['harmful'].mean()) if len(selected) else None,
            missed_helpful_count=int(dd['missed_helpful'].sum()),
            teacher_positive_task_harmful_fraction=float(teacher_positive['harmful'].mean()) if len(teacher_positive) else None,
            MAD_local_task_gain_spearman=float(corr) if corr is not None and np.isfinite(corr) else None,
            imitation_Brier_against_PROXY_teacher=float(np.mean((dd['imitation_probability']-dd['teacher_refresh'])**2)),
            mean_actual_state_absolute_z=float(c['standardized_abs_mean'].mean()),
            mean_actual_state_fraction_abs_z_gt3=float(c['fraction_abs_gt3'].mean())))
    timing=pd.read_csv(OUT/'timing_per_video.csv',float_precision='round_trip')
    seconds=timing.pivot(index=['clip','label','group'],columns='method',values='median_seconds')
    costs=[]
    for method in ['periodic','periodic_unused_gate','RandomB4_seeded','motion','history','history_motion','dense16']:
        f=timing[timing['method']==method]
        costs.append(dict(method=method,groups=len(f),median_video_median_seconds=float(f['median_seconds'].median()),
            median_paired_latency_minus_periodic_seconds=float((seconds[method]-seconds['periodic']).median()),
            median_paired_dense_over_method_ratio=float((seconds['dense16']/seconds[method]).median()),
            median_thumbnail_seconds=float(f['median_cheap_thumbnail_seconds'].median()),
            median_cheap_statistics_seconds=float(f['median_cheap_statistics_seconds'].median()),
            median_policy_decide_seconds=float(f['median_controller_decide_seconds'].median()),
            median_observation_preparation_seconds=float(f['median_observation_preparation_seconds'].median()),
            median_peak_allocated_CUDA_bytes=float(f['median_peak_cuda_bytes'].median())))
    return cases,dict(scope='POST-HOC DESCRIPTIVE DEVELOPMENT failure subsets within the planned failure analysis; no model or primary change',
        subsets=result,measured_costs=costs,
        interpretation='Imitation scores predict proxy teacher actions, not beneficial true-label task updates. Local gains are descriptive and budget-constrained; skipped positive local gains do not prove globally wrong decisions.',
        weighting='Failure subset summaries are descriptive clips/decisions, not fixed101-class population estimates; no p-values or new primary intervals.',
        overhead='Observation-preparation time includes thumbnail conversion, cheap statistics and history packing. These components overlap and MUST NOT be summed.',
        timing='Per-video medians from30 groups; repeats are not independent sources. Periodic uses the same causal loop, without cheap gate; it is not a separately optimized minimal periodic implementation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    cases,result=summarize()
    if args.write:
        cases.to_csv(OUT/'failure_case_comparison.csv',index=False,float_format='%.17g')
        (OUT/'failure_subset_review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
