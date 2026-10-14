"""Acquire, collect and confirm frozen refresh policies; resume without resampling."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from fresh_confirmation_data import (strict_json,check_frozen,prior_exclusions,make_sampling_plan,
    ReferenceCollector,acquire_and_collect)
from fresh_confirmation_eval import (focused_checks,compatibility,load_references,replay,per_video,statistics)
from research_utils import ROOT,now,relative_path,sha256_file

SOURCES=['run_fresh_refresh_confirmation.py','fresh_confirmation_data.py','fresh_confirmation_eval.py','fresh_confirmation_report.py']


def csv(path:Path,table:pd.DataFrame) -> None:
    table.to_csv(path,index=False,float_format='%.17g',na_rep='')


def previous_inventory() -> dict:
    files=subprocess.check_output(['git','ls-files','-z'],text=True).split('\0')
    return {name:sha256_file(ROOT/name) for name in files if name and name not in ['README.md','.gitignore']}


def audit_saved(output:Path,protocol:dict,models:dict,settings:dict) -> dict:
    manifest=json.loads((output/'frozen_test_manifest.json').read_text()); exclusions=json.loads((output/'exclusions.json').read_text())
    videos=manifest['videos']; assert len(videos)==protocol['videos']==100
    assert Counter(e['label'] for e in videos)==Counter({c:5 for c in protocol['classes']})
    ids={e['source_video_id'] for e in videos}; hashes={e['file_identity']['sha256'] for e in videos}
    assert len(ids)==len(hashes)==100 and not ids&set(exclusions['source_ids']) and not hashes&set(exclusions['file_hashes'])
    identity=json.loads((output/'frozen_artifact_identity.json').read_text())
    for name,digest in identity['artifact_sha256'].items(): assert sha256_file(output/name)==digest,name
    assert sha256_file(output/'frozen_test_manifest.json')==identity['manifest_sha256']
    traces=pd.read_csv(output/'test_policy_traces.csv',float_precision='round_trip')
    trajectories=pd.read_csv(output/'test_trajectories.csv',float_precision='round_trip')
    expected=protocol['videos']*(1+len(protocol['budgets'])*(len(protocol['policies'])-1+10))
    assert len(trajectories)==expected==5500 and len(traces)==expected*16==88000
    summaries=trajectories.set_index('trajectory_id')
    for tid,g in traces.groupby('trajectory_id',sort=False):
        g=g.sort_values('candidate_ordinal'); m=summaries.loc[tid]
        assert len(g)==16 and g.candidate_ordinal.tolist()==list(range(16)) and g.refresh.sum()==m.budget and bool(g.refresh.iloc[0])
        latest=0
        for row in g.itertuples():
            t=int(row.candidate_ordinal)
            if t:
                assert row.cached_before_ordinal==latest
                if row.remaining_budget_before==0: assert not row.refresh
                if row.remaining_budget_before==16-t: assert row.refresh
                feasible=[v for v in [row.evaluator_Q_reuse_sum,row.evaluator_Q_refresh_sum] if pd.notna(v)]
                chosen=row.evaluator_Q_refresh_sum if row.refresh else row.evaluator_Q_reuse_sum
                assert pd.notna(chosen) and abs(row.evaluator_action_regret-(chosen-min(feasible))/16)<=1e-12
            if row.refresh: latest=t; assert abs(row.output_feature_error)<=1e-12
            assert row.cached_output_ordinal==latest
        assert abs(g.output_feature_error.mean()-m.mean_output_error)<=1e-12
        assert abs(g.evaluator_action_regret.sum()-(m.mean_output_error-m.oracle_optimum_error))<=1e-12
        assert json.loads(m.refresh_positions)==g[g.refresh].candidate_ordinal.tolist()
    computed=per_video(trajectories,100,protocol)
    saved=pd.read_csv(output/'test_per_video_metrics.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(computed,saved,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    aggregate,paired,intervals,class_stats=statistics(saved,trajectories,protocol)
    assert aggregate==json.loads((output/'test_aggregate_metrics.json').read_text())
    assert intervals==json.loads((output/'bootstrap_intervals.json').read_text())
    # Independent means and bootstrap loop directly from per-video saved rows.
    rng=np.random.Generator(np.random.PCG64(protocol['uncertainty']['seed']))
    idx=rng.integers(0,100,size=(10000,100),dtype=np.int64)
    for interval in intervals['comparisons']:
        pivot=saved[saved.budget==interval['budget']].pivot(index='source_video_id',columns='policy',values='mean_output_error').sort_index()
        deltas=pivot[interval['reference']].to_numpy()-pivot[interval['candidate']].to_numpy()
        independent=np.array([np.sum(deltas[row])/100 for row in idx],dtype=np.float64)
        bounds=np.quantile(independent,[.025,.975],method='linear')
        np.testing.assert_allclose(bounds,[interval['CI95_lower'],interval['CI95_upper']],rtol=1e-12,atol=1e-12)
        assert abs(sum(deltas)/100-interval['mean_paired_absolute_improvement'])<=1e-12
    return {'passed':True,'checked_at_utc':now(),'videos':100,'classes':20,'five_per_class':True,'source_hash_exclusions_and_uniqueness':True,
        'trajectories':5500,'trace_rows':88000,'per_video_rows':2800,'saved_cache_counters_errors_regret_verified':True,
        'random_seed_averaging_and_all_metrics_reproduced':True,'bootstrap_shared_indices_reproducible_and_independent_loop_verified':True}


def execute(args,output:Path) -> int:
    frozen=args.frozen_run
    protocol_path=frozen/'fresh_test_protocol.json'
    if not protocol_path.exists(): raise FileNotFoundError(relative_path(protocol_path))
    protocol=json.loads(protocol_path.read_text()); models,settings=check_frozen(frozen,protocol)
    copied=output/'fresh_test_protocol.json'
    if copied.exists(): assert copied.read_bytes()==protocol_path.read_bytes()
    else: copied.write_bytes(protocol_path.read_bytes())
    config={'frozen_run':relative_path(frozen),'protocol_sha256':sha256_file(protocol_path),'device':args.device,
        'max_download_gib':args.max_download_gib,'download_workers':args.download_workers}
    status_path=output/'status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text()); assert status['configuration']==config,'Resume configuration mismatch'
        assert status['source_code_sha256']=={name:sha256_file(ROOT/name) for name in SOURCES},'Resume implementation identity changed; preserve original and document correction'
        assert status['protected_original_sha256']==previous_inventory()
        if status['status']=='complete':
            result=audit_saved(output,protocol,models,settings)
            print('Existing completed confirmation independently reproduced; no new acquisition or collection.',flush=True)
            print(result); return 0
        if not args.resume: raise RuntimeError('Incomplete run exists; use --resume; cohort will not be resampled')
    else:
        status={'status':'initializing','started_at_utc':now(),'configuration':config,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_code_sha256':{name:sha256_file(ROOT/name) for name in SOURCES},'protected_original_sha256':previous_inventory()}
        strict_json(status_path,status)
    execution_path=output/'execution_plan.json'
    if not execution_path.exists():
        strict_json(execution_path,{'frozen_at_utc':now(),'authoritative_protocol_sha256':sha256_file(copied),
            'configuration':config,'sampling':'cached verified shards first; stable SHA256 URL ranking within cached/uncached groups; candidate SHA256(seed|class|filename) within each archive; same-class next candidate for genuine input ineligibility',
            'exclusions':'freeze union of authoritative210 and all prior persisted membership/inspection source identities; do not rescan on resume',
            'download_accounting':'verified cache bytes free; reserve entire uncached Content-Length before GET, partial/resume counted once; actual response body bytes separately; retransmissions/headers can differ',
            'retry_rules':'metadata/HEAD timeout45s; archive64MiB ranges, four retries with2/4/8s backoff and90s response timeout; preserve failed shard and source candidate for resume; CUDA/machine/config errors block, never replace; decoded invalid input or persistently nonfinite features permits next predetermined same-class candidate',
            'evaluation':'no fitting/threshold selection; all authoritative sparse policies, B2/B4/B8, denseB16; same ten random seeds; original40 development compatibility before fresh evaluation',
            'comparisons':{'primary':'B4 motion value vs periodic','protocol_secondary':['B4 motion value vs pacing','B4 motion value vs no-motion value','corresponding B2/B8 comparisons'],
                           'additional_declared_secondary':['age/budget value vs periodic at B2/B4/B8']},
            'statistics':'Generator(PCG64), seed20261014, common10000x100 integer resampling matrix over sorted whole source IDs; linear2.5/97.5 percentiles; secondary intervals marginal/unadjusted',
            'descriptive':'all frozen-policy means/medians/percentages/gaps, random seed variation, per-class means; top2 positive gains/positive net only if net>1e-12; B4 largest gain/largest loss timelines with source-ID tie break',
            'output_schemas':{'features':'16x512 float32 NPZ plus candidate JSON','motion':'one existing descriptor per decoded frame CSV',
                'manifest':'100 identities, five/class, codec/count/PTS/hash/artifact identities, frozen before policy evaluation',
                'traces':'candidate/cache/budget/score/action/reason plus evaluator-only drift/Q/regret',
                'metrics':'per source/policy/B, random averaged within source','intervals':'paired whole-video differences, CI95, shared resampling SHA'}})
    models,settings=check_frozen(frozen,protocol)
    software=focused_checks(models,settings,protocol); strict_json(output/'software_verification.json',software)
    print(f"Frozen-policy software checks PASS: {len(software['checks'])}",flush=True)
    compatibility_path=output/'development_compatibility.json'
    if not compatibility_path.exists(): compatibility(frozen,protocol,models,settings,output)
    else: assert json.loads(compatibility_path.read_text())['passed']
    manifest_path=output/'frozen_test_manifest.json'
    if not manifest_path.exists():
        exclusions=prior_exclusions(output,protocol)
        plan=make_sampling_plan(output,protocol,exclusions)
        assert plan['annotations_sha256']==sha256_file(Path('data/source/cache_drift_v2/val.csv'))
        assert plan['archive_list_sha256']==sha256_file(Path('data/source/cache_drift_v2/k400_val_path.txt'))
        assert plan['exclusions_sha256']==sha256_file(output/'exclusions.json')
        collector=ReferenceCollector(output,protocol); collector.smoke()
        acquire_and_collect(output,protocol,plan,exclusions,collector,args.max_download_gib,args.download_workers)
    manifest=json.loads(manifest_path.read_text()); assert manifest['status']=='frozen'
    assert len(manifest['videos'])==100 and manifest['class_counts']=={c:5 for c in protocol['classes']}
    progress=json.loads((output/'collection_progress.json').read_text())
    hashes={name:digest for v in progress['videos'] for name,digest in v['artifact_sha256'].items()}
    identity_path=output/'frozen_artifact_identity.json'
    identity={'manifest_sha256':sha256_file(manifest_path),'artifact_sha256':hashes,'models_sha256':protocol['fixed_models_sha256'],'settings_sha256':protocol['fixed_settings_sha256']}
    if identity_path.exists(): assert json.loads(identity_path.read_text())==identity
    else: strict_json(identity_path,identity)
    videos=load_references(manifest['videos'],output,hashes)
    models,settings=check_frozen(frozen,protocol)
    status.update(status='evaluating',fresh_evaluation_opened_at_utc=status.get('fresh_evaluation_opened_at_utc',now())); strict_json(status_path,status)
    # Partial evaluation from an interrupted process is not shown or interpreted.
    traces,trajectories,checks=replay(videos,models,settings,protocol)
    per=per_video(trajectories,100,protocol); aggregate,paired,intervals,classes=statistics(per,trajectories,protocol)
    for name,table in [('test_policy_traces.csv',traces),('test_trajectories.csv',trajectories),('test_per_video_metrics.csv',per),
                       ('paired_comparisons.csv',paired),('per_class_descriptive.csv',classes)]: csv(output/name,table)
    strict_json(output/'test_aggregate_metrics.json',aggregate); strict_json(output/'bootstrap_intervals.json',intervals)
    independent=audit_saved(output,protocol,models,settings)
    # Real-input isolation, with all model/settings parameters still fixed.
    from dataclasses import replace
    import budget_aware_refresh_utils as u
    for v in videos[:2]:
        cutoff=v.cheap.indices[8]
        future=replace(v.cheap,motion=tuple({**f,'mv_available':True,'mean_magnitude_px':1e6,'p90_magnitude_px':1e7} if f['decoded_frame_index']>cutoff else f for f in v.cheap.motion))
        for B in protocol['budgets']:
            for policy in ['age_only','age_plus_motion','paced_age_plus_motion',*u.VALUE_COLUMNS]:
                setting=settings[policy][str(B)] if policy in settings else None
                a=u.plan_deployable(v.cheap,policy,B,models,setting)
                assert a[:9]==u.plan_deployable(future,policy,B,models,setting)[:9]
                evaluated,_=u.evaluate(v.cheap,a,u.ActionOracle(u.distance_matrix(v.references[::-1])),B)
                assert [r['score_raw'] for r in a]==[r['score_raw'] for r in evaluated]
    check_frozen(frozen,protocol); assert status['protected_original_sha256']==previous_inventory()
    release=subprocess.run([sys.executable,str(ROOT/'validate_release.py')],capture_output=True,text=True,cwd=ROOT)
    assert release.returncode==0,release.stdout+release.stderr
    verification={**independent,'software':software,'development_compatibility':json.loads(compatibility_path.read_text()),
        'replay':checks,'reference_features':1600,'models_settings_hashes_unchanged_before_and_after':True,
        'real_future_motion_and_RGB_information_isolation':True,'prior_sources_snapshots_unchanged':len(status['protected_original_sha256']),
        'release_validator':{'returncode':release.returncode,'stdout':release.stdout}}
    strict_json(output/'verification.json',verification)
    from fresh_confirmation_report import create_report
    create_report(output,protocol,per,trajectories,traces,aggregate,intervals,classes,args.frozen_run)
    status.update(status='complete',completed_at_utc=now(),reference_features=1600,actual_collection_forward_calls=json.loads((output/'collection_metadata.json').read_text())['actual_forward_calls_entered'],
        replay_encoder_calls=0,trajectories=5500,candidate_trace_rows=88000,per_video_policy_rows=2800)
    strict_json(status_path,status)
    files=sorted(p for p in output.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (output/'SHA256SUMS.txt').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(output).as_posix()}\n' for p in files),encoding='utf-8')
    print(f'Confirmation COMPLETE: {relative_path(output)};100 videos;1600 retained refs;5500 trajectories;88000 trace rows;2800 per-video rows.',flush=True)
    for r in intervals['comparisons']:
        print(f"B{r['budget']} {r['candidate']} vs {r['reference']}: {r['mean_paired_absolute_improvement']:+.6f} [{r['CI95_lower']:+.6f},{r['CI95_upper']:+.6f}] {r['interpretation']}",flush=True)
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-run',type=Path,default=Path('results/budget_aware_refresh_v2'))
    parser.add_argument('--output',type=Path,default=Path('results/fresh_refresh_confirmation_v1'))
    parser.add_argument('--device',choices=['cuda'],default='cuda'); parser.add_argument('--max-download-gib',type=float,default=12)
    parser.add_argument('--download-workers',type=int,default=8); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.max_download_gib<=0 or not 1<=args.download_workers<=16: parser.error('Positive download cap, 1..16 workers required')
    args.output.mkdir(parents=True,exist_ok=True)
    try: return execute(args,args.output)
    except Exception as exc:
        path=args.output/'status.json'; state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('status')!='complete':
            state.update(status='incomplete',blocked_at_utc=now(),blocker=f'{type(exc).__name__}: {exc}')
            progress_path=args.output/'collection_progress.json'
            if progress_path.exists():
                progress=json.loads(progress_path.read_text()); counts=Counter(v['entry']['label'] for v in progress['videos'])
                protocol=json.loads((args.frozen_run/'fresh_test_protocol.json').read_text())
                state['retained_videos']=sum(counts.values()); state['shortfall_per_class']={c:max(0,5-counts[c]) for c in protocol['classes']}
            strict_json(path,state)
        print(f'INCOMPLETE: {type(exc).__name__}: {exc}; preserve {relative_path(args.output)} and resume unchanged configuration.',file=sys.stderr)
        raise


if __name__=='__main__': raise SystemExit(main())
