"""Read-only identities, actual-call replay, empirical inference checks and status report."""
from __future__ import annotations
from collections import Counter
import numpy as np
from causal_observability_common import *


def smoke_source_check(smoke):
    """Permit only the documented extraction-checkpoint hardening after smoke."""
    import ast
    for name,digest in smoke['code_sha256'].items():
        current=ROOT/name
        if sha(current)==digest:continue
        archived=OUT/'pretest_corrections/runtime_at_successful_native_smoke.py'
        if name!='causal_observability_runtime.py' or not archived.exists() or sha(archived)!=digest:
            raise Blocker('Native smoke source identity unavailable')
        def execution_tree(path):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name not in ['collect_dense','features']]
            return ast.dump(tree,include_attributes=False)
        if execution_tree(current)!=execution_tree(archived):raise Blocker('Live execution changed since genuine native smoke')


def audit():
    """No output mutation, network, GPU initialization, fitting or timing."""
    result=dict(at_utc=now(),scope='Read-only audit; tests are distinct from completed empirical stages')
    if (OUT/'implementation_snapshot.json').exists():
        verify(read(OUT/'implementation_snapshot.json')['sha256']);result['implementation_identity']='verified pretest snapshot; not an evaluation seal'
    for name,(_,digest,_) in SOURCES.items():
        p=DATA/name
        if not p.exists():raise Blocker('Missing original acquisition bytes: '+name)
        if sha(p)!=digest:raise Blocker('Original archive identity mismatch: '+name)
    plan=read(OUT/'official_membership.json');verify(plan['annotation_sha256'])
    result['original_archives_and_annotations']='verified';result['classes']=len(plan['classes'])
    if (OUT/'eligibility_archive.json').exists():
        r=read(OUT/'eligibility_archive.json');verify({r['path']:r['sha256']})
        import gzip
        with gzip.open(ROOT/r['path'],'rb') as f:
            if hashlib.sha256(f.read()).hexdigest()!=r['decompressed_sha256']:raise Blocker('Eligibility journal archive mismatch')
    events=[]
    if (OUT/'encoder_events.jsonl').exists():events=[json.loads(line) for line in (OUT/'encoder_events.jsonl').read_text().splitlines()]
    counts=Counter((r['phase'],r['event']) for r in events)
    result['actual_encoder_events']=[dict(phase=k[0],event=k[1],count=n) for k,n in sorted(counts.items())]
    test_calls=sum(n for (phase,event),n in counts.items() if phase.startswith('test') and event=='start')
    result['official_test_encoder_starts']=test_calls
    if test_calls and not (OUT/'evaluation_exposure.json').exists():raise Blocker('Test model calls without exposure ledger')
    if (OUT/'GPU_smoke.json').exists():
        smoke=read(OUT/'GPU_smoke.json');verify({smoke['native_head_path']:smoke['native_head_sha256']})
        population=read(DATA/'eligible_input_population.json')['videos']
        for member in read(OUT/'GPU_smoke_membership.json')['clips']:
            row=next(v for v in population if v['label']==member['label'] and v['clip']==member['clip'])
            if row['official_flag']!=1 or row['sha256']!=member['sha256']:raise Blocker('Smoke source membership changed')
            verify({row['path']:row['sha256']})
        smoke_source_check(smoke)
        head=read(ROOT/smoke['native_head_path']);maximum=0
        for item in smoke['results']:
            verify({item['path']:item['sha256']});r=read(ROOT/item['path'])
            if r['encoder_starts']!=4 or r['encoder_completions']!=4 or r['positions'][0]!=0:raise Blocker('Real smoke budget failure')
            selected={int(k):np.asarray(v,dtype=float) for k,v in r['refreshed_features'].items()}
            last=0;cached=[]
            for t in range(16):
                if t in selected:last=t
                cached.append(selected[last])
            expected=np.asarray(head['weights']) @ np.asarray(cached,dtype=float).mean(0)+np.asarray(head['bias'])
            np.testing.assert_allclose(r['logits'],expected,atol=1e-10,rtol=1e-10)
            maximum=max(maximum,float(max(abs(np.asarray(r['logits'])-expected))))
        result['real_native400_integration']=dict(trials=len(smoke['results']),maximum_independent_logit_difference=maximum,
            new51_class_head_integration=smoke['new51_head_integration'])
    if (OUT/'evaluation_seal.json').exists():
        seal_check((OUT/'evaluation_exposure.json').exists());result['seal']='verified'
    else:result['seal']='NOT CREATED; population/training gate pending'
    if (OUT/'primary_interval.json').exists():
        import pandas as pd
        manifest=read(OUT/'eligible_manifest.json');entries=[v for v in manifest['videos'] if v['role']=='test'];classes=manifest['classes']
        table=pd.read_csv(OUT/'per_source_metrics.csv',float_precision='round_trip')
        pivot=table.pivot(index='key',columns='method',values='CE')
        keys=[v['label']+'/'+v['clip'] for v in entries];delta=(pivot.loc[keys,'periodic']-pivot.loc[keys,'history_motion']).to_numpy()
        meta=read(OUT/'bootstrap_clusters.json');families=meta['families'];idx=np.load(OUT/'bootstrap_cluster_indices.npy',allow_pickle=False)
        multiplicities=np.array([np.bincount(row,minlength=len(families)) for row in idx])
        fi=np.array([families.index(v['source_family']) for v in entries]);draws=np.zeros(len(idx));means=[]
        for c in classes:
            chosen=np.array([v['label']==c for v in entries]);w=multiplicities[:,fi[chosen]]
            draws+=(w @ delta[chosen])/w.sum(1)/len(classes);means.append(float(delta[chosen].mean()))
        ci=np.percentile(draws,[2.5,97.5]);primary=read(OUT/'primary_interval.json')
        np.testing.assert_allclose([np.mean(means),*ci],[primary['mean'],primary['CI95_lower'],primary['CI95_upper']],atol=1e-12,rtol=0)
        result['primary_interval']='Independently reproduced from executed per-source table and cluster multiplicities'
    review=read(OUT/'repository_review.json');original={v.split('\t',1)[1]:v.split('\t',1)[0] for v in review['old_scientific_blobs']}
    current={v.split('\t',1)[1]:v.split('\t',1)[0] for v in git('ls-files','-s','results').splitlines()}
    for path,blob in original.items():
        if current.get(path)!=blob:raise Blocker('Historical scientific blob changed: '+path)
    if set(git('diff','--name-only','--','results').splitlines())&set(original):raise Blocker('Historical scientific working-tree changes')
    result['historical_results']='All original tracked scientific blobs preserved'
    if (OUT/'SHA256SUMS.json').exists():verify(read(OUT/'SHA256SUMS.json'));result['artifact_hashes']='verified'
    result['status']=read(OUT/'status.json');return result


def input_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=read(OUT/'population_design.json');classes=p['classes']
    fig,axes=plt.subplots(2,1,figsize=(14,7),layout='constrained',sharex=True)
    axes[0].bar(range(len(classes)),[p['isolated_training_per_class'][c] for c in classes],color='#297b8a')
    axes[0].axhline(70,color='#b24b35',label='45+15+10 required',ls='--');axes[0].legend();axes[0].set(ylabel='Eligible isolated train clips',title='Input-only population feasibility; no HMDB recognition outcomes')
    axes[1].bar(range(len(classes)),[p['eligible_official_test_per_class'][c] for c in classes],color='#5753a3');axes[1].axhline(30,color='gray',ls='--');axes[1].set(ylabel='Eligible official test clips')
    axes[1].set_xticks(range(len(classes)),classes,rotation=90,fontsize=8)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'input_population.png',dpi=180);fig.savefig(folder/'input_population.pdf');plt.close(fig)


def report(blocker=None):
    status=read(OUT/'status.json');p=read(OUT/'population_design.json') if (OUT/'population_design.json').exists() else {}
    tests=read(OUT/'unit_tests.json') if (OUT/'unit_tests.json').exists() else {}
    smoke=read(OUT/'GPU_smoke.json') if (OUT/'GPU_smoke.json').exists() else {}
    primary=read(OUT/'primary_interval.json') if (OUT/'primary_interval.json').exists() else None
    rows=['# Experiment 19: Causal Observability Gap', '',f'Execution state: **{status["status"]}**, stage `{status["current_stage"]}`.','']
    if blocker:rows+=['Population/resource blocker: '+blocker,'']
    if primary:rows+=[f'Primary Delta_CE = {primary["mean"]:+.6f} nats, 95% paired source-family CI [{primary["CI95_lower"]:+.6f},{primary["CI95_upper"]:+.6f}]. Positive favors the executed H+M policy.','']
    else:rows+=['**Primary estimate and interval: not available. No HMDB test recognition result, trained-controller benefit, observability-gap estimate or final latency comparison is claimed.**','']
    if p:
        rows += [f'Original archive/split metadata: {p["official_clips"]} clips, 51 classes. Input screening: {p["technically_eligible"]} eligible and {p["technical_failures"]} failures under the historical 1.57-second trailing-snippet rule. All failures are inadequate support for16 complete distinct-frame snippets.',
            f'After official-test-family priority and duplicate joins: {sum(p["isolated_training_per_class"].values())} eligible isolated official training clips and {sum(p["eligible_official_test_per_class"].values())} eligible official test clips. {len(p["deficient_classes"])} classes fail the exact70-training requirement. Classes meeting only the clip-count prerequisite (family-role packing not yet proven): '+', '.join(c for c,n in p['isolated_training_per_class'].items() if n==70)+'.',
            'Classes with no eligible official test clip under this temporal rule: '+', '.join(c for c,n in p['eligible_official_test_per_class'].items() if n==0)+'.','',
            'The official lists contain70 training clips/class. Exact45/15/10 leaves no spare clips for source-family quarantine or technical exclusions. Input metadata alone found137 title-based families crossing official train/test, involving813 training clips across42 classes. These source conflicts remain even if temporal eligibility is later redesigned. No flag0/test replacements or outcome-based substitutions were used.','']
    rows += ['## What ran versus what remains','',f'Synthetic safety tests: return code {tests.get("returncode","not run")}. The test log states the actual count. These are unit fixtures, not scientific observations.',
        'Real GPU integration: '+smoke.get('status','not completed')+'.',
        f'Completed native400 smoke calls retained: {smoke.get("completed_smoke_calls",0)}. The preserved earlier failed verification attempt has four additional genuine start/completion events. The failure was float32 accumulation in an independent reference; correction uses float64 without relaxing tolerances.',
        'The real native400 checks validate decode/PTS/chronology,16 cached outputs,four calls per execution,current gray observations and pooled head replay. They are not a newly fitted51-class recognition head, trained H/H+M policy evaluation or the final timing study.','',
        'Pending when population-blocked: new51-class head fitting; controller training/development selection; new51-class GPU integration; pretest seal and first-test exposure; all official test rollouts; privileged dense/oracle diagnostics; paired task intervals; final interleaved timing; four task-quality/cost figures. Those figures are intentionally absent rather than filled with synthetic data. The input-only population figure is separate.','',
        '## Claim / evidence','', '|Claim|Evidence and limit|','|---|---|',
        '|Fixed population is feasible|Not established; population_design.json records the failed exact quotas and empty test support for catch|',
        '|Causal runtime respects interface/budget|Synthetic contracts plus actual training-only four-forward native400 executions; no trained-policy quality claim|',
        '|Causal learned policy beats periodic|Unmeasured; no primary interval exists before actual test execution|',
        '|Offline advantage is deployable|Unmeasured; past UCF Z_pool used all16 features; its 3.945x figure is dense/periodic latency, not deployed Z_pool|',
        '|Source independence|Conservative title-family/exact/near-duplicate constraints; external parent IDs and pretraining mappings remain missing|','',
        '## Research decision','',
        'Preserve this blocked fixed-design attempt. An all51-class empirical study needs an explicitly approved input-only redesign of role counts and short-clip temporal support, followed by a new frozen protocol before recognition outcomes. Restricting to three eligible classes would be a different, strongly selected task and would not answer the requested51-class question. Do not bypass source isolation, manufacture outcomes or optimize for significance.','',
        'The implementation starts from reviewed merge017fe79; the actual initial checkout was a50d130 with the same tree. No historical scientific outputs were reset or overwritten. Any future post-exposure correction must archive and disclose original evidence.','',
        '```powershell',r'& .\.venv_temporal\Scripts\python.exe .\run_causal_observability.py --resume',r'& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability.py --audit-only','```','']
    if primary:
        finding=('The paired primary interval supports lower CE for the executed H+M controller.' if primary['CI95_lower']>0
            else 'The paired primary interval favors periodic; the learned controller did not establish improvement.' if primary['CI95_upper']<0
            else 'The primary interval crosses zero: inconclusive, not equivalence. No positive decision utility is established.')
        rows=[finding if s.startswith('Preserve this blocked fixed-design attempt.') else s for s in rows]
        rows=['|Causal learned policy beats periodic|'+finding+'|' if s.startswith('|Causal learned policy beats periodic|') else s for s in rows]
        rows=['Completed empirical measurements and remaining review are identified by status.json; secondary intervals are marginal/unadjusted. No post hoc primary-method change or test extension is authorized.' if s.startswith('Pending when population-blocked:') else s for s in rows]
    (OUT/'findings.md').write_text('\n'.join(rows),encoding='utf-8')
