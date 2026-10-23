"""Independent read-only empirical/identity verifier. No model, fit or generator.

Scalar schedule/branch calculations below do not call the runner's metric or
branch implementations. A blocked preflight can pass identity verification
without passing the empirical study or being relabeled COMPLETE.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def scalar_loss(z, schedule, label):
    cached = 0
    totals = [[] for _ in range(400)]
    for t in range(16):
        if t in schedule:
            cached = t
        for c in range(400):
            totals[c].append(float(z[cached,c])/16)
    pooled = [math.fsum(values) for values in totals]
    maximum = max(pooled)
    ce = maximum+math.log(math.fsum(math.exp(v-maximum) for v in pooled))-pooled[label]
    return ce, max(range(400),key=lambda c:pooled[c])


def independent_repair(t, action):
    chosen = [u for u in (0,4,8,12) if u < t]
    if action:
        chosen.append(t)
    for u in range(t+1,16):
        left = 4-len(chosen)
        anchors = sum(v>=u for v in (4,8,12))
        if left and ((u in (4,8,12) and anchors<=left) or (u not in (4,8,12) and anchors<left)):
            chosen.append(u)
    return tuple(chosen)


def empirical_source(row, feature_file, predictions, table_file=None):
    schedules = [(0,*tail) for tail in combinations(range(1,16),3)]
    if table_file is not None:
        with np.load(table_file,allow_pickle=False) as table:
            losses = table['loss'].tolist()
        if len(losses)!=455 or not np.isfinite(losses).all():
            raise ValueError('Invalid sealed schedule loss table')
    else:
        with np.load(feature_file,allow_pickle=False) as data:
            z = data['logits'].astype(float)
        losses = [scalar_loss(z,schedule,row['y'])[0] for schedule in schedules]
    periodic = losses[schedules.index((0,4,8,12))]
    result = {}
    branch = []
    for t in range(1,13):
        branch.append([losses[schedules.index(independent_repair(t,a))] for a in (0,1)])
    for view,pred in predictions.items():
        action = [1 if p>1e-12 else 0 if p< -1e-12 else int(t in (4,8,12)) for t,p in enumerate(pred,1)]
        result['V_'+view] = math.fsum(periodic-branch[t][a] for t,a in enumerate(action))/12
    result['primary'] = result['V_HGH']-result['V_HG']
    result['opportunity'] = math.fsum(periodic-min(pair) for pair in branch)/12
    return result,losses


def verify(out, write=False):
    out = Path(out)
    checks = {}
    errors = []
    for name,expected in read(out/'historical_fingerprint.json').items():
        if not (ROOT/name).is_file() or digest(ROOT/name)!=expected:
            errors.append('Historical file changed/missing: '+name)
    checks['historical_files'] = len(read(out/'historical_fingerprint.json'))
    config = read(out/'frozen_config.json')
    for name,expected in config['implementation'].items():
        if digest(ROOT/name)!=expected:
            errors.append('Configuration code identity changed: '+name)
    model = read(out/'model_identity.json')
    if digest(ROOT/model['path'])!=model['expected_sha256']:
        errors.append('Checkpoint hash differs')
    checks['checkpoint_authenticated'] = not any('Checkpoint' in e for e in errors)
    mapping_path = out/'class_mapping.json'
    if mapping_path.exists():
        mapping = read(mapping_path)['annotation_to_native']
        if len(mapping)!=400 or set(mapping.values())!=set(range(400)):
            errors.append('Class mapping not bijective')
        checks['native_classes'] = len(mapping)
    for identity in (ROOT/'data/source/native_head_value_confirmation_v1/metadata').glob('*.identity.json'):
        obj = read(identity)
        path = ROOT/obj['path']
        if digest(path)!=obj['sha256'] or path.stat().st_size!=obj['bytes']:
            errors.append('Official metadata identity differs: '+str(path.name))
    if (out/'archive_inventory.json').exists():
        archives = read(out/'archive_inventory.json')
        resources = read(out/'resource_plan.json')
        total = sum(r['content_length'] for r in archives if r['content_length'] is not None and not r['cached_path'])
        if total!=resources['conservative_new_complete_route_bytes']:
            errors.append('Archive full-transfer accounting differs')
        checks['archives_priced'] = len(archives)
        checks['complete_uncached_route_bytes'] = total
    status = read(out/'status.json')
    empirical = (out/'evaluation_exposure.json').exists()
    if not empirical:
        if status['scientific']!='NOT_EVALUATED' or status.get('primary_estimate') is not None:
            errors.append('Unexposed study mislabeled or contains an empirical estimate')
        if any((out/n).exists() for n in ('summary.json','eval_source_summaries.json','bootstrap_draws.npz')):
            errors.append('Unexpected outcome artifacts before recorded exposure')
        checks['new_eval_exposure_absent'] = True
    else:
        seal = read(out/'evaluation_seal.json')
        for name,expected in seal['sha256'].items():
            if digest(ROOT/name)!=expected:
                errors.append('Evaluation seal mismatch: '+name)
        if not (out/'eval_source_summaries.json').exists():
            errors.append('Empirical stage incomplete')
        else:
            # Use saved state predictions as evidence, never refit/reconstruct.
            states = out/'eval_state_predictions.npz'
            if not states.exists():
                errors.append('Independent verification needs saved genuine state predictions')
            else:
                rows = read(out/'test_manifest.json')
                summaries = read(out/'eval_source_summaries.json')
                if len({r['component'] for r in rows})!=len(rows):
                    errors.append('Shared source component, including across classes')
                with np.load(states,allow_pickle=False) as stored:
                    pp = {v:stored['advantage_'+v] for v in ('S','H','HG','HGH')}
                for i,row in enumerate(rows):
                    import hashlib
                    key = hashlib.sha256(f'native_head_value_confirmation_v1|20261023|feature|{row["filename"]}'.encode()).hexdigest()
                    fp = ROOT/'data/source/native_head_value_confirmation_v1/features'/(key+'.npz')
                    table = out/'schedule_tables'/(key+'.npz')
                    independently,_ = empirical_source(row,fp,{v:a[i*12:(i+1)*12] for v,a in pp.items()},table)
                    for key,value in independently.items():
                        if abs(value-summaries[i][key])>1e-10:
                            errors.append(f'Source scalar consequence mismatch: {i}/{key}')
                checks['independent_scalar_sources'] = len(rows)
                # Full scalar 455-schedule loss checks on preselected genuine
                # fitting smoke sources; EVAL consequences replay saved tables.
                smoke = read(out/'smoke_manifest.json')
                for row in smoke:
                    import hashlib
                    key = hashlib.sha256(f'native_head_value_confirmation_v1|20261023|feature|{row["filename"]}'.encode()).hexdigest()
                    fp = ROOT/'data/source/native_head_value_confirmation_v1/features'/(key+'.npz')
                    with np.load(fp,allow_pickle=False) as stored:
                        z = stored['logits'].astype(float)
                    with np.load(out/'fit_schedule_tables'/(key+'.npz'),allow_pickle=False) as stored:
                        losses = stored['loss']
                    scalar = [scalar_loss(z,s,row['y'])[0] for s in [(0,*tail) for tail in combinations(range(1,16),3)]]
                    if not np.allclose(losses,scalar,atol=1e-10,rtol=0):
                        errors.append('Fitting smoke scalar native loss mismatch')
                checks['scalar_smoke_sources_all455'] = len(smoke)
                summary = read(out/'summary.json')
                names = summary['columns']
                x = np.array([[r[n] for n in names] for r in summaries])
                groups = [np.array([i for i,r in enumerate(rows) if r['y']==c]) for c in range(400)]
                with np.load(out/'bootstrap_draws.npz',allow_pickle=False) as stored:
                    draws = stored['indices']
                boot = np.zeros((10000,len(names)))
                offset = 0
                vc = []
                for group in groups:
                    picked = draws[:,offset:offset+len(group)]
                    if not np.isin(picked,group).all():
                        errors.append('Bootstrap component crosses native class')
                    boot += x[picked].mean(1)/400
                    vc.append(np.var(x[group],axis=0,ddof=1)/(400**2*len(group)))
                    offset += len(group)
                theta = np.mean([x[g].mean(0) for g in groups],axis=0)
                variance = np.sum(vc,axis=0)
                denominator = np.sum([v*v/(len(g)-1) for v,g in zip(vc,groups)],axis=0)
                df = np.divide(variance**2,denominator,out=np.full_like(variance,np.inf),where=denominator>0)
                half = student_t.ppf(.975,df)*np.sqrt(variance)
                endpoints = np.quantile(boot,[.025,.975],axis=0,method='linear').T
                for actual,expected in [(theta,summary['inference']['mean']),(endpoints,summary['inference']['bootstrap95']),
                                        (np.array([theta-half,theta+half]).T,summary['inference']['analytic95'])]:
                    if not np.allclose(actual,expected,atol=1e-10,rtol=0):
                        errors.append('Independent aggregation/interval endpoint mismatch')
                checks['bootstrap_draws'] = 10000
                timing_path = out/'timing_trials.jsonl'
                if not timing_path.exists():
                    errors.append('Genuine complete-system timing absent')
                else:
                    trials = [json.loads(s) for s in timing_path.read_text().splitlines()]
                    if len(trials)!=60*7*4:
                        errors.append('Timing trial count mismatch')
                    for trial in trials:
                        expected_calls = 16 if trial['arm']=='Dense16' else 4
                        if trial['encoder_calls']!=expected_calls:
                            errors.append('Timing encoder budget mismatch')
                    paired = {(r['component'],r['repetition'],r['arm']):r for r in trials}
                    for row in read(out/'timing_manifest.json'):
                        for rep in range(-2,5):
                            base = paired[row['component'],rep,'PeriodicB4']['logits']
                            if any(paired[row['component'],rep,a]['logits']!=base for a in ('PeriodicB4_unused_HG','PeriodicB4_unused_HGH')):
                                errors.append('Timing unused-observation prediction equality fails')
    result = dict(passed=not errors,scope='identity/resource audit' if not empirical else 'independent genuine scalar replay',
        technical_status=status['technical'],scientific_status=status['scientific'],empirical_verification='NOT_RUN' if not empirical else 'REPLAYED',
        checks=checks,errors=errors,no_model_or_decode_calls=True,synthetic_data_used=False,
        completion_note='A passed blocked-state audit is not technical completion of the experiment or scientific evidence.')
    if write:
        from native_head_value_confirmation import save
        save(out/'independent_verification.json',result)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/native_head_value_confirmation_v1')
    p.add_argument('--write',action='store_true')
    args = p.parse_args()
    result = verify(args.output,args.write)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
