"""Experiment 25. Explicit output paths; CPU schedule-table analysis only.

Run `seal`, `ucf`, then the separate HMDB adapter, then `report`.
Historical runners and their module-global output paths are never invoked.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/continuation_value_gap_v1'
DATA = ROOT / 'data/source/continuation_value_gap_v1'
P = (0, 4, 8, 12)
POS = np.array([(0, *s) for s in combinations(range(1, 16), 3)], dtype=int)
LOOKUP = {tuple(p): i for i, p in enumerate(POS)}
TOL = 1e-10
SEED = 20261026


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def event(kind, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    if 'at_utc' in kw:
        kw['detail_at_utc'] = kw.pop('at_utc')
    with (OUT/'exposure_resource_ledger.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(at_utc=now(), kind=kind, **kw), ensure_ascii=False)+'\n')


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.17g')


def verify_hashes(records):
    for name, h in records.items():
        assert sha(ROOT/name) == h, f'Frozen identity changed: {name}'


def seal_check():
    s = read(OUT/'design_seal.json')
    amendment = OUT/'amendments/mechanical_001.json'
    amended = read(amendment)['changed_files'] if amendment.exists() else {}
    for name,h in s['sha256'].items():
        if name in amended:
            a=amended[name]
            assert a['original_sha256']==h and sha(ROOT/a['preserved_original'])==h
            assert sha(ROOT/name)==a['amended_sha256']
        else:
            assert sha(ROOT/name)==h, 'Frozen identity changed: '+name
    return s


def branch(t, action):
    prefix = [v for v in P if v < t]
    return np.array([i for i, p in enumerate(POS) if [v for v in p if v < t] == prefix
                     and (t in p) == bool(action)], dtype=int)


def repair(t, action):
    prefix = [v for v in P if v < t]
    b = 4-len(prefix)-action
    anchors = [v for v in P[1:] if v > t]
    anchors = anchors[-b:] if b else []
    if len(anchors) < b:
        anchors += [v for v in range(t+1, 16) if v not in anchors][:b-len(anchors)]
    return tuple(prefix+([t] if action else [])+sorted(anchors))


def sequential_repair(t, action):
    result = [v for v in P if v < t]+([t] if action else [])
    for u in range(t+1, 16):
        b = 4-len(result)
        nanchors = sum(v >= u for v in P[1:])
        # Earliest anchors are sacrificed; earliest nonanchors fill a deficit.
        refresh = (u in P and nanchors <= b) or (u not in P and nanchors < b)
        if refresh and b:
            result.append(u)
    return tuple(result)


def pool_weights():
    a = np.zeros((455, 16), dtype=np.float64)
    for j, p in enumerate(POS):
        for k, u in enumerate(p):
            a[j, u] = ((p[k+1] if k < 3 else 16)-u)/16
    return a


def weights(rows):
    classes = sorted({r['label'] for r in rows})
    fam = {c: {r['family'] for r in rows if r['label'] == c} for c in classes}
    counts = Counter((r['label'], r['family']) for r in rows)
    w = np.array([1/(len(classes)*len(fam[r['label']])*counts[r['label'], r['family']]) for r in rows])
    assert abs(w.sum()-1) < 1e-12
    return w


def cells(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r['label'], r['family']].append(r[key])
    return [dict(label=c, family=g, value=float(np.mean(v))) for (c, g), v in sorted(d.items())]


def cr2(rows, key='value'):
    """Specified class-indicator CR2 contrast, clustering GLOBAL family IDs."""
    cf = cells(rows, key)
    classes = sorted({r['label'] for r in cf}); C = len(classes)
    groups = sorted({r['family'] for r in cf}); ids = {g: i for i, g in enumerate(groups)}
    means = {c: np.mean([r['value'] for r in cf if r['label'] == c]) for c in classes}
    theta = float(np.mean(list(means.values())))
    singleton = [c for c in classes if sum(r['label'] == c for r in cf) < 2]
    if singleton:
        return dict(mean=theta, interval=None, df=None, status='UNIDENTIFIABLE_SINGLETON_CLASSES', singleton_classes=singleton)
    K = np.zeros((len(groups), len(groups))); u = np.zeros(len(groups))
    for c in classes:
        subset = [r for r in cf if r['label'] == c]; n = len(subset)
        q = 1/(C*n); a = 1/np.sqrt(1-1/n)
        ix = [ids[r['family']] for r in subset]
        K[np.ix_(ix, ix)] += (q*a)**2*(np.eye(n)-np.ones((n,n))/n)
        for r, j in zip(subset, ix):
            u[j] += q*a*(r['value']-means[c])
    denom = float(np.square(K).sum()); variance = float(u@u)
    assert denom > 0 and np.isfinite(K).all()
    df = float(np.trace(K)**2/denom)
    radius = float(student_t.ppf(.975, df)*np.sqrt(variance))
    ci = [theta-radius, theta+radius]
    outcome = ('material harm' if ci[1] < -.01 else 'material benefit' if ci[0] > .01 else
               'bounded small mean' if ci[0] >= -.01 and ci[1] <= .01 else 'unresolved')
    if variance == 0:
        outcome = 'unresolved'
    return dict(mean=theta, variance=variance, df=df, interval=ci, zero_variance=variance == 0,
                outcome=outcome, status='APPROXIMATE_CR2_SATTERTHWAITE')


def analyze_tables(dataset, meta, losses, predictions, dense_ce, dense_pred, candidate_logits=None):
    """Every state resets to ORIGINAL Periodic prefix. One intervention only."""
    dst = OUT/dataset; dst.mkdir(parents=True, exist_ok=True)
    w = weights(meta); state_rows = []; clip_rows = []
    A = pool_weights(); pid = LOOKUP[P]
    for i, m in enumerate(meta):
        L = losses[i]; pred = predictions[i]; y = m['y']; base = float(L[pid])
        proxy = None
        if candidate_logits is not None:
            z = candidate_logits[i]; zc = z-z.mean(1, keepdims=True)
            proxy = np.square(A@zc-zc.mean(0)).mean(1)
        sr = []
        for t in range(1, 13):
            ids = [branch(t, a) for a in (0,1)]
            star = np.array([L[j].min() for j in ids]); uni = np.array([L[j].mean() for j in ids])
            rp = [LOOKUP[repair(t, a)] for a in (0,1)]; qp = L[rp]
            ds, dp, du = float(star[0]-star[1]), float(qp[0]-qp[1]), float(uni[0]-uni[1])
            original = int(t in P)
            astar = original if abs(ds) <= TOL else int(ds > 0)
            ap = original if abs(dp) <= TOL else int(dp > 0)
            au = original if abs(du) <= TOL else int(du > 0)
            q = (4-sum(v < t for v in P))/(16-t)
            vu = (1-q)*uni[0]+q*uni[1]
            r = dict(clip=m['clip'], label=m['label'], family=m['family'], t=t,
                prefix=';'.join(str(v) for v in P if v < t), original_action=original,
                hold_count=len(ids[0]), refresh_count=len(ids[1]), hold_schedule=rp[0], refresh_schedule=rp[1],
                Q_star_H=float(star[0]), Q_star_R=float(star[1]), D_star=ds,
                Q_P_H=float(qp[0]), Q_P_R=float(qp[1]), D_P=dp,
                Q_U_H=float(uni[0]), Q_U_R=float(uni[1]), D_U=du, q=q, V_U=float(vu),
                a_star=astar, a_P=ap, a_U=au, numerical_tie=int(abs(ds) <= TOL),
                G_star=base-float(qp[astar]), G_matched=base-float(qp.min()),
                R_star=float(qp[astar]-qp.min()),
                G_star_U=float(vu-uni[astar]), G_matched_U=float(vu-uni.min()),
                R_star_U=float(uni[astar]-uni.min()),
                gap_H=float(qp[0]-star[0]), gap_R=float(qp[1]-star[1]),
                disagreement_P=int(astar != ap), disagreement_U=int(astar != au),
                material_P=int(abs(dp) >= .01), material_U=int(abs(du) >= .01),
                accuracy_star=int(pred[rp[astar]] == y), accuracy_matched=int(pred[rp[ap]] == y),
                correct_to_wrong=int(pred[pid] == y and pred[rp[astar]] != y),
                wrong_to_correct=int(pred[pid] != y and pred[rp[astar]] == y),
                tie_gain_low=base-float(qp.max()) if abs(ds) <= TOL else base-float(qp[astar]),
                tie_gain_high=base-float(qp.min()) if abs(ds) <= TOL else base-float(qp[astar]))
            if proxy is not None:
                union = np.sort(np.r_[ids[0], ids[1]])
                teacher_id = int(union[np.flatnonzero(proxy[union] <= proxy[union].min()+1e-12)[0]])
                at = int(t in POS[teacher_id]); r['a_Z_pool'] = at
                r['G_Z_pool'] = base-float(qp[at])
            assert np.min(qp-star) >= -TOL
            assert abs(dp-ds-(r['gap_H']-r['gap_R'])) < TOL
            assert abs(r['R_star']-r['G_matched']+r['G_star']) < TOL
            assert abs(r['R_star_U']-r['G_matched_U']+r['G_star_U']) < TOL
            sr.append(r)
        cr = dict(**m, weight=float(w[i]), Periodic_CE=base, Dense_CE=float(dense_ce[i]),
                  Task_oracle_CE=float(L.min()), Periodic_accuracy=int(pred[pid] == y),
                  Dense_accuracy=int(dense_pred[i] == y), Task_oracle_accuracy=int(pred[int(L.argmin())] == y))
        numeric = [k for k, v in sr[0].items() if isinstance(v, (int, float, np.number)) and k != 't']
        cr.update({k: float(np.mean([r[k] for r in sr])) for k in numeric})
        state_rows.extend(sr); clip_rows.append(cr)
    write_csv(dst/'per_state.csv.gz', state_rows); write_csv(dst/'per_clip.csv', clip_rows)
    summary = dict(dataset=dataset, clips=len(meta), states=len(state_rows), classes=len({r['label'] for r in meta}),
                   families=len({r['family'] for r in meta}), execution='COMPLETE',
                   primary=cr2(clip_rows, 'G_star'), weights='class -> within-class global-family -> clip -> 12 states',
                   Z_pool='AVAILABLE' if candidate_logits is not None else 'UNAVAILABLE_GENUINE_LOGITS_MISSING')
    keys = ['G_star','G_matched','R_star','G_star_U','G_matched_U','R_star_U','Periodic_CE','Dense_CE',
            'Task_oracle_CE','Periodic_accuracy','Dense_accuracy','Task_oracle_accuracy','accuracy_star',
            'accuracy_matched','correct_to_wrong','wrong_to_correct','numerical_tie','tie_gain_low','tie_gain_high']
    if candidate_logits is not None:
        keys += ['G_Z_pool']
    summary['estimates'] = {k: dict(mean=float(w@np.array([r[k] for r in clip_rows])),
                                    interval=cr2(clip_rows,k).get('interval')) for k in keys}
    sw = np.repeat(w/12, 12)
    for tail in ['P','U']:
        for material in [False, True]:
            mask = np.array([bool(r['material_'+tail]) if material else True for r in state_rows])
            dis = np.array([r['disagreement_'+tail] for r in state_rows])
            regret = np.array([r['R_star'+('_U' if tail == 'U' else '')] for r in state_rows])
            mass = float(sw[mask].sum()); dm = mask & (dis == 1)
            summary[f'disagreement_{tail}_'+('material' if material else 'all')] = dict(
                states=int(mask.sum()), denominator_weight=mass, disagree_states=int(dm.sum()),
                fraction=None if mass == 0 else float(sw[mask]@dis[mask]/mass),
                paired_regret_conditional=None if not dm.any() else float(sw[dm]@regret[dm]/sw[dm].sum()),
                paired_regret_whole=float(sw[dm]@regret[dm]))
    summary['clip_distributions'] = {}
    for k in ['G_star','R_star','G_star_U','R_star_U']:
        v = np.array([r[k] for r in clip_rows]); ix = np.argsort(v); cw = np.cumsum(w[ix])
        probs = [0,.01,.05,.25,.5,.75,.95,.99,1]
        summary['clip_distributions'][k] = dict(quantile_probabilities=probs,
            weighted_quantiles=[float(v[ix[min(np.searchsorted(cw, q),len(v)-1)]]) for q in probs],
            benefit_ge_001_mass=float(w@(v >= .01)), harm_le_minus001_mass=float(w@(v <= -.01)),
            benefit_clips=int((v >= .01).sum()), harm_clips=int((v <= -.01).sum()), denominator=len(v))
    family_table = []
    for k in keys:
        family_table.extend(dict(endpoint=k, **r) for r in cells(clip_rows, k))
    write_csv(dst/'class_family.csv', family_table)
    class_table = []
    for k in keys:
        cf = cells(clip_rows, k)
        for c in sorted({r['label'] for r in cf}):
            vals = [r['value'] for r in cf if r['label'] == c]
            class_table.append(dict(endpoint=k, label=c, families=len(vals), mean=float(np.mean(vals))))
    write_csv(dst/'per_class.csv', class_table)
    cf = cells(clip_rows, 'G_star'); logo = []
    for g in sorted({r['family'] for r in cf}):
        retained = [r for r in cf if r['family'] != g]
        if len({r['label'] for r in retained}) != summary['classes']:
            logo.append(dict(family=g, mean=None, reason='would remove a fixed class'))
        else:
            logo.append(dict(family=g, mean=cr2(retained)['mean']))
    write_csv(dst/'leave_one_family_out.csv', logo)
    family_mass = defaultdict(float)
    for r, wi in zip(meta,w): family_mass[r['family']] += float(wi)
    summary['family_concentration'] = dict(maximum_weight=max(family_mass.values()),
        sum_squared_weights=sum(v*v for v in family_mass.values()),
        effective_components=1/sum(v*v for v in family_mass.values()))
    save(dst/'estimates.json', summary)
    return summary


def seal():
    assert not (OUT/'design_seal.json').exists(), 'Existing seal: verify/resume; never overwrite'
    OUT.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for n in ['numpy','scipy','pandas','matplotlib','torch','torchvision','av','Pillow']:
        try: packages[n] = version(n)
        except PackageNotFoundError: packages[n] = None
    save(OUT/'environment.json', dict(executable=sys.executable, python=sys.version, platform=platform.platform(), packages=packages))
    prior = ROOT/'results/safe_schedule_density_v1'
    cohort = read(prior/'cohort_manifest.json'); assert len(cohort['clips']) == 400
    inventory = read(prior/'input_inventory.json')
    required = ['results/task_utility_observability_v1/states_execution_periodic.json.gz',
                'results/task_utility_observability_v1/prediction_inventory.json',
                'results/task_utility_observability_v1/offline_headroom.json',
                'results/safe_schedule_density_v1/cohort_manifest.json',
                'results/safe_schedule_density_v1/input_inventory.json',
                'results/safe_schedule_density_v1/design_seal.json',
                'results/safe_schedule_density_v1/full/per_video.csv',
                'results/safe_schedule_density_v1/full/completion.json']
    inv = {n:sha(ROOT/n) for n in required}
    oldseal = read(prior/'design_seal.json'); verify_hashes(oldseal['sha256'])
    oldcomplete = read(prior/'full/completion.json')
    reuse_name = 'results/ucf101_causal_development_v1/recognition_head_reuse.json'
    reuse = read(ROOT/reuse_name); assert sha(ROOT/reuse['path']) == reuse['sha256']
    hp = read(ROOT/reuse['path']); assert hp['classes'] == cohort['classes'] == reuse['classes']
    inv[reuse_name] = sha(ROOT/reuse_name); inv[reuse['path']] = reuse['sha256']
    types = Counter(); available = 0
    for m in cohort['clips']:
        assert m['official_role'] == 'official_train' and m['study_role'] == 'execution'
        n = f"results/safe_schedule_density_v1/full/clips/{m['clip']}.npz"
        assert sha(ROOT/n) == oldcomplete['sha256']['clips/'+m['clip']+'.npz']
        with np.load(ROOT/n, allow_pickle=False) as a:
            types[str(a['delta_K4_C'].dtype)] += 1
            assert a['delta_K4_C'].shape == (455,) and np.isfinite(a['delta_K4_C']).all()
        inv[n] = sha(ROOT/n)
        f = ROOT/m['feature']['path']
        if f.exists():
            assert sha(f) == m['feature']['sha256']; available += 1
            inv[m['feature']['path']] = sha(f)
    assert types == {'float64':400}, 'Prospective precision amendment needed before endpoints'
    # Authenticate execution identity against independent E21 manifest, without accessing official-test results.
    em = read(ROOT/'results/ucf101_causal_development_v1/evaluation_manifest.json')
    assert {v['clip'] for v in em['videos']} == {v['clip'] for v in cohort['clips']}
    for n in ['results/ucf101_causal_development_v1/evaluation_manifest.json',
              'results/ucf101_proxy_replication_v1/encoder_identity.json']:
        assert sha(ROOT/n) == inventory[n]['sha256']; inv[n] = sha(ROOT/n)
    save(OUT/'ucf_manifest.json', cohort)
    save(OUT/'input_manifest.json', inv)
    save(OUT/'numerical_provenance.json',dict(original_array_dtypes=dict(types),
        provenance='Experiment24 stable float64 CE of C pooled float64 affine logits; stored delta from Dense CE',
        reconstruction='float64 delta + authenticated full-precision per-clip Dense CE; no precision-restoring claim',
        tie_tolerance=TOL, verification_atol=1e-10, available_genuine_feature_clips=available,
        sensitivity_tolerances=[0,1e-12,1e-10,1e-8], full_ucf_interval='unidentifiable: BenchPress and JavelinThrow singleton classes'))
    # Blob inventory reads paths/hashes only; protected historical scientific content is not parsed.
    tracked = subprocess.check_output(['git','ls-files','-s','results'],cwd=ROOT,text=True)
    save(OUT/'historical_git_blobs.json', dict(lines=tracked.splitlines()))
    save(OUT/'checkout_review.json',dict(start_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        reference='aa147a1f98b1f82ca1bc3866bbceacf4633553af',
        tree_diff=subprocess.check_output(['git','diff','aa147a1','HEAD','--stat'],text=True),
        newer_commits=subprocess.check_output(['git','log','--oneline','aa147a1..HEAD'],text=True),
        explanation='Reference fetched; merge commit content tree identical; no reset. New isolated branch.',
        initial_unrelated=['presentation_outputs/','seminar_build/','seminar_output/']))
    files = ['continuation_value_gap.py','verify_continuation_value_gap.py','continuation_value_gap_hmdb.py',
             'results/continuation_value_gap_v1/protocol.md','results/continuation_value_gap_v1/frozen_config.json',
             'results/continuation_value_gap_v1/input_manifest.json','results/continuation_value_gap_v1/ucf_manifest.json',
             'results/continuation_value_gap_v1/numerical_provenance.json']
    save(OUT/'design_seal.json',dict(at_utc=now(),new_endpoints_computed=False,sha256={n:sha(ROOT/n) for n in files}))
    event('common_design_and_UCF_sealed', seal_sha256=sha(OUT/'design_seal.json'))


def run_ucf():
    seal_check(); verify_hashes(read(OUT/'input_manifest.json'))
    dst = OUT/'ucf'; assert not (dst/'estimates.json').exists(), 'Completed UCF: use independent verifier'
    started=time.perf_counter(); event('UCF_first_new_endpoint_computation', seal_sha256=sha(OUT/'design_seal.json'))
    cohort = read(OUT/'ucf_manifest.json'); classes=cohort['classes']
    prior = pd.read_csv(ROOT/'results/safe_schedule_density_v1/full/per_video.csv',float_precision='round_trip')
    prior = prior[(prior.K == 4)&(prior.pool == 'C')&(prior.epsilon == 0)].set_index('clip')
    hp = read(ROOT/read(ROOT/'results/ucf101_causal_development_v1/recognition_head_reuse.json')['path'])
    head = {k:np.asarray(v,dtype=np.float64) for k,v in hp['parameters'].items()}
    L=[]; pred=[]; dc=[]; dp=[]; logits=[]; meta=[]
    for m in cohort['clips']:
        r = prior.loc[m['clip']]
        with np.load(ROOT/f"results/safe_schedule_density_v1/full/clips/{m['clip']}.npz") as a:
            L.append(a['delta_K4_C']+float(r.dense_CE)); pred.append(a['prediction_K4_C'].copy())
        dc.append(float(r.dense_CE)); dp.append(int(r.dense_prediction))
        fp=ROOT/m['feature']['path']
        if fp.exists():
            with np.load(fp) as a: f=a['x3d_xs'].astype(np.float64)
            logits.append(((f-head['means'])/head['scales'])@head['weights'].T+head['bias'])
        meta.append(dict(clip=m['clip'],label=m['label'],family=m['label']+'/'+m['group'],y=classes.index(m['label'])))
    L=np.array(L); pred=np.array(pred); z=np.array(logits) if len(logits)==400 else None
    w=weights(meta); archived=dict(Periodic=.24742722945956003,Dense=.24432429514344095,task_oracle=.13548268541350716)
    actual=dict(Periodic=float(w@L[:,LOOKUP[P]]),Dense=float(w@dc),task_oracle=float(w@L.min(1)))
    for k in archived: assert abs(archived[k]-actual[k]) < TOL
    save(dst/'archive_reproduction.json',dict(archived=archived,reproduced=actual))
    dst.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dst/'outcome_table.npz',loss=L,prediction=pred,dense_ce=dc,dense_prediction=dp,
                        positions=POS, candidate_logits=z if z is not None else np.empty((0,)))
    save(dst/'population.json',meta)
    save(dst/'outcome_seal.json',dict(at_utc=now(),sha256={str(p.relative_to(ROOT).as_posix()):sha(p) for p in [dst/'outcome_table.npz',dst/'population.json']}))
    result=analyze_tables('ucf',meta,L,pred,dc,dp,z)
    sensitivity=[]
    for tol in [0,1e-12,1e-10,1e-8]:
        s=pd.read_csv(dst/'per_state.csv.gz',float_precision='round_trip')
        a=np.where(abs(s.D_star.to_numpy())<=tol,s.original_action.to_numpy(),(s.D_star.to_numpy()>0).astype(int))
        q=np.where(a,s.Q_P_R,s.Q_P_H); base=np.repeat(L[:,LOOKUP[P]],12)
        sensitivity.append(dict(tolerance=tol,tie_states=int((abs(s.D_star)<=tol).sum()),
                                G_star=float(np.repeat(w/12,12)@(base-q))))
    save(dst/'tie_sensitivity.json',sensitivity)
    event('UCF_complete',seconds=time.perf_counter()-started,new_encoder_calls=0,decodes=0,fits=0,
          primary=result['primary'])


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    datasets=[n for n in ['ucf','hmdb'] if (OUT/n/'estimates.json').exists()]
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,name in zip(axs,['ucf','hmdb']):
        if name not in datasets:
            ax.text(.5,.5,'HMDB: NOT EVALUATED\nSee input/resource gate',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
        s=pd.read_csv(OUT/name/'per_state.csv.gz'); wc=pd.read_csv(OUT/name/'per_clip.csv').weight.to_numpy()
        lim=max(.02,float(np.max(abs(s[['D_star','D_P']].to_numpy())))*1.03)
        # All states displayed; area proportional to hierarchical state weight.
        material=abs(s.D_P)>=.01; tie=abs(s.D_star)<=TOL
        for mask,color,label in [(~material & ~tie,'#7897ad','|D_P| < .01'),(material & ~tie,'#ac4a43','|D_P| >= .01'),(tie,'#d4a627','numerical tie')]:
            ax.scatter(s.D_star[mask],s.D_P[mask],s=(np.repeat(wc/12,12)[mask]*len(s)*10),alpha=.45,c=color,label=label,rasterized=True,linewidths=0)
        ax.plot([-lim,lim],[-lim,lim],color='.5',lw=.7);ax.axhline(0,color='.7',lw=.6);ax.axvline(0,color='.7',lw=.6)
        ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='D_star (nats)',ylabel='D_P (nats)',title=f'{name.upper()}: all {len(s):,} reset states')
        ax.legend(fontsize=8)
    fig.suptitle('Oracle advantage versus periodic-repair advantage\nFixed equal x/y scales within dataset; area = hierarchical state weight')
    for ext in ['pdf','png']:fig.savefig(OUT/f'figure_1_advantages.{ext}',dpi=240)
    plt.close(fig)
    keys=['G_star','G_matched','R_star','G_star_U','G_matched_U','R_star_U']
    fig,axs=plt.subplots(1,2,figsize=(10,4.4),layout='constrained',sharey=True)
    for ax,name in zip(axs,['ucf','hmdb']):
        if name not in datasets:
            ax.text(.5,.5,'HMDB: NOT EVALUATED',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
        e=read(OUT/name/'estimates.json');vals=[e['estimates'][k]['mean'] for k in keys]
        ax.bar(range(6),vals,color=['#a54b43','#477d98','#b28b3e']*2)
        for j,k in enumerate(keys):
            ci=e['estimates'][k]['interval']
            if ci is not None:ax.errorbar(j,vals[j],yerr=[[vals[j]-ci[0]],[ci[1]-vals[j]]],color='k',capsize=3)
        ax.axhline(0,color='.3',lw=.7);ax.set(xticks=range(6),xticklabels=['G*','G matched','R*','G* U','G matched U','R* U'],title=name.upper(),ylabel='Weighted CE gain / regret (nats)')
        ax.tick_params(axis='x',rotation=30)
        ax.text(.02,.98,'P tail: Periodic CE baseline\nU tail: V_U baseline\n'+('CR2 interval unidentifiable\n(2 singleton classes)' if name=='ucf' else 'Approximate 95% CR2 intervals'),va='top',transform=ax.transAxes,fontsize=8)
    fig.suptitle('Single privileged intervention: gain, matched opportunity, regret')
    for ext in ['pdf','png']:fig.savefig(OUT/f'figure_2_endpoints.{ext}',dpi=240)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for ax,name in zip(axs,['ucf','hmdb']):
        if name not in datasets:
            ax.text(.5,.5,'HMDB: NOT EVALUATED',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
        c=pd.read_csv(OUT/name/'per_clip.csv')
        for key,col in [('G_star','#a54b43'),('R_star','#477d98'),('G_star_U','#b28b3e'),('R_star_U','#587b50')]:
            order=np.argsort(c[key]);ax.step(c[key].to_numpy()[order],np.cumsum(c.weight.to_numpy()[order]),where='post',color=col,label=key)
        ax.axvline(0,color='.6',lw=.7);ax.axvline(-.01,color='.6',lw=.7,ls=':');ax.axvline(.01,color='.6',lw=.7,ls=':')
        ax.set(xlabel='Signed clip-mean gain / regret (nats)',ylabel='Hierarchically weighted clip CDF',title=name.upper(),ylim=(0,1));ax.legend(fontsize=8)
    fig.suptitle('Paired clip distributions, including adverse tails (no clipping)')
    for ext in ['pdf','png']:fig.savefig(OUT/f'figure_3_distributions.{ext}',dpi=240)
    plt.close(fig)


def report():
    figures()
    u=read(OUT/'ucf/estimates.json');h=read(OUT/'hmdb/estimates.json') if (OUT/'hmdb/estimates.json').exists() else None
    gate=read(OUT/'hmdb_gate.json') if (OUT/'hmdb_gate.json').exists() else {'status':'NOT_RUN'}
    lines=['# Experiment 25 — continuation value gap','',
        'This is an offline, label/future-informed diagnostic. UCF is exposed development; raw CE is never pooled across datasets. '+
        'Each clip averages 12 alternative SINGLE interventions at reset Periodic prefixes, not a rollout of oracle actions.','',
        '|Dataset|Execution|G_star (nats)|95% interval|Inferential outcome|','|---|---|---:|---|---|']
    for name,e in [('UCF',u),('HMDB',h)]:
        if e:
            p=e['primary'];lines += [f"|{name}|COMPLETE|{p['mean']:.9g}|{p['interval']}|{p.get('outcome','unresolved (invalid/unavailable uncertainty)')}|"]
        else:lines += [f"|{name}|{gate['status']}|N/A|N/A|NOT_EVALUATED|"]
    lines+=['','UCF retains all 400 clips, 101 fixed classes and 200 class-scoped groups. BenchPress and JavelinThrow each have one group: '+
        'the full-cohort CR2 interval is unidentifiable. No substitute narrow interval is reported.','',
        '|Endpoint (nats)|UCF|HMDB|','|---|---:|---:|']
    for key in ['G_star','G_matched','R_star','G_star_U','G_matched_U','R_star_U','G_Z_pool','Dense_CE','Periodic_CE','Task_oracle_CE']:
        a=u['estimates'].get(key,{}).get('mean'); b=h['estimates'].get(key,{}).get('mean') if h else None
        lines += [f'|{key}|{a}|{b}|']
    for name,e in [('UCF',u),('HMDB',h)]:
        if not e:continue
        lines += ['',f'## {name} paired consequences','',f"Accuracy: Dense {e['estimates']['Dense_accuracy']['mean']:.4%}, Periodic {e['estimates']['Periodic_accuracy']['mean']:.4%}, "+
                  f"oracle-selected single intervention {e['estimates']['accuracy_star']['mean']:.4%}; correct-to-wrong {e['estimates']['correct_to_wrong']['mean']:.4%}, "+
                  f"wrong-to-correct {e['estimates']['wrong_to_correct']['mean']:.4%}. Accuracy averages individual complete-schedule indicators; no intervention ensemble.",
                  '',f"All-state and material-margin disagreement denominators and paired regret: `{name.lower()}/estimates.json`. "+
                  'Weighted quantiles, material harm/benefit fractions, class estimates and leave-one-family-out results accompany the complete state/clip tables.']
        d=e['clip_distributions']['G_star'];lines += ['',f"Clip-mean gain >=.01: {d['benefit_ge_001_mass']:.4%} weighted mass ({d['benefit_clips']}/{d['denominator']} clips); "+
                  f"gain <=-.01: {d['harm_le_minus001_mass']:.4%} ({d['harm_clips']}/{d['denominator']}).",'',
                  f"Primary df: {e['primary']['df']}; component concentration: {e['family_concentration']}."]
    lines+=['','## Validation gate and resources','',json.dumps(gate,ensure_ascii=False,indent=2),'',
        'Actual collection events, failed starts/completions, decode/write-inclusive elapsed times, resume reuse and setup/check calls are in '+
        '`exposure_resource_ledger.jsonl` and `collection_resources.json`. CPU analysis/collection durations are diagnostic resource use, not deployed latency.',
        '', '## Interpretation and next decision','',
        'Positive matched gain and regret are nonnegative by construction; their existence is not a success. The sign and magnitude of G_star measure dependence on this exact continuation. '+
        'Small effects cannot establish that adaptive scheduling is impossible. Effects under uniform and periodic-repair tails have different baselines and only support continuation-specific claims.',
        '', 'The defensible contribution is an exact empirical diagnosis of how optimistic task-oracle action preferences transport to two fixed continuation rules in budgeted video inference. '+
        'It is not a new general sequential-learning principle, a learned policy, evidence that observations predict the advantage, or a speedup.',
        '', 'The old H/H+M controllers used Z_pool; no historical failure is attributed to training on D_star. Experiments 19/19.1/23A/23B remain unchanged. '+
        'HMDB (when evaluated) is previously task-outcome-unexposed under isolation of known conservative source components, conditional on a fixed head. '+
        'Unknown parent identities, reuploads and pretraining overlap remain uncertain. The adaptive frame-index HMDB recipe differs physically from UCF sampling; semantic domain is not the sole difference.',
        '', 'Next scientific action: inspect the signed primary estimate together with matched-tail opportunity and regret, then resolve the stated fresh-evaluation gate if HMDB did not run. '+
        'Do not fit another controller or tune a head using these outcomes. A new controller study would require a new prospective protocol and separate outcome-unexposed data.',
        '', '## Claim-to-evidence','', '|Claim|Evidence|Limit|','|---|---|---|',
        '|Exact oracle versus fixed-tail action consequences|per_state.csv.gz; sealed outcome_table.npz; independent_verification.json|Privileged labels/future outcomes; single intervention|',
        '|Distribution and classification consequences|per_clip.csv; estimates.json; figure_3_distributions|Fixed head and classes; CE is not error|',
        '|Continuation sensitivity|Q_U exact branch means, V_U, uniform endpoints|Different whole-tail baseline|',
        '|Provenance and execution|design/population/evaluation seals; ledgers; manifests|Unknown source and pretraining dependencies|',
        '', '## Related methodology','',
        'Ross and Bagnell, [AggreVaTe section 2.5](https://arxiv.org/html/1406.5979v1), discuss overly optimistic expert cost-to-go when a learner cannot execute the same recovery. '+
        'Our diagnostic tests a related assumption with exact finite schedule outcomes; it does not claim to discover that general issue.',
        '', 'Approximate small-sample cluster-robust uncertainty follows [Pustejovsky and Tipton](https://doi.org/10.1080/07350015.2016.1247004), '+
        'the [corrigendum](https://doi.org/10.1080/07350015.2023.2174123), and current official '+
        '[CR2](https://jepusto.github.io/clubSandwich/reference/vcovCR.html) / '+
        '[Satterthwaite contrast](https://jepusto.github.io/clubSandwich/reference/linear_contrast.html) documentation. '+
        'Intervals condition on fixed head/classes and assumed independent global components, omit training uncertainty, and have no exact finite-sample coverage guarantee. '+
        'Zero empirical variance cannot prove equivalence. Secondary intervals are descriptive and unadjusted.']
    (OUT/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['seal','ucf','report']);args=parser.parse_args()
    {'seal':seal,'ucf':run_ucf,'report':report}[args.stage]()
