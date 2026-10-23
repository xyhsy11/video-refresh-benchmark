"""Genuine-input stages for Experiment 27. All gates precede EVAL inference.

Acquisition is deliberately separate: provide official original RGB at
data/source/native_head_value_confirmation_v1/videos/{val,test}/<official name>.
Missing higher-ranked input cannot be silently skipped to fill a quota.
"""
from __future__ import annotations

from collections import defaultdict
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from native_head_value_confirmation import *
from run_native_head_value_confirmation import ROOT, OUT, CACHE, ASSETS, CODE, CAPS, event, rel, now


def identities():
    names = CODE+['results/'+STUDY+'/'+n for n in
        ('protocol.md','frozen_config.json','class_mapping.json','model_identity.json','historical_source_evidence.json')]
    return {n:sha(ROOT/n) for n in names}


def check_seal():
    seal = read(OUT/'evaluation_seal.json')
    for name, digest in seal['sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Sealed identity changed: '+name)
    return seal


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.writing')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def counters():
    events = [json.loads(line) for line in (OUT/'resource_exposure_ledger.jsonl').read_text().splitlines()]
    return {k:sum(e.get(k,0) for e in events) for k in
            ('encoder_starts','encoder_completions','decodes','decoded_frames','probe_fits','collection_seconds','cpu_seconds','timing_seconds')}


def guard(stage):
    current = counters()
    key = stage+'_seconds'
    if current.get(key,0) >= CAPS[key]:
        raise RuntimeError('BLOCKED_RESOURCE: cumulative '+stage+' cap')
    saved = sum(p.stat().st_size for base in (CACHE,OUT) for p in base.rglob('*') if p.is_file())
    if saved > CAPS['retained_bytes']:
        raise RuntimeError('BLOCKED_RESOURCE: additional retained-byte cap')


def decode(encoded, pixels=False, needed=None):
    import av
    from research_utils import configure_decoder
    event('decode_start', decodes=1)
    pts, frames = [], {}
    try:
        with av.open(io.BytesIO(encoded)) as container:
            stream = container.streams.video[0]
            configure_decoder(stream)
            for i, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise ValueError('Missing genuine PTS')
                pts.append(dict(pts=int(frame.pts), time_base=str(frame.time_base), seconds=float(frame.pts*frame.time_base)))
                if pixels and (needed is None or i in needed):
                    frames[i] = frame if pixels == 'raw' else (frame.to_ndarray(format='rgb24'),
                                 frame.reformat(width=32,height=32,format='gray').to_ndarray())
        seconds = np.array([p['seconds'] for p in pts])
        if not len(pts) or not np.isfinite(seconds).all() or not (np.diff(seconds)>0).all():
            raise ValueError('Genuine finite strictly increasing presentation PTS required')
        event('decode_complete', decoded_frames=len(pts))
        return pts, frames
    except Exception as exc:
        event('decode_failure', error=str(exc), decoded_frames=len(pts))
        raise


def local_path(name, split):
    locations = [CACHE/'videos'/split/name]
    if split == 'val':
        locations += [ROOT/'data/source/cache_drift_v2/videos'/name, ASSETS/'videos'/name]
    return next((p for p in locations if p.is_file()), None)


def screen_population(split, m):
    from temporal_refresh_common import snippet_grid
    candidates = read(CACHE/'candidate_manifest.json')
    grouped = defaultdict(list)
    for row in candidates:
        if row['split'] == split:
            grouped[row['component']].append(row)
    # Canonical clip is fixed across all labels in each component BEFORE quotas.
    canonical = []
    failures = []
    for component, members in sorted(grouped.items(), key=lambda kv: rank('component',kv[0])):
        chosen = None
        for member in sorted(members, key=lambda r:r['canonical_rank']):
            path = local_path(member['filename'], split)
            if path is None:
                # Unavailable bytes have unknown technical eligibility, so no
                # lower-ranked clip/component can replace them silently.
                failures.append(dict(component=component, filename=member['filename'], reason='missing_original_RGB'))
                break
            guard('collection')
            start = time.perf_counter()
            try:
                pts, _ = decode(path.read_bytes())
                grid = snippet_grid([p['seconds'] for p in pts])
                chosen = dict(member, path=rel(path), encoded_sha256=sha(path),
                              encoded_bytes=path.stat().st_size, pts=pts, grid=grid)
                break
            except (ValueError, RuntimeError) as exc:
                failures.append(dict(component=component, filename=member['filename'], reason='technical_ineligible', detail=str(exc)))
            finally:
                event('screen_elapsed', collection_seconds=time.perf_counter()-start)
        if chosen:
            canonical.append(chosen)
    save(OUT/f'{split}_screening_exclusions.json', failures)
    if any(r['reason']=='missing_original_RGB' for r in failures):
        raise RuntimeError(f'BLOCKED_INPUT: unresolved canonical {split} inputs; see screening exclusions')
    selected = []
    for y in range(400):
        cc = sorted([r for r in canonical if r['y']==y], key=lambda r:rank('component',r['component']))
        if len(cc) < m:
            raise RuntimeError(f'BLOCKED_PRECISION_OR_POPULATION: {split} native class {y} needs {m}, has {len(cc)}')
        selected.extend(cc[:m])
    validate_components(selected, m)
    save(OUT/f'{split}_manifest.json', selected)
    return selected


class NativeEncoder:
    def __init__(self):
        import torch
        import pytorchvideo
        from pytorchvideo.models.hub.x3d import x3d_xs
        from benchmark import configure_numerics
        from torchvision.transforms._functional_video import center_crop, normalize
        import importlib.util
        if not torch.cuda.is_available():
            raise RuntimeError('BLOCKED_RESOURCE: genuine CUDA device unavailable')
        configure_numerics(SEED)
        torch.set_num_threads(1)
        source = ASSETS/f'pytorchvideo-{REVISION}'
        installed = Path(pytorchvideo.__file__).parent
        for name in ('models/hub/x3d.py','models/x3d.py','models/head.py','transforms/functional.py'):
            if sha(installed/name) != sha(source/'pytorchvideo'/name):
                raise RuntimeError('BLOCKED_INPUT: installed PyTorchVideo differs from pinned revision')
        if sha(ASSETS/'X3D_XS.pyth') != CHECKPOINT_SHA:
            raise RuntimeError('Checkpoint identity mismatch')
        self.torch = torch
        self.model = x3d_xs(pretrained=False)
        self.model.load_state_dict(torch.load(ASSETS/'X3D_XS.pyth',map_location='cpu',weights_only=False)['model_state'], strict=True)
        self.model.eval().cuda()
        if self.model.blocks[-1].activation is not None:
            raise RuntimeError('Native output is not pre-softmax logits')
        self.capture = {}
        def capture(module, args, result):
            self.capture = dict(input=args[0], projected=result)
        self.model.blocks[-1].proj.register_forward_hook(capture)
        spec = importlib.util.spec_from_file_location('native27_pinned_functional', source/'pytorchvideo/transforms/functional.py')
        self.functional = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.functional)
        self.center_crop, self.normalize = center_crop, normalize

    def forward(self, rgbs, phase):
        torch = self.torch
        x = torch.from_numpy(np.stack(rgbs)).permute(3,0,1,2).float()/255
        x = self.normalize(x,[.45]*3,[.225]*3)
        x = self.center_crop(self.functional.short_side_scale(x,182),[182,182]).unsqueeze(0).cuda()
        event('encoder_start', phase=phase, encoder_starts=1)
        with torch.inference_mode():
            native = self.model(x)
            f = self.capture['input'].mean(tuple(range(1,self.capture['input'].ndim-1)))
            projected = self.capture['projected'].mean((1,2,3))
            affine = torch.nn.functional.linear(f,self.model.blocks[-1].proj.weight,self.model.blocks[-1].proj.bias)
            torch.testing.assert_close(native,projected,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(native,affine,atol=2e-5,rtol=2e-5)
            torch.cuda.synchronize()
            result = f[0].cpu().numpy(), native[0].cpu().numpy()
        event('encoder_complete', phase=phase, encoder_completions=1,
              native_affine_max_error=float((native-affine).abs().max().item()))
        return result


def feature_path(row):
    return CACHE/'features'/(rank('feature',row['filename'])+'.npz')


def collect(rows, encoder, phase):
    result = []
    for row in rows:
        guard('collection')
        if phase == 'EVAL':
            seal = check_seal()
            if not (OUT/'evaluation_exposure.json').exists():
                save(OUT/'evaluation_exposure.json', dict(at_utc=now(), seal_sha256=sha(OUT/'evaluation_seal.json'),
                    status='EXPOSED_FROM_FIRST_MODEL_ATTEMPT', completed=False))
        target = feature_path(row)
        identity = target.with_suffix('.json')
        expected = dict(encoded_sha256=row['encoded_sha256'], grid_sha256=rank('grid',json.dumps(row['grid'],sort_keys=True)),
                        checkpoint_sha256=CHECKPOINT_SHA, phase=phase, implementation=identities())
        if target.exists() or identity.exists():
            if not target.exists() or not identity.exists():
                raise RuntimeError('Partial output needs audit, never silently reused')
            old = read(identity)
            if old['input'] != expected or old['array_sha256'] != sha(target):
                raise RuntimeError('Atomic source resume identity mismatch')
        else:
            start = time.perf_counter()
            path = ROOT/row['path']
            if sha(path) != row['encoded_sha256']:
                raise RuntimeError('Sealed video bytes changed')
            needed = set(sum(row['grid']['frame_indices'],[]))
            pts, frames = decode(path.read_bytes(),pixels=True,needed=needed)
            if pts != row['pts']:
                raise RuntimeError('Sealed PTS identity changed')
            f, z, thumbnails = [], [], []
            for indices in row['grid']['frame_indices']:
                ff, zz = encoder.forward([frames[i][0] for i in indices],phase)
                f.append(ff);z.append(zz);thumbnails.append(frames[indices[-1]][1])
            atomic_npz(target,features=np.array(f),logits=np.array(z),thumbnails=np.array(thumbnails))
            save(identity,dict(input=expected,array_sha256=sha(target)))
            event('source_collection_complete',phase=phase,source=row['component'],collection_seconds=time.perf_counter()-start)
        with np.load(target,allow_pickle=False) as saved:
            f,z,thumbs = saved['features'],saved['logits'],saved['thumbnails']
        target_values = consequences(z,f,row['y'])
        table_dir = 'schedule_tables' if phase=='EVAL' else 'fit_schedule_tables'
        atomic_npz(OUT/table_dir/(rank('feature',row['filename'])+'.npz'),
                   loss=target_values['loss'],prediction=target_values['prediction'])
        result.append(dict(row=row,x=trajectory(thumbs,f,z),targets=target_values))
    return result


def fit_models(data, train, test, predictions, fold):
    from executable_value_observability import ridge_fit, score
    x = np.concatenate([d['x'] for d in data])
    for target in ('advantage','drift'):
        y = np.concatenate([d['targets'][target] for d in data])
        train_rows = np.concatenate([np.arange(i*12,(i+1)*12) for i in train])
        test_rows = np.concatenate([np.arange(i*12,(i+1)*12) for i in test]) if len(test) else np.array([],int)
        # Exactly equal class/source/state weights, recomputed within train fold.
        counts = Counter(data[i]['row']['y'] for i in train)
        w = np.concatenate([np.full(12,1/(400*counts[data[i]['row']['y']]*12)) for i in train])
        for view, size in VIEWS.items():
            guard('cpu')
            started = time.perf_counter()
            model = ridge_fit(x[train_rows,:size],y[train_rows],w)
            pred_train = score(model,x[train_rows,:size])
            mean = float(np.average(y[train_rows],weights=w))
            report = dict(model=model, target=target, view=view, fold=fold,
                train_mae=float(np.average(abs(pred_train-y[train_rows]),weights=w)),
                constant_train_mae=float(np.average(abs(mean-y[train_rows]),weights=w)), constant=mean)
            if len(test):
                pred = score(model,x[test_rows,:size])
                predictions[target][view][test_rows] = pred
                report.update(heldout_mae=float(np.mean(abs(pred-y[test_rows]))),constant_heldout_mae=float(np.mean(abs(mean-y[test_rows]))))
            save(OUT/'probes'/f'{fold}_{target}_{view}.json',report)
            event('probe_fit_complete',probe_fits=1,cpu_seconds=time.perf_counter()-started,target=target,view=view,fold=fold)


def source_values(data, predictions):
    from executable_value_observability import actions
    values = []
    for i, d in enumerate(data):
        target = d['targets']
        record = {k:d['row'][k] for k in ('component','youtube_id','filename','y','label')}
        record.update(periodic_loss=float(target['periodic_loss']),dense_loss=float(target['dense_loss']),
                      dense_correct=float(target['dense_correct']),periodic_correct=float(target['periodic_correct']),
                      full_oracle_loss=float(target['loss'].min()),
                      opportunity=float(np.mean(target['periodic_loss']-target['branch_loss'].min(1))))
        for view in VIEWS:
            pred = predictions['advantage'][view][i*12:(i+1)*12]
            act = actions(pred,np.arange(1,13))
            record['V_'+view] = float(np.mean(target['periodic_loss']-target['branch_loss'][np.arange(12),act]))
            record['regret_'+view] = record['opportunity']-record['V_'+view]
            ids = target['branches'][np.arange(12),act]
            record['top1_change_'+view] = float(np.mean(target['prediction'][ids]==d['row']['y'])-target['periodic_correct'])
            for name in ('advantage','drift'):
                yy = target[name]
                pp = predictions[name][view][i*12:(i+1)*12]
                record[name+'_mae_'+view] = float(np.mean(abs(pp-yy)))
        record['primary'] = record['V_HGH']-record['V_HG']
        record['H_minus_S'] = record['V_H']-record['V_S']
        record['HG_minus_H'] = record['V_HG']-record['V_H']
        record['HGH_minus_H'] = record['V_HGH']-record['V_H']
        record['drift_contrast'] = record['drift_mae_HG']-record['drift_mae_HGH']
        values.append(record)
    return values


def plan():
    status = read(OUT/'status.json')
    # This document is deliberately absent while mixed legacy reservation
    # evidence remains unresolved. Never turn unknown reservation into permission.
    certificate = OUT/'source_isolation_certificate.json'
    if not certificate.exists():
        print(json.dumps(dict(technical=status['technical'],scientific='NOT_EVALUATED',
            message='Source-role certificate missing; no extraction/fitting/EVAL calls. See findings.md.')))
        return 2
    certificate = read(certificate)
    if certificate.get('historical_source_evidence_sha256') != sha(OUT/'historical_source_evidence.json') or not certificate.get('all_legacy_reservations_resolved'):
        raise RuntimeError('BLOCKED_INPUT: source-role certificate is incomplete/stale')
    # A local full population can resolve the network gate without downloading.
    candidates = read(CACHE/'candidate_manifest.json')
    missing = [r['filename'] for r in candidates if local_path(r['filename'],r['split']) is None]
    if missing:
        print(json.dumps(dict(technical=status['technical'],scientific='NOT_EVALUATED',
            unresolved_original_RGB=len(missing),message='Preflight gate retained; no extraction/fitting/EVAL calls. See resource_plan.json.')))
        return 2
    fit = screen_population('val',8)
    encoder = NativeEncoder()
    smoke = sorted(fit,key=lambda r:rank('smoke',r['component']))[:8]
    save(OUT/'smoke_manifest.json',smoke)
    started = time.perf_counter()
    smoke_data = collect(smoke,encoder,'PROBE_FIT')
    elapsed = time.perf_counter()-started
    from verify_native_head_value_confirmation import scalar_loss
    from itertools import combinations
    maximum_error = 0.
    for d in smoke_data:
        with np.load(feature_path(d['row']),allow_pickle=False) as saved:
            zz = saved['logits'].astype(float)
        scalar = [scalar_loss(zz,(0,*tail),d['row']['y'])[0] for tail in combinations(range(1,16),3)]
        error = float(np.max(abs(np.asarray(scalar)-d['targets']['loss'])))
        maximum_error = max(maximum_error,error)
        if error>1e-10:
            raise RuntimeError('Fitting-only all455 independent scalar verification failed')
    save(OUT/'smoke_independent_verification.json',dict(sources=8,schedules_per_source=455,max_ce_error=maximum_error,passed=True))
    projection = elapsed/8*(3200+8000)
    save(OUT/'throughput_projection.json',dict(real_smoke_clips=8,elapsed_seconds=elapsed,
        conservative_collection_seconds=projection,source='real native400 smoke, includes fresh decode/encoding/writes'))
    if projection > CAPS['collection_seconds']:
        raise RuntimeError('BLOCKED_RESOURCE: genuine throughput projection exceeds collection cap')
    data = collect(fit,encoder,'PROBE_FIT')
    fs = folds(fit)
    predictions = {target:{v:np.full(len(fit)*12,np.nan) for v in VIEWS} for target in ('advantage','drift')}
    for f in range(4):
        fit_models(data,np.flatnonzero(fs!=f),np.flatnonzero(fs==f),predictions,f)
    summaries = source_values(data,predictions)
    save(OUT/'fit_oof_source_summaries.json',summaries)
    atomic_npz(OUT/'fit_oof_state_predictions.npz',**{target+'_'+view:value for target,views in predictions.items() for view,value in views.items()})
    save(OUT/'fit_folds.json',fs.tolist())
    pp = precision_plan(fit,[r['primary'] for r in summaries])
    save(OUT/'precision_plan.json',pp)
    if pp['selected_m'] is None:
        raise RuntimeError('BLOCKED_PRECISION_OR_POPULATION: projected half-width needs m>20')
    evaluation = screen_population('test',pp['selected_m'])
    if {r['component'] for r in fit} & {r['component'] for r in evaluation}:
        raise RuntimeError('Global component roles collide')
    fit_models(data,np.arange(len(fit)),[],predictions,'final')
    timing_classes = sorted(range(400),key=lambda c:rank('timing-class',c))[:60]
    timing = [min([r for r in evaluation if r['y']==c],key=lambda r:rank('timing-source',r['component'])) for c in timing_classes]
    save(OUT/'timing_manifest.json',timing)
    frozen = identities()
    for path in [OUT/'val_manifest.json',OUT/'test_manifest.json',OUT/'precision_plan.json',OUT/'timing_manifest.json',*sorted((OUT/'probes').glob('final_*.json'))]:
        frozen[rel(path)] = sha(path)
    # Include original bytes and immutable per-source fitting output identities.
    for row in fit+evaluation:
        frozen[row['path']] = row['encoded_sha256']
    save(OUT/'evaluation_seal.json',dict(at_utc=now(),sha256=frozen,eval_sources=len(evaluation),fit_sources=len(fit)))
    save(OUT/'status.json',dict(technical='INCOMPLETE',scientific='NOT_EVALUATED',stage='SEALED',no_outcomes_exposed=True))
    return 0


def run():
    if not (OUT/'evaluation_seal.json').exists():
        rc = plan()
        if rc:
            return rc
    check_seal()
    if read(OUT/'status.json').get('technical') == 'COMPLETE':
        from run_native_head_value_confirmation import audit
        return 0 if audit()['passed'] else 1
    from executable_value_observability import score
    rows = read(OUT/'test_manifest.json')
    data = collect(rows,NativeEncoder(),'EVAL')
    predictions = {target:{v:np.concatenate([score(read(OUT/'probes'/f'final_{target}_{v}.json')['model'],d['x'][:,:size]) for d in data]) for v,size in VIEWS.items()} for target in ('advantage','drift')}
    atomic_npz(OUT/'eval_state_predictions.npz',**{target+'_'+view:value for target,views in predictions.items() for view,value in views.items()})
    for d in data:
        atomic_npz(OUT/'schedule_tables'/(rank('feature',d['row']['filename'])+'.npz'),
                   loss=d['targets']['loss'],prediction=d['targets']['prediction'])
    source = source_values(data,predictions)
    save(OUT/'eval_source_summaries.json',source)
    names = ['primary','opportunity','drift_contrast','H_minus_S','HG_minus_H','HGH_minus_H']+['V_'+v for v in VIEWS]
    started_cpu = time.perf_counter()
    infer = source_inference(rows,[[r[n] for n in names] for r in source],OUT/'bootstrap_draws.npz')
    primary_claims = interval_claims(infer['bootstrap95'][0])
    analytic_claims = interval_claims(infer['analytic95'][0])
    save(OUT/'summary.json',dict(inference=infer,columns=names,primary_claims=primary_claims,
        analytic_claims=analytic_claims,inference_sensitivity=primary_claims!=analytic_claims,
        scope='fixed native checkpoint/final probes/classes; approximate conditional intervals'))
    numeric = [n for n,v in source[0].items() if isinstance(v,(float,int)) and n!='y']
    save(OUT/'class_summaries.json',[dict(y=c,sources=sum(r['y']==c for r in source),
        means={n:float(np.mean([r[n] for r in source if r['y']==c])) for n in numeric}) for c in range(400)])
    save(OUT/'adverse_source_quantiles.json',{n:np.quantile([r[n] for r in source],[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist() for n in numeric})
    prediction_report(data,predictions)
    event('evaluation_cpu_summary',cpu_seconds=time.perf_counter()-started_cpu)
    timing(rows)
    from verify_native_head_value_confirmation import verify
    verified = verify(OUT,write=True)
    if not verified['passed']:
        raise RuntimeError('Independent empirical verification failed')
    render_results(source,infer,names)
    # Visual review is a separate real action; never auto-attest to viewing.
    save(OUT/'status.json',dict(technical='INCOMPLETE',scientific='UNRESOLVED',stage='VERIFIED_PENDING_VISUAL_REVIEW',
                              no_outcomes_exposed=False,eval_sources=len(rows)))
    return 2


def prediction_report(data,predictions):
    from sklearn.metrics import roc_auc_score, average_precision_score
    metrics = {}
    for target in ('advantage','drift'):
        y = np.concatenate([d['targets'][target] for d in data])
        metrics[target] = dict(distribution=np.quantile(y,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
            exact_zero=int(np.sum(y==0)),near_zero=int(np.sum(abs(y)<=1e-12)),rows=len(y),views={})
        for v in VIEWS:
            p = predictions[target][v]
            training = read(OUT/'probes'/f'final_{target}_{v}.json')
            # Signed fixed-width bins use fitting target scale only.
            scale = max(abs(training['constant']),1e-12)
            edges = np.r_[-np.inf,np.array([-2,-1,-.5,0,.5,1,2])*scale,np.inf]
            bins = []
            for lo,hi in zip(edges[:-1],edges[1:]):
                mask = (p>=lo)&(p<hi)
                bins.append(dict(lower=None if not np.isfinite(lo) else float(lo),upper=None if not np.isfinite(hi) else float(hi),
                    rows=int(mask.sum()),predicted=float(p[mask].mean()) if mask.any() else None,observed=float(y[mask].mean()) if mask.any() else None))
            rank_corr = spearmanr(y,p).statistic if np.ptp(y)>0 and np.ptp(p)>0 else None
            metrics[target]['views'][v] = dict(mae=float(np.mean(abs(y-p))),
                spearman=float(rank_corr) if rank_corr is not None else None,
                auroc=float(roc_auc_score(y>0,p)) if len(set(y>0))==2 else None,
                average_precision=float(average_precision_score(y>0,p)) if len(set(y>0))==2 else None,
                constant_reference_mae=float(np.mean(abs(y-training['constant']))),signed_bins=bins)
    save(OUT/'prediction_metrics.json',metrics)


def timing(rows):
    from executable_value_observability import score, hog, actions
    from causal_observability_policy import cheap_signal
    from temporal_refresh_common import snippet_grid
    cohort = read(OUT/'timing_manifest.json')
    encoder = NativeEncoder()
    models = {v:read(OUT/'probes'/f'final_advantage_{v}.json')['model'] for v in ('HG','HGH')}
    arms = ('Dense16','PeriodicB4','PeriodicB4_unused_HG','PeriodicB4_unused_HGH')
    trials_path = OUT/'timing_trials.jsonl'
    existing = [json.loads(line) for line in trials_path.read_text().splitlines()] if trials_path.exists() else []
    completed = {(r['component'],r['repetition'],r['arm']) for r in existing}
    for row in cohort:
        with np.load(feature_path(row),allow_pickle=False) as saved:
            reference = saved['logits'].astype(float)
        expected = reference[list(PERIODIC)].mean(0)
        for repetition in range(-2,5):
            order = sorted(arms,key=lambda a:rank('timing-order',f'{row["component"]}|{repetition}|{a}'))
            for arm in order:
                key = row['component'],repetition,arm
                if key in completed:
                    continue
                guard('timing')
                load_start = time.perf_counter()
                encoded = (ROOT/row['path']).read_bytes()
                load_seconds = time.perf_counter()-load_start
                encoder.torch.cuda.synchronize()
                start = time.perf_counter()
                encoder.torch.cuda.reset_peak_memory_stats()
                pts,frames = decode(encoded,pixels='raw')
                grid_start = time.perf_counter()
                grid = snippet_grid([p['seconds'] for p in pts])
                if pts!=row['pts'] or grid!=row['grid']:
                    raise RuntimeError('Timing PTS/grid identity mismatch')
                grid_seconds = time.perf_counter()-grid_start
                decode_seconds = grid_start-start
                history = {}
                logits = {}
                thumbnails = []
                descriptors = []
                conversion_seconds = gray_seconds = hog_seconds = history_seconds = scoring_seconds = encoder_seconds = 0.
                decisions = []
                extra = arm.startswith('PeriodicB4_unused_')
                view = arm.rsplit('_',1)[-1] if extra else None
                for t,indices in enumerate(grid['frame_indices']):
                    if extra:
                        tick = time.perf_counter()
                        thumb = frames[indices[-1]].reformat(width=32,height=32,format='gray').to_ndarray()
                        thumbnails.append(thumb)
                        conversion_seconds += time.perf_counter()-tick
                        if view=='HGH':
                            tick = time.perf_counter()
                            descriptors.append(hog(thumb))
                            hog_seconds += time.perf_counter()-tick
                        if t in range(1,13):
                            tick = time.perf_counter()
                            gray = cheap_signal(thumb,thumbnails[t-1])
                            gray_seconds += time.perf_counter()-tick
                            tick = time.perf_counter()
                            hh = None
                            if view=='HGH':
                                r = max(u for u in PERIODIC if u<t)
                                hh = np.r_[np.linalg.norm(descriptors[t]-descriptors[r],axis=1),np.linalg.norm(descriptors[t]-descriptors[t-1],axis=1)]
                            hog_seconds += time.perf_counter()-tick
                            tick = time.perf_counter()
                            x = native_observation(t,{u:history[u] for u in PERIODIC if u<t},gray,hh)
                            history_seconds += time.perf_counter()-tick
                            tick = time.perf_counter()
                            value = float(score(models[view],x))
                            decisions.append(dict(t=t,prediction=value,ignored_action=int(actions(np.array([value]),np.array([t]))[0])))
                            scoring_seconds += time.perf_counter()-tick
                    if arm=='Dense16' or t in PERIODIC:
                        tick = time.perf_counter()
                        ff,zz = encoder.forward([frames[i].to_ndarray(format='rgb24') for i in indices],'TIMING')
                        encoder_seconds += time.perf_counter()-tick
                        logits[t] = zz
                        if extra:
                            history[t] = (ff,zz)
                output = np.mean([logits[t] for t in sorted(logits)],axis=0,dtype=np.float64)
                encoder.torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                wanted = reference.mean(0) if arm=='Dense16' else expected
                np.testing.assert_allclose(output,wanted,atol=2e-5,rtol=2e-5)
                calls = 16 if arm=='Dense16' else 4
                if len(logits)!=calls:
                    raise RuntimeError('Timing encoder budget mismatch')
                record = dict(component=row['component'],y=row['y'],repetition=repetition,arm=arm,
                    warmup=repetition<0,seconds=elapsed,file_load_seconds=load_seconds,encoder_calls=calls,
                    decoded_frames=len(pts),decode_seconds=decode_seconds,grid_seconds=grid_seconds,
                    conversion_seconds=conversion_seconds,gray_seconds=gray_seconds,hog_seconds=hog_seconds,
                    history_seconds=history_seconds,scoring_seconds=scoring_seconds,encoder_seconds=encoder_seconds,
                    cuda_peak_allocated_bytes=encoder.torch.cuda.max_memory_allocated(),
                    logits=output.tolist(),ignored_decisions=decisions,
                    boundaries='Encoder timer nests selected RGB conversion/preprocessing/transfers/native inference; observation conversion is separate; raw-frame full decode is common; component medians must not be summed')
                with trials_path.open('a',encoding='utf-8',newline='\n') as f:
                    f.write(json.dumps(record)+'\n')
                existing.append(record)
                event('timing_complete',timing_seconds=elapsed)
    # Equality of all Periodic arms is exact across real trial predictions.
    paired = []
    for row in cohort:
        relevant = [r for r in existing if r['component']==row['component'] and not r['warmup']]
        by = {(r['repetition'],r['arm']):r for r in relevant}
        for rep in range(5):
            base = by[rep,'PeriodicB4']['logits']
            for arm in arms[2:]:
                if base!=by[rep,arm]['logits']:
                    raise RuntimeError('Unused-observation timing arms changed native predictions')
        pairs = [('HG_minus_Periodic',arms[2],arms[1]),('HGH_minus_Periodic',arms[3],arms[1]),('HGH_minus_HG',arms[3],arms[2])]
        record = dict(component=row['component'],y=row['y'])
        for name,a,b in pairs:
            record[name] = float(np.median([by[rep,a]['seconds']-by[rep,b]['seconds'] for rep in range(5)]))*1000
        for arm in arms:
            record[arm] = float(np.median([by[rep,arm]['seconds'] for rep in range(5)]))*1000
        paired.append(record)
    save(OUT/'timing_source_summaries.json',paired)
    save(OUT/'timing_summary.json',{k:dict(median_ms=float(np.median([r[k] for r in paired])),p95_ms=float(np.quantile([r[k] for r in paired],.95))) for k in paired[0] if k not in ('component','y')})


def render_results(source,infer,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = []
    fig,ax = plt.subplots(figsize=(8,4))
    labels = ['primary']+['V_'+v for v in VIEWS]
    for j,name in enumerate(labels):
        i = names.index(name); lo,hi = infer['bootstrap95'][i]; mean = infer['mean'][i]
        ax.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',color='navy')
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='CE reduction (nats / mean single intervention)')
    ax.axvline(0,color='gray');ax.axvline(.01,color='orange',linestyle='--');fig.tight_layout();figures.append((1,fig))
    fig,axes = plt.subplots(1,2,figsize=(9,4))
    metrics = read(OUT/'prediction_metrics.json')
    for ax,target,title in zip(axes,('drift','advantage'),('Cosine drift','Signed task advantage')):
        ax.bar(list(VIEWS),[metrics[target]['views'][v]['mae'] for v in VIEWS]);ax.set(title=title,ylabel='MAE (cosine units)' if target=='drift' else 'MAE (nats)')
    fig.tight_layout();figures.append((2,fig))
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(['Opportunity']+list(VIEWS),[np.mean([r['opportunity'] for r in source])]+[np.mean([r['V_'+v] for r in source]) for v in VIEWS]);axes[0].set(ylabel='CE reduction (nats)')
    for v in VIEWS:
        x = sorted(r['V_'+v] for r in source);axes[1].plot(x,np.arange(1,len(x)+1)/len(x),label=v)
    axes[1].axvline(0,color='gray');axes[1].set(xlabel='Source mean value (nats)',ylabel='Empirical CDF');axes[1].legend();fig.tight_layout();figures.append((3,fig))
    fig,ax = plt.subplots(figsize=(9,4));summary = read(OUT/'timing_summary.json');arms = ['Dense16','PeriodicB4','PeriodicB4_unused_HG','PeriodicB4_unused_HGH']
    ax.barh(arms,[summary[a]['median_ms'] for a in arms]);ax.set(xlabel='Median complete-system latency (ms)',title='Measured controls; unused decisions do not change Periodic predictions');fig.tight_layout();figures.append((4,fig))
    for number,fig in figures:
        fig.savefig(OUT/f'figure_{number}.png',dpi=170);fig.savefig(OUT/f'figure_{number}.pdf');plt.close(fig)


def execute(mode):
    if not (OUT/'status.json').exists():
        raise RuntimeError('Run preflight first')
    try:
        with threadpool_limits(limits=1):
            if mode=='finalize':
                check_seal()
                review = read(OUT/'visual_review.json')
                if review.get('actually_inspected') is not True:
                    raise RuntimeError('Visual inspection is not recorded')
                for i in range(1,5):
                    for extension in ('png','pdf'):
                        name = f'figure_{i}.{extension}'
                        if review.get('sha256',{}).get(name)!=sha(OUT/name):
                            raise RuntimeError('Visual review identity mismatch: '+name)
                from verify_native_head_value_confirmation import verify
                if not verify(OUT,write=True)['passed']:
                    raise RuntimeError('Final independent verification failed')
                summary = read(OUT/'summary.json')
                claims = summary['primary_claims']
                scientific = 'UNRESOLVED'
                if not summary['inference_sensitivity']:
                    scientific = next((v for v in ('MATERIAL_INCREMENTAL_BENEFIT','MATERIAL_HARM','BOUNDED_SMALL_EFFECT','POSITIVE_BUT_MATERIALITY_UNRESOLVED') if v in claims),'UNRESOLVED')
                exposure = read(OUT/'evaluation_exposure.json')
                exposure.update(completed=True,completed_at_utc=now())
                save(OUT/'evaluation_exposure.json',exposure)
                save(OUT/'status.json',dict(technical='COMPLETE',scientific=scientific,no_outcomes_exposed=False,
                    supported_claims=claims,primary_interval=summary['inference']['bootstrap95'][0]))
                return 0
            return plan() if mode == 'plan' else run()
    except Exception as exc:
        text = str(exc)
        technical = next((s for s in ('BLOCKED_RESOURCE','BLOCKED_PRECISION_OR_POPULATION','BLOCKED_INPUT') if s in text),'VERIFICATION_FAILED')
        exposed = (OUT/'evaluation_exposure.json').exists()
        save(OUT/'status.json',dict(technical=technical,scientific='NOT_EVALUATED' if not exposed else 'UNRESOLVED',
                                  blocker=text,no_outcomes_exposed=not exposed))
        event('stage_failure',mode=mode,error=text)
        print(text)
        return 2
