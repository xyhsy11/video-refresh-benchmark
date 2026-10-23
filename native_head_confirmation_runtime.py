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
from native_head_confirmation_state import metered, advance, state, STAGES


def identities():
    names = CODE+['results/'+STUDY+'/'+n for n in
        ('protocol.md','frozen_config.json','class_mapping.json','model_identity.json','historical_source_evidence.json')]
    return {n:sha(ROOT/n) for n in names}


def check_seal(full=True,source=None):
    with metered('cpu','seal-identity-audit'):
        seal = read(OUT/'evaluation_seal.json')
        video_names=set(seal.get('video_paths',[]))
        for name, digest in seal['sha256'].items():
            if not full and name not in seal.get('use_controls',[]) and (source is None or name!=source['path']):
                continue
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
    from native_head_confirmation_state import totals
    return totals()


def guard(stage):
    from native_head_confirmation_state import guard as check
    check(stage)


def decode(encoded, pixels=False, needed=None,log=True,counts=None):
    import av
    from research_utils import configure_decoder
    if log:event('decode_start', decodes=1)
    pts, frames = [], {}
    try:
        with av.open(io.BytesIO(encoded)) as container:
            stream = container.streams.video[0]
            configure_decoder(stream)
            for i, frame in enumerate(container.decode(stream)):
                if counts is not None:counts['decoded_frames']=i+1
                if frame.pts is None:
                    raise ValueError('Missing genuine PTS')
                pts.append(dict(pts=int(frame.pts), time_base=str(frame.time_base), seconds=float(frame.pts*frame.time_base)))
                if pixels and (needed is None or i in needed):
                    frames[i] = frame if pixels == 'raw' else (frame.to_ndarray(format='rgb24'),
                                 frame.reformat(width=32,height=32,format='gray').to_ndarray())
        seconds = np.array([p['seconds'] for p in pts])
        if not len(pts) or not np.isfinite(seconds).all() or not (np.diff(seconds)>0).all():
            raise ValueError('Genuine finite strictly increasing presentation PTS required')
        if log:event('decode_complete', decoded_frames=len(pts))
        return pts, frames
    except Exception as exc:
        if log:event('decode_failure', error=str(exc), decoded_frames=len(pts))
        raise


def local_path(name, split):
    locations = [CACHE/'videos'/split/name]
    if split == 'val':
        locations += [ROOT/'data/source/cache_drift_v2/videos'/name, ASSETS/'videos'/name,ASSETS/'evaluation_candidates'/name]
    return next((p for p in locations if p.is_file()), None)


def screen_population(split, m):
    import av
    from temporal_refresh_common import snippet_grid
    from native_head_confirmation_inputs import candidate_groups
    from scipy.fft import dctn
    from PIL import Image
    counts=Counter();selected=[];prefix=[]
    for component, members in candidate_groups(split):
        if all(counts[r['y']]>=m for r in members):continue
        chosen = None
        for member in members:
            from native_head_confirmation_availability import membership
            availability=membership(split,member['filename'])
            if availability['state']!='PRESENT':
                from native_head_confirmation_inputs import MissingNeededInput
                raise MissingNeededInput(split,member['filename'])
            path = local_path(member['filename'], split)
            if path is None:
                save(OUT/f'{split}_prefix_certificate.json',dict(complete=False,per_class=m,counts=dict(counts),prefix=prefix,
                    unresolved=dict(component=component,filename=member['filename'],reason='missing_needed_original_RGB')))
                from native_head_confirmation_inputs import MissingNeededInput
                raise MissingNeededInput(split,member['filename'])
            receipt=CACHE/'screening'/(rank('screen',split+'|'+member['filename'])+'.json')
            if receipt.exists():
                old=read(receipt)
                if old['encoded_sha256']!=sha(path):raise RuntimeError('Screened source bytes changed')
                if old['encoded_sha256']!=availability['origins'][0]['encoded_sha256']:
                    raise RuntimeError('BLOCKED_INPUT: screened source is not the pinned archive original')
                if old['eligible']:
                    chosen=dict(old['row'],**member);break
                prefix.append(dict(component=component,filename=member['filename'],reason='technical_ineligible',receipt=rel(receipt)))
                continue
            guard('collection')
            digest=sha(path)
            if digest!=availability['origins'][0]['encoded_sha256']:
                raise RuntimeError('BLOCKED_INPUT: cached original differs from pinned archive member')
            with metered('collection','technical-screen',source=member['filename']):
                try:
                    encoded=path.read_bytes();pts,_=decode(encoded)
                    grid=snippet_grid([p['seconds'] for p in pts])
                    indexes=[int(q*(len(pts)-1)) for q in (.1,.5,.9)]
                    needed=set(indexes)|set(sum(grid['frame_indices'],[]))
                    checked,frames=decode(encoded,pixels='raw',needed=needed)
                    if checked!=pts:raise RuntimeError('Input PTS changed across fresh decode passes')
                    for indices in grid['frame_indices']:
                        if len({(frames[i].width,frames[i].height) for i in indices})!=1:
                            raise ValueError('Frozen snippet has unsupported changing spatial resolution')
                    hashes={}
                    for variant in ('rgb_pil','gray_bicubic'):
                        hashes[variant]=[]
                        for i in indexes:
                            frame=frames[i]
                            image=(Image.fromarray(frame.to_ndarray(format='rgb24')).convert('L').resize((32,32)) if variant=='rgb_pil'
                                   else Image.fromarray(frame.to_ndarray(format='gray')).resize((32,32),resample=Image.Resampling.BICUBIC))
                            low=dctn(np.asarray(image,dtype=float),norm='ortho')[:8,:8].ravel()[1:]
                            hashes[variant].append(''.join('1' if v>np.median(low) else '0' for v in low))
                    chosen=dict(member,path=rel(path),encoded_sha256=digest,encoded_bytes=path.stat().st_size,
                        pts=pts,grid=grid,perceptual_fingerprints=hashes,pts_span=pts[-1]['seconds']-pts[0]['seconds'],
                        peak_retained_raw_frame_bytes=sum(frames[i].width*frames[i].height*3 for i in needed),
                        dimensions=sorted({(frames[i].width,frames[i].height) for i in needed}))
                    save(receipt,dict(eligible=True,encoded_sha256=digest,row=chosen))
                except (ValueError,av.error.FFmpegError) as exc:
                    save(receipt,dict(eligible=False,encoded_sha256=digest,detail=str(exc)))
                    prefix.append(dict(component=component,filename=member['filename'],reason='technical_ineligible',receipt=rel(receipt),detail=str(exc)))
        if chosen:
            y=chosen['y']
            if counts[y]<m:selected.append(chosen);counts[y]+=1
            prefix.append(dict(component=component,filename=chosen['filename'],reason='selected' if chosen in selected else 'canonical_class_already_full'))
        if len(counts)==400 and min(counts.values())==m:break
    if len(counts)!=400 or min(counts.values())<m:
        save(OUT/f'{split}_shortfalls.json',{str(y):max(0,m-counts[y]) for y in range(400)})
        raise RuntimeError('BLOCKED_PRECISION_OR_POPULATION: all-class deterministic prefix shortfall')
    selected.sort(key=lambda r:(r['y'],rank('component',r['component'])))
    validate_components(selected, m)
    save(OUT/f'{split}_prefix_certificate.json',dict(complete=True,per_class=m,counts={str(y):counts[y] for y in range(400)},prefix=prefix,
        candidate_sha256=sha(CACHE/'candidate_manifest.json'),lower_ranked_locality_required=False))
    save(OUT/f'{split}_manifest.json.gz', selected)
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
        self.starts=self.completions=0
        self.validate_native=False
        self.native_checks=[]

    def forward(self, rgbs, phase,validate=False):
        torch = self.torch
        x = torch.from_numpy(np.stack(rgbs)).permute(3,0,1,2).float()/255
        x = self.normalize(x,[.45]*3,[.225]*3)
        x = self.center_crop(self.functional.short_side_scale(x,182),[182,182]).unsqueeze(0).cuda()
        self.starts+=1
        with torch.inference_mode():
            native = self.model(x)
            f = self.capture['input'].mean(tuple(range(1,self.capture['input'].ndim-1)))
            if validate or self.validate_native:
                projected = self.capture['projected'].mean((1,2,3))
                affine = torch.nn.functional.linear(f,self.model.blocks[-1].proj.weight,self.model.blocks[-1].proj.bias)
                torch.testing.assert_close(native,projected,atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(native,affine,atol=2e-5,rtol=2e-5)
                self.native_checks.append(dict(native_vs_projection_max_error=float((native-projected).abs().max().item()),
                    native_vs_affine_max_error=float((native-affine).abs().max().item()),native_classes=int(native.shape[-1])))
            torch.cuda.synchronize()
            result = f[0].cpu().numpy(), native[0].cpu().numpy()
        self.completions+=1
        return result


def feature_path(row):
    return CACHE/'features'/(rank('feature',row['filename'])+'.npz')


def collect(rows, encoder, phase):
    result = []
    if phase=='EVAL':check_seal()
    for row in rows:
        with metered('cpu','source-input-controls'):
            guard('collection')
            if phase == 'EVAL':
                check_seal(full=False,source=row)
            target = feature_path(row)
            identity = target.with_suffix('.json')
            expected = dict(encoded_sha256=row['encoded_sha256'], grid_sha256=rank('grid',json.dumps(row['grid'],sort_keys=True)),
                            checkpoint_sha256=CHECKPOINT_SHA, phase=phase, implementation=identities())

        if target.exists() or identity.exists():
            with metered('cpu','cached-source-authentication'):
                if not target.exists() or not identity.exists():
                    raise RuntimeError('Partial output needs audit, never silently reused')
                old = read(identity)
                if old['input'] != expected or old['array_sha256'] != sha(target):
                    raise RuntimeError('Atomic source resume identity mismatch')

        else:
            resource_start=time.perf_counter()
            with metered('collection','native-reference',phase=phase,source=row['component'],planned_encoder_calls=16):
                start = time.perf_counter();before=(encoder.starts,encoder.completions);checks_before=len(encoder.native_checks)
                try:
                    path = ROOT/row['path']
                    if sha(path) != row['encoded_sha256']:raise RuntimeError('Sealed video bytes changed')
                    needed = set(sum(row['grid']['frame_indices'],[]))
                    pts, frames = decode(path.read_bytes(),pixels=True,needed=needed)
                    if pts != row['pts']:raise RuntimeError('Sealed PTS identity changed')
                    f, z, thumbnails = [], [], []
                    if phase=='EVAL' and not (OUT/'evaluation_exposure.json').exists():
                        save(OUT/'evaluation_exposure.json', dict(at_utc=now(), seal_sha256=sha(OUT/'evaluation_seal.json'),
                            status='EXPOSED_FROM_FIRST_MODEL_ATTEMPT', completed=False))
                    for indices in row['grid']['frame_indices']:
                        ff, zz = encoder.forward([frames[i][0] for i in indices],phase,validate=phase=='PROBE_FIT_SMOKE')
                        f.append(ff);z.append(zz);thumbnails.append(frames[indices[-1]][1])
                    atomic_npz(target,features=np.array(f),logits=np.array(z),thumbnails=np.array(thumbnails))
                    save(identity,dict(input=expected,array_sha256=sha(target),genuine_collection_seconds=time.perf_counter()-start,
                        encoder_calls=encoder.starts-before[0],completed_calls=encoder.completions-before[1],
                        native_checks=encoder.native_checks[checks_before:],
                        cuda_peak_allocated_bytes=encoder.torch.cuda.max_memory_allocated(),
                        cuda_peak_reserved_bytes=encoder.torch.cuda.max_memory_reserved()))
                finally:
                    event('source_collection_attempt',phase=phase,source=row['component'],
                        encoder_starts=encoder.starts-before[0],encoder_completions=encoder.completions-before[1])
            receipt=read(identity);receipt['genuine_resource_elapsed_seconds']=time.perf_counter()-resource_start
            save(identity,receipt)
        with metered('cpu','offline-source-enumeration',source=row['component']):
            with np.load(target,allow_pickle=False) as saved:
                f,z,thumbs = saved['features'],saved['logits'],saved['thumbnails']
            target_values = consequences(z,f,row['y'])
            table_dir = 'schedule_tables' if phase=='EVAL' else 'fit_schedule_tables'
            table=OUT/table_dir/(rank('feature',row['filename'])+'.npz')
            if not table.exists():atomic_npz(table,loss=target_values['loss'],prediction=target_values['prediction'])
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
            model_path=OUT/'probes'/f'{fold}_{target}_{view}.json'
            training_identity=rank('fit',json.dumps(dict(train=[data[i]['row']['encoded_sha256'] for i in train],
                target=target,view=view,fold=fold,implementation=identities()),sort_keys=True))
            if model_path.exists():
                cached=read(model_path)
                if cached.get('training_identity')!=training_identity:raise RuntimeError('Probe resume identity mismatch')
                model=cached['model']
                fit_elapsed=cached['genuine_fit_seconds']
            else:
                fit_started=time.perf_counter()
                with metered('cpu','ridge-fit',target=target,view=view,fold=fold):
                    event('probe_fit_attempt',probe_fits=1,target=target,view=view,fold=fold)
                    model = ridge_fit(x[train_rows,:size],y[train_rows],w)
                fit_elapsed=time.perf_counter()-fit_started
            pred_train = score(model,x[train_rows,:size])
            mean = float(np.average(y[train_rows],weights=w))
            report = dict(model=model,genuine_fit_seconds=fit_elapsed,training_identity=training_identity,target=target, view=view, fold=fold,
                train_mae=float(np.average(abs(pred_train-y[train_rows]),weights=w)),
                constant_train_mae=float(np.average(abs(mean-y[train_rows]),weights=w)), constant=mean)
            if len(test):
                pred = score(model,x[test_rows,:size])
                predictions[target][view][test_rows] = pred
                report.update(heldout_mae=float(np.mean(abs(pred-y[test_rows]))),constant_heldout_mae=float(np.mean(abs(mean-y[test_rows]))))
            if not model_path.exists():
                save(model_path,report)
                event('probe_fit_complete',probe_completions=1,target=target,view=view,fold=fold)


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
    from native_head_confirmation_inputs import PlanningMembershipChanged
    seen=set()
    while True:
        try:return plan_once()
        except PlanningMembershipChanged as exc:
            if (OUT/'evaluation_exposure.json').exists() or (OUT/'evaluation_seal.json').exists():raise
            signature=sha(CACHE/'candidate_manifest.json')
            if signature in seen:raise RuntimeError('BLOCKED_INPUT: joint fitting membership reconciliation did not converge') from exc
            seen.add(signature)
            # Preserve every superseded fitting-only output before rebuilding
            # OOF precision. Feature arrays stay reusable for unchanged inputs;
            # all ledger counters and prior stage receipts remain cumulative.
            import shutil
            generation=rank('planning-generation',now())
            history=OUT/'planning_history'/generation;history.mkdir(parents=True)
            names=['probes','precision_plan.json','fit_diagnostics.json','fit_oof_source_summaries.json',
                'fit_oof_state_predictions.npz','fit_folds.json','cpu_projection.json','fitting_source_certificate.json',
                'fitting_source_certificate_verification.json','joint_source_certificate.json','joint_source_certificate_verification.json',
                'certificates','val_manifest.json.gz','test_manifest.json.gz','val_prefix_certificate.json','test_prefix_certificate.json',
                'smoke_manifest.json','native_smoke_verification.json','smoke_independent_verification.json','throughput_projection.json']
            preserved={}
            for name in names:
                p=OUT/name
                if p.exists():
                    if p.is_file():preserved[name]=sha(p)
                    shutil.move(str(p),str(history/name))
            current=state();save(history/'prior_stage_state.json',current)
            current['receipts'].append(dict(stage='PREPARED',at_utc=now(),sha256={},reason=str(exc),prior_stage=current['stage'],generation=generation))
            current['stage']='PREPARED';current['planning_generation']=generation
            save(OUT/'stage_state.json',current)
            save(history/'supersession.json',dict(reason=str(exc),input_candidate_sha256=signature,preserved=preserved,
                no_eval_exposure=True,counters_reset=False))
            event('input_only_planning_generation_superseded',reason=str(exc),generation=generation,counters_reset=False)


def plan_once():
    from native_head_confirmation_inputs import acquire,certify,INPUT_CODE_SHA_AT_IMPORT,TRANSFER_IO_RECIPE
    from verify_native_head_value_confirmation import verify_source_certificate,scalar_loss
    from itertools import combinations
    if (OUT/'evaluation_seal.json').exists():
        check_seal();return 0
    if not (OUT/'fitting_source_certificate.json').exists():
        acquire('val')
    if read(OUT/'fitting_source_certificate.json').get('input_audit_implementation_sha256')!=INPUT_CODE_SHA_AT_IMPORT:
        certify('val')
    with metered('cpu','independent-fitting-source-check'):
        checked=verify_source_certificate(OUT,'fitting')
    if not checked['passed']:raise RuntimeError('BLOCKED_INPUT: fitting certification failed '+repr(checked['errors']))
    fit=read(OUT/'val_manifest.json.gz')
    from run_native_head_value_confirmation import freeze_config
    controls=read(OUT/'frozen_config.json')
    current_code={n:sha(ROOT/n) for n in CODE}
    if controls['implementation']!=current_code:
        if counters()['encoder_starts'] or counters()['probe_fits']:
            raise RuntimeError('BLOCKED_INPUT: code identity changed after real scientific computation; explicit preserved identity migration required')
        prior=OUT/'control_history'/(sha(OUT/'frozen_config.json')+'.json')
        if not prior.exists():save(prior,controls)
        freeze_config()
        event('pre_model_control_freeze',code_sha256=current_code,encoder_calls_before_freeze=0,
            reason='Final pre-exposure implementation repair; original preparation/config identities preserved')
    with metered('collection','native-model-setup'):encoder=NativeEncoder()
    smoke=sorted(fit,key=lambda r:rank('smoke',r['component']))[:8]
    if not (OUT/'smoke_manifest.json').exists():save(OUT/'smoke_manifest.json',smoke)
    if state().get('planning_generation') and not (OUT/'native_smoke_verification.json').exists():
        # Replanning may replace an original smoke source. Preserve prior
        # arrays/receipts, then run eight actual currently eligible sources;
        # cached vectors never stand in for new native checks or throughput.
        import shutil
        history=CACHE/'smoke_history'/state()['planning_generation']/rank('smoke-attempt',now())
        for row in smoke:
            p=feature_path(row)
            if p.exists():
                history.mkdir(parents=True,exist_ok=True)
                for original in (p,p.with_suffix('.json')):shutil.move(str(original),str(history/original.name))
        event('replanned_genuine_smoke_required',planning_generation=state()['planning_generation'],sources=8,counters_reset=False)
    encoder.validate_native=True
    encoder.torch.cuda.reset_peak_memory_stats()
    smoke_data=collect(smoke,encoder,'PROBE_FIT')
    encoder.validate_native=False
    native_checks=[check for r in smoke for check in read(feature_path(r).with_suffix('.json'))['native_checks']]
    if len(native_checks)!=128 or any(c['native_classes']!=400 for c in native_checks):
        raise RuntimeError('BLOCKED_INPUT: genuine native fitting-smoke checks missing')
    if not (OUT/'native_smoke_verification.json').exists():
        from native_head_confirmation_state import memory_snapshot
        save(OUT/'native_smoke_verification.json',dict(sources=8,genuine_native_forwards=len(native_checks),checks=native_checks,
            cuda_peak_allocated_bytes=max(read(feature_path(r).with_suffix('.json'))['cuda_peak_allocated_bytes'] for r in smoke),
            cuda_peak_reserved_bytes=max(read(feature_path(r).with_suffix('.json'))['cuda_peak_reserved_bytes'] for r in smoke),
            cpu_memory=memory_snapshot(),cached_replay_used_for_throughput=False))
    if not (OUT/'smoke_independent_verification.json').exists():
        with metered('cpu','smoke-scalar-verification'):
            maximum_error=0.
            for d in smoke_data:
                with np.load(feature_path(d['row']),allow_pickle=False) as saved:zz=saved['logits'].astype(float)
                scalar=[scalar_loss(zz,(0,*tail),d['row']['y'])[0] for tail in combinations(range(1,16),3)]
                error=float(np.max(abs(np.asarray(scalar)-d['targets']['loss'])));maximum_error=max(maximum_error,error)
                if error>1e-10:raise RuntimeError('Fitting-only all455 independent scalar verification failed')
            save(OUT/'smoke_independent_verification.json',dict(sources=8,schedules_per_source=455,max_ce_error=maximum_error,passed=True))
    # Sum saved genuine first-collection receipts. Cache replay wall time never
    # enters a throughput projection.
    genuine=sum(read(feature_path(r).with_suffix('.json'))['genuine_resource_elapsed_seconds'] for r in smoke)
    rate=genuine/8
    remaining_fit=rate*sum(not feature_path(r).exists() for r in fit)*1.5
    projection=dict(real_smoke_clips=8,genuine_collection_seconds=genuine,safety_factor=1.5,
        remaining_fitting_projection_seconds=remaining_fit,source='first genuine decode/native inference receipts; cached replay excluded')
    projection['timing_projection_seconds']=60*7*4*rate*2
    projection['timing_projection_basis']='twice the real dense-smoke resource time per timing arm, all four arms including fresh full decode; conservative projection, not measured timing'
    save(OUT/'throughput_projection.json',projection)
    if counters()['collection_seconds']+remaining_fit>CAPS['collection_seconds']:
        raise RuntimeError('BLOCKED_RESOURCE: fitting-only genuine throughput projection exceeds remaining collection cap')
    data=collect(fit,encoder,'PROBE_FIT')
    fs=folds(fit)
    predictions={target:{v:np.full(len(fit)*12,np.nan) for v in VIEWS} for target in ('advantage','drift')}
    with metered('cpu','fitting-OOF-planning'):
        for f in range(4):
            fold_start=time.perf_counter()
            fit_models(data,np.flatnonzero(fs!=f),np.flatnonzero(fs==f),predictions,f)
            if f==0 and not (OUT/'cpu_projection.json').exists():
                measured=sum(read(OUT/'probes'/f'0_{target}_{view}.json')['genuine_fit_seconds'] for target in ('advantage','drift') for view in VIEWS)
                remaining=measured*4*1.5
                save(OUT/'cpu_projection.json',dict(first_genuine_fixed_fold_seconds=measured,remaining_folds_and_final_projection_seconds=remaining,
                    safety_factor=1.5,includes_fixed_probes=8,search=False))
                if counters()['cpu_seconds']+remaining>CAPS['cpu_seconds']:
                    raise RuntimeError('BLOCKED_RESOURCE: actual first fixed-fold fitting projects beyond CPU budget')
        summaries=source_values(data,predictions)
        if not (OUT/'fit_diagnostics.json').exists():
            diagnostics=dict(scope='fitting-only OOF development, not fresh confirmation',targets={},fold_variation={})
            for target in ('advantage','drift'):
                actual=np.concatenate([d['targets'][target] for d in data])
                diagnostics['targets'][target]=dict(rows=len(actual),quantiles=np.quantile(actual,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
                    zero=int((actual==0).sum()),near_zero=int((abs(actual)<=1e-12).sum()))
                diagnostics['fold_variation'][target]={}
                for view in VIEWS:
                    reports=[read(OUT/'probes'/f'{fold}_{target}_{view}.json') for fold in range(4)]
                    diagnostics['fold_variation'][target][view]={name:[r[name] for r in reports] for name in
                        ('train_mae','heldout_mae','constant_train_mae','constant_heldout_mae')}
            save(OUT/'fit_diagnostics.json',diagnostics)
        if not (OUT/'precision_plan.json').exists():
            save(OUT/'fit_oof_source_summaries.json',summaries)
            atomic_npz(OUT/'fit_oof_state_predictions.npz',**{target+'_'+view:value for target,views in predictions.items() for view,value in views.items()})
            save(OUT/'fit_folds.json',fs.tolist())
            save(OUT/'precision_plan.json',precision_plan(fit,[r['primary'] for r in summaries]))
        pp=read(OUT/'precision_plan.json')
        advance('OOF_COMPLETE',[OUT/'precision_plan.json',OUT/'fit_oof_state_predictions.npz'])
    if pp['selected_m'] is None:raise RuntimeError('BLOCKED_PRECISION_OR_POPULATION: projected half-width needs m>20')
    exact_projection=rate*400*pp['selected_m']*1.5
    projection.update(exact_selected_m=pp['selected_m'],remaining_evaluation_projection_seconds=exact_projection)
    # Transfer/IO must fit the collection allowance as well as encoder work.
    events=[json.loads(line) for line in (OUT/'resource_exposure_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    transfers=[e for e in events if e['kind']=='archive_transfer' and e.get('complete') and e.get('bytes',0)>0 and e.get('io_recipe')==TRANSFER_IO_RECIPE]
    test_archives=[r for r in read(OUT/'archive_inventory.json') if r['split']=='test' and not r.get('cached_path')]
    remaining_transfer=sum(r['content_length'] for r in test_archives if not (CACHE/'catalogs'/(rank('archive',r['url'])+'.json')).exists())
    from native_head_confirmation_availability import remaining_work_projection
    work=remaining_work_projection(pp['selected_m'],events)
    projection['remaining_work']=work
    from native_head_confirmation_state import retained_bytes
    feature_bytes=sum(feature_path(r).stat().st_size+feature_path(r).with_suffix('.json').stat().st_size for r in smoke)/len(smoke)
    projection['projected_additional_test_feature_bytes']=feature_bytes*400*pp['selected_m']*1.3
    if retained_bytes()+work['projected_test_original_bytes']+projection['projected_additional_test_feature_bytes']>CAPS['retained_bytes']:
        save(OUT/'throughput_projection.json',projection)
        raise RuntimeError('BLOCKED_RESOURCE: actual fitting-byte projection for selected test originals/features exceeds retained allowance')
    if counters()['cpu_seconds']+work['independent_verification_projection_seconds']>CAPS['cpu_seconds']:
        save(OUT/'throughput_projection.json',projection)
        raise RuntimeError('BLOCKED_RESOURCE: measured independent-verification projection exceeds remaining CPU allowance')
    if remaining_transfer and transfers:
        seconds=sum(e['acquisition_seconds'] for e in transfers);byte_count=sum(e['bytes'] for e in transfers)
        transfer_projection=remaining_transfer*seconds/byte_count*1.2
        projection.update(measured_archive_transfer_bytes=byte_count,measured_archive_transfer_seconds=seconds,
            remaining_test_archive_bytes=remaining_transfer,remaining_transfer_projection_seconds=transfer_projection,
            transfer_safety_factor=1.2,network_scope='real completed resumed/full archive body requests, including their ledger and file validation cost')
        minimum_remaining_collection=transfer_projection+exact_projection+sum(work[k] for k in
            ('test_screening_projection_seconds','test_catalog_projection_seconds','test_duplicate_projection_seconds'))
        projection['minimum_remaining_collection_projection_seconds']=minimum_remaining_collection
        projection['included_projection_terms']='Full transfers, cataloging/hashing/retention, technical screening/authentication, duplicate screening, genuine reference extraction; independent verification separately gated in CPU.'
        save(OUT/'throughput_projection.json',projection)
        if counters()['collection_seconds']+minimum_remaining_collection>CAPS['collection_seconds']:
            needed=counters()['collection_seconds']+minimum_remaining_collection
            raise RuntimeError(f'BLOCKED_RESOURCE: exact m={pp["selected_m"]} requires projected collection allowance at least {needed:.0f}s before additional screening/catalog work; current cap {CAPS["collection_seconds"]}s')
    save(OUT/'throughput_projection.json',projection)
    remaining_collection=exact_projection+projection.get('remaining_transfer_projection_seconds',0)+sum(work[k] for k in
        ('test_screening_projection_seconds','test_catalog_projection_seconds','test_duplicate_projection_seconds'))
    projection['remaining_work_collection_projection_seconds']=remaining_collection
    save(OUT/'throughput_projection.json',projection)
    if counters()['collection_seconds']+remaining_collection>CAPS['collection_seconds']:
        raise RuntimeError('BLOCKED_RESOURCE: exact precision-selected EVAL m exceeds remaining collection cap')
    if counters()['timing_seconds']+projection['timing_projection_seconds']>CAPS['timing_seconds']:
        raise RuntimeError('BLOCKED_RESOURCE: genuine-smoke projection exceeds the fixed complete-system timing allowance')
    if not (OUT/'joint_source_certificate.json').exists():acquire('test')
    with metered('cpu','independent-joint-source-check'):
        checked=verify_source_certificate(OUT,'joint')
    if not checked['passed']:raise RuntimeError('BLOCKED_INPUT: final joint certification failed '+repr(checked['errors']))
    evaluation=read(OUT/'test_manifest.json.gz')
    with metered('cpu','final-probe-freeze'):
        fit_models(data,np.arange(len(fit)),[],predictions,'final')
        timing_classes=sorted(range(400),key=lambda c:rank('timing-class',c))[:60]
        timing_cohort=[min([r for r in evaluation if r['y']==c],key=lambda r:rank('timing-source',r['component'])) for c in timing_classes]
        save(OUT/'timing_manifest.json',timing_cohort)
        frozen=identities()
        paths=[OUT/n for n in ('val_manifest.json.gz','test_manifest.json.gz','precision_plan.json','timing_manifest.json',
               'fitting_source_certificate.json','joint_source_certificate.json','candidate_manifest_identity.json',
               'source_audit_identity.json','source_exclusion_ledger.json','input_duplicate_screen_identity.json')]
        paths+=sorted((OUT/'probes').glob('final_*.json'))
        paths+=[ROOT/read(OUT/'source_audit_identity.json')['path'],ROOT/read(OUT/'candidate_manifest_identity.json')['path'],
                ROOT/read(OUT/'input_duplicate_screen_identity.json')['path']]
        for scope in ('fitting','joint'):
            for name in read(OUT/f'{scope}_source_certificate.json')['sha256']:paths.append(ROOT/name)
        for row in fit:
            paths.extend([feature_path(row),feature_path(row).with_suffix('.json')])
        for path in paths:frozen[rel(path)]=sha(path)
        for row in fit+evaluation:frozen[row['path']]=row['encoded_sha256']
        use_controls=CODE+[rel(OUT/'protocol.md'),rel(OUT/'frozen_config.json')]+[rel(p) for p in sorted((OUT/'probes').glob('final_*.json'))]
        save(OUT/'evaluation_seal.json',dict(at_utc=now(),sha256=frozen,use_controls=use_controls,video_paths=[r['path'] for r in fit+evaluation],
            eval_sources=len(evaluation),fit_sources=len(fit)))
        advance('SEALED',[OUT/'evaluation_seal.json'])
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
    if STAGES.index(state()['stage'])>=STAGES.index('ANALYZED'):
        rows=read(OUT/'test_manifest.json.gz')
        if STAGES.index(state()['stage'])<STAGES.index('TIMED'):
            timing(rows);advance('TIMED',[OUT/'timing_trials.jsonl'])
        return finish_evaluation()
    from executable_value_observability import score
    rows = read(OUT/'test_manifest.json.gz')
    with metered('collection','native-evaluation-model-setup'):encoder=NativeEncoder()
    data = collect(rows,encoder,'EVAL')
    advance('EVAL_COLLECTED')
    with metered('cpu','sealed-source-scoring-and-analysis'):
        predictions = {target:{v:np.concatenate([score(read(OUT/'probes'/f'final_{target}_{v}.json')['model'],d['x'][:,:size]) for d in data]) for v,size in VIEWS.items()} for target in ('advantage','drift')}
        atomic_npz(OUT/'eval_state_predictions.npz',**{target+'_'+view:value for target,views in predictions.items() for view,value in views.items()})
        source = source_values(data,predictions)
        save(OUT/'eval_source_summaries.json',source)
        names = ['primary','opportunity','drift_contrast','H_minus_S','HG_minus_H','HGH_minus_H']+['V_'+v for v in VIEWS]
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
    advance('ANALYZED',[OUT/'summary.json'])
    timing(rows)
    advance('TIMED',[OUT/'timing_trials.jsonl'])
    return finish_evaluation()


def finish_evaluation():
    if state()['stage']=='VERIFIED_PENDING_VISUAL_REVIEW':return 2
    from verify_native_head_value_confirmation import verify
    with metered('cpu','independent-empirical-verification'):
        verified = verify(OUT,write=True)
    if not verified['passed']:
        raise RuntimeError('Independent empirical verification failed')
    summary=read(OUT/'summary.json');source=read(OUT/'eval_source_summaries.json')
    with metered('cpu','render-results'):render_results(source,summary['inference'],summary['columns'])
    advance('VERIFIED_PENDING_VISUAL_REVIEW')
    # Visual review is a separate real action; never auto-attest to viewing.
    save(OUT/'status.json',dict(technical='INCOMPLETE',scientific='UNRESOLVED',stage='VERIFIED_PENDING_VISUAL_REVIEW',
                              no_outcomes_exposed=False,eval_sources=len(read(OUT/'test_manifest.json.gz'))))
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
    with metered('timing','native-timing-model-setup'):encoder = NativeEncoder()
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
                check_seal(full=False,source=row)
                with metered('timing','complete-system-trial',source=row['component'],arm=arm,repetition=repetition,
                             planned_encoder_calls=16 if arm=='Dense16' else 4):
                    before=(encoder.starts,encoder.completions)
                    decode_counts={'decodes':0,'decoded_frames':0}
                    try:
                        load_start = time.perf_counter()
                        encoded = (ROOT/row['path']).read_bytes()
                        load_seconds = time.perf_counter()-load_start
                        encoder.torch.cuda.synchronize()
                        start = time.perf_counter()
                        encoder.torch.cuda.reset_peak_memory_stats()
                        needed=set(sum(row['grid']['frame_indices'],[]))
                        decode_counts['decodes']=1
                        pts,frames = decode(encoded,pixels='raw',needed=needed,log=False,counts=decode_counts)
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
                        event('timing_complete',measured_seconds=elapsed)
                    finally:
                        event('timing_attempt_receipt',source=row['component'],arm=arm,
                              encoder_starts=encoder.starts-before[0],encoder_completions=encoder.completions-before[1],**decode_counts)

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
        ax.plot([lo,hi],[j,j],color='navy');ax.plot(mean,j,'o',color='navy')
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='CE reduction (nats / mean single intervention)')
    ax.axvline(0,color='gray');ax.axvline(.01,color='orange',linestyle='--');fig.tight_layout();figures.append((1,fig))
    fig,axes = plt.subplots(1,2,figsize=(9,4))
    metrics = read(OUT/'prediction_metrics.json')
    for ax,target,title in zip(axes,('drift','advantage'),('Cosine drift','Signed task advantage')):
        ax.bar(list(VIEWS),[metrics[target]['views'][v]['mae'] for v in VIEWS]);ax.set(title=title,ylabel='MAE (cosine units)' if target=='drift' else 'MAE (nats)')
    fig.tight_layout();figures.append((2,fig))
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    labels=['opportunity']+['V_'+v for v in VIEWS]
    for j,name in enumerate(labels):
        i=names.index(name);lo,hi=infer['bootstrap95'][i];mean=infer['mean'][i]
        axes[0].plot([j,j],[lo,hi],color='navy');axes[0].plot(j,mean,'o',color='navy')
    axes[0].set(xticks=range(len(labels)),xticklabels=['Opportunity']+list(VIEWS),ylabel='CE reduction (nats)')
    axes[0].axhline(.01,color='orange',linestyle='--');axes[0].axhline(0,color='gray')
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
        if read(OUT/'status.json').get('technical')=='COMPLETE':
            from run_native_head_value_confirmation import audit
            return 0 if audit()['passed'] else 1
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
                with metered('cpu','final-independent-verification'):
                    verified=verify(OUT,write=True)
                if not verified['passed']:
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
                advance('COMPLETE')
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
    finally:
        from native_head_confirmation_state import snapshot_accounting
        snapshot_accounting()
        from native_head_confirmation_availability import selection_attrition
        with metered('cpu','population-attrition-report'):selection_attrition()
        snapshot_accounting()
        write_findings()


def execute_input_stage(mode,split):
    from native_head_confirmation_inputs import acquire,certify
    from native_head_confirmation_state import snapshot_accounting
    event('input_stage_enter',mode=mode,split=split,previous_status=read(OUT/'status.json'))
    save(OUT/'status.json',dict(technical='INCOMPLETE',scientific='NOT_EVALUATED',stage=mode.upper()+'_'+split.upper(),no_outcomes_exposed=True))
    try:
        rows=acquire(split) if mode=='acquire' else certify(split)
        save(OUT/'status.json',dict(technical='INCOMPLETE',scientific='NOT_EVALUATED',stage=state()['stage'],
            selected_sources=len(rows),no_outcomes_exposed=True))
        return 0
    except Exception as exc:
        message=str(exc)
        technical=next((s for s in ('BLOCKED_RESOURCE','BLOCKED_PRECISION_OR_POPULATION','BLOCKED_INPUT') if s in message),'VERIFICATION_FAILED')
        save(OUT/'status.json',dict(technical=technical,scientific='NOT_EVALUATED',blocker=message,no_outcomes_exposed=True))
        event('input_stage_failure',mode=mode,error=message)
        print(message,flush=True)
        return 2
    finally:
        snapshot_accounting()
        write_findings()


def write_findings():
    """Only report actual stored artifacts; absence is explicit, never filled."""
    status=read(OUT/'status.json');account=read(OUT/'execution_accounting.json')
    fit=read(OUT/'val_manifest.json.gz') if (OUT/'fitting_source_certificate.json').exists() else []
    evaluation=read(OUT/'test_manifest.json.gz') if (OUT/'joint_source_certificate.json').exists() else []
    summary=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
    lines=['# Experiment 27: native-head confirmation','',
        'Technical status: **'+status['technical']+'**. Scientific status: **'+status['scientific']+'**.', '',
        'The immutable blocked preparation is commit a9e1942f226f341071e1a990ea446dc717a070bb and preparation_a9e1942/.',
        'This is a native-classifier transfer/domain replication of the E26 diagnostic; it cannot identify a head-only causal effect.', '',
        f'Authenticated native classifier: 400 outputs. Certified fitting sources: {len(fit)}. Certified EVAL sources: {len(evaluation)}.',
        f'Encoder attempts/completions in the cumulative ledger: {account["encoder_starts"]}/{account["encoder_completions"]}.',
        'No simulated or fabricated empirical data were used. Archived feature replay in contract checks is explicitly development/verification only.','']
    if status.get('blocker'):lines+=['Remaining prerequisite: '+status['blocker'],'']
    acquisition=OUT/'acquisition_inventory_verification.json'
    if acquisition.exists():
        observed=read(acquisition)
        lines += [f'Actual input-only work: {observed["validated_archives"]} complete validation archives cataloged, {observed["cataloged_original_members"]} archive members, {observed["retained_originals"]} retained original candidate/reserve files.',
            f'Technical screening: {observed["screened_inputs"]} candidates; {observed["technically_eligible_receipts"]} eligible receipts and {observed["technically_ineligible_receipts"]} technical failures. These are not a certified all-class cohort.','']
    amended=(OUT/'population_rule.json').exists()
    if amended:
        attrition=read(OUT/'archive_population_attrition.json')
        lines+=['Population rule: archive_available_v1. Inference concerns eligible archive-available official K400 clips in the fixed 400 classes; availability is not assumed missing at random.',
            'Archive membership states: '+json.dumps({s:r['states'] for s,r in attrition['splits'].items()})+'.',
            'The d978026 blocked attempt and old missing-input reports remain historical evidence; confirmed release absence supersedes their blocker status. See population_amendment.md and blocked_d978026/snapshot_identity.json.','']
    coverage=OUT/'annotation_archive_coverage.json'
    if coverage.exists() and not amended:
        missing=read(coverage)['prospective_prefix_missing']
        if missing:lines += ['Missing prospective fitting inputs: '+', '.join(r['filename']+' ('+r['label']+')' for r in missing)+'.',
            'The first missing parent returned oEmbed404; its public watch page returned playability ERROR (cannot be played). The second returned public oEmbed metadata, which does not supply authenticated RGB. See missing_input_public_availability.json and missing_input_parent_playability.json.','']
    if summary:
        infer=summary['inference'];names=summary['columns']
        lines+=['| Claim / endpoint | Real evidence |','|---|---|']
        for name in ('primary','opportunity','drift_contrast'):
            i=names.index(name);lines.append(f'| {name} | mean {infer["mean"][i]:.12g}; approximate conditional bootstrap95 {infer["bootstrap95"][i]}; analytic95 {infer["analytic95"][i]} |')
        lines+=['','Intervals condition on the fixed checkpoint, fitted/scaled probes, fixed classes and known-source independence assumptions. They omit fitting variation, selection of architecture/checkpoint and unknown duplicates. Supporting intervals are marginal and unadjusted.',
            'Literal primary claims: '+', '.join(summary['primary_claims'])+'.',
            'Inference sensitivity: '+str(summary['inference_sensitivity'])+'.','']
    else:
        lines+=['Primary estimate/interval, supporting task opportunity and matched drift contrast: **not evaluated**. No scientific conclusion or outcome figures are supplied.','',
            '| Claim | Evidence |','|---|---|','| Native checkpoint/class mapping authenticated | model_identity.json, class_mapping.json |',
            '| Scientific confirmation | Not executed; no EVAL effect or interval |',
            '| Implementation verification | focused_checks.json and independent_verification.json; passing these is not empirical completion |','']
    lines+=['Resource scope: 61 GiB cumulative network, 20 GiB additional retained data, 8h collection, 4h CPU, 2h timing; 10 GiB free-disk reserve.',
        f'Known recorded HTTP response bodies: {account["known_recorded_http_body_bytes"]} bytes; conservative cap charge: {account["network_cap_charged_bytes"]} bytes (includes {account["uncertain_network_upper_bound_extra_bytes"]} bytes of interrupted-read uncertainty). Current additional retained data: {account["additional_retained_bytes"]} bytes.',
        'Elapsed categories (seconds): '+json.dumps({k:account[k] for k in ('collection_seconds','cpu_seconds','timing_seconds','acquisition_seconds')})+'.',
        'Acquisition is nested within collection and must not be added again; its subtotal includes '+str(account['acquisition_upper_bound_extra_seconds'])+' seconds of conservative interrupted-transfer upper charge.',
        'Original preparation failed-attempt wall times and Git/browser control transfer are not fully metered. No complete historical end-to-end resource claim is made.',
        'Known-parent, encoded-byte and available input-only perceptual duplicate evidence cannot exclude unknown reuploads or undocumented pretraining. No universal independence or adaptive-controller speedup claim is supported.', '',
        'Use the PowerShell commands in README.md. A completed invocation performs a read-only audit; interrupted attempts remain in receipts.', '',
        'Next action: resume the same amended study after resolving the concrete current gate recorded above.' if not summary else
        'Next research decision: assess probe adequacy, conditional precision and observation overhead together before deciding whether this fixed diagnostic supports a focused empirical manuscript.',
        'Technical completion is separate from scientific benefit and does not guarantee TMLR acceptance.']
    (OUT/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
