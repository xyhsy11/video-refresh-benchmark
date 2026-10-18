"""One frozen X3D-XS, real native-codec inputs and measured end-to-end trials."""
from __future__ import annotations
import importlib.util
import io
import json
import time
import numpy as np
from ucf101_proxy_common import *


class X3D:
    def __init__(self):
        setup_start = time.perf_counter()
        import torch
        import pytorchvideo
        from pytorchvideo.models.hub.x3d import x3d_xs
        from benchmark import configure_numerics
        self.torch = torch
        if not torch.cuda.is_available():
            raise Blocker('CUDA unavailable in existing temporal environment')
        self.numerics = configure_numerics(SEED)
        source = ROOT/f'data/source/temporal_refresh_generalization_v1/pytorchvideo-{REVISION}'
        if not Path(pytorchvideo.__file__).resolve().is_relative_to(source.resolve()):
            raise Blocker('Imported PyTorchVideo is not the pinned local revision')
        historical = read(ROOT/'results/temporal_refresh_generalization_v1/encoder_provenance.json')
        verify_hashes(historical['upstream_code_hashes'])
        setup = read(ROOT/'results/temporal_refresh_generalization_v1/dependency_setup.json')
        source_zip = source.parent/'pytorchvideo.zip'
        if sha(source_zip) != setup['upstream_zip_sha256']:
            raise Blocker('Pinned upstream source archive changed')
        import zipfile
        import hashlib
        with zipfile.ZipFile(source_zip) as archive:
            source_members = [n for n in archive.namelist() if '/pytorchvideo/' in n and n.endswith('.py')]
            for name in source_members:
                p = source.parent/name
                if not p.exists() or sha(p) != hashlib.sha256(archive.read(name)).hexdigest():
                    raise Blocker('Pinned upstream Python source mismatch: '+name)
        checkpoint = source.parent/'X3D_XS.pyth'
        if sha(checkpoint) != CHECKPOINT_SHA:
            raise Blocker('Official X3D checkpoint bytes mismatch')
        self.model = x3d_xs(pretrained=False)
        loaded = self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state'], strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise Blocker('Strict loading failed')
        self.model.eval().to(device='cuda', dtype=torch.float32)
        self.model.blocks[-1].proj.register_forward_hook(self.capture)
        spec = importlib.util.spec_from_file_location('ucf_pinned_video_functional', source/'pytorchvideo/transforms/functional.py')
        self.functional = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.functional)
        self.starts = self.completions = 0
        self.log = True
        self.phase = 'setup'
        self.clip = None
        self.identity = dict(revision=REVISION, checkpoint_path=rel(checkpoint), checkpoint_sha256=sha(checkpoint),
            upstream_code_hashes=historical['upstream_code_hashes'], factory='unchanged x3d_xs(pretrained=False), nominal crop160',
            preprocessing='4 RGB; C,T,H,W /255; mean .45 std .225; pinned bilinear shortside182; center crop182',
            feature='read-only proj input hook; mean residual spatial/temporal axes; 2048 pre-projection features',
            native_head='400-class projection and native averaging executed, activation=None; included in timings',
            numerics=self.numerics, strict_loading=True, torch=torch.__version__, torch_cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), checkpoint_terms='Official PyTorchVideo model-zoo checkpoint; source Apache-2.0; no new terms accepted')
        from importlib.metadata import version
        self.identity.update(upstream_zip_sha256=sha(source_zip), upstream_python_files_verified=len(source_members),
            package_versions={n: version(n) for n in ['torch', 'torchvision', 'av', 'numpy', 'scipy', 'pandas', 'Pillow']})
        p = OUT/'encoder_identity.json'
        if p.exists() and read(p) != self.identity:
            raise Blocker('Encoder/runtime identity changed; preserve old outputs and disclose amendment')
        if not p.exists():
            save(p, self.identity)
        with (OUT/'model_setup_events.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(dict(at_utc=now(), seconds=time.perf_counter()-setup_start,
                excluded_from_inference_timing=True, encoder_identity_sha256=sha(p)))+'\n')

    def capture(self, module, args, output):
        self.projection_input = args[0]
        self.projection_output = output

    def event(self, event, endpoint):
        if self.log:
            with (OUT/'encoder_events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(event=event, at_utc=now(), phase=self.phase, clip=self.clip, endpoint=endpoint))+'\n')

    def prepare(self, rgbs):
        from torchvision.transforms._functional_video import center_crop, normalize
        t = self.torch.from_numpy(np.stack(rgbs)).permute(3, 0, 1, 2).to(dtype=self.torch.float32)/255.
        t = normalize(t, [.45]*3, [.225]*3)
        return center_crop(self.functional.short_side_scale(t, 182), [182, 182]).unsqueeze(0)

    def feature(self, rgbs, endpoint):
        t = self.prepare(rgbs).to('cuda', non_blocking=True)
        self.starts += 1; self.event('forward_start', endpoint)
        self.native = self.model(t)
        f = self.projection_input.mean(dim=tuple(range(1, self.projection_input.ndim-1)))
        if f.shape != (1, 2048):
            raise Blocker(f'Unexpected feature shape {f.shape}')
        # Collection logs mark device completion; timed trials synchronize once at
        # their complete-prediction boundary and retain independent whole-trial counts.
        if self.log:
            self.torch.cuda.synchronize()
        self.completions += 1; self.event('forward_complete', endpoint)
        return f

    def native_check(self):
        torch = self.torch
        head = self.model.blocks[-1]
        f = self.projection_input.mean(dim=tuple(range(1, self.projection_input.ndim-1)))
        projected = self.projection_output.mean(dim=(1, 2, 3))
        with torch.inference_mode():
            affine = torch.nn.functional.linear(f, head.proj.weight, head.proj.bias)
        torch.testing.assert_close(affine, projected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(self.native, projected, atol=2e-5, rtol=2e-5)
        if head.activation is not None:
            raise Blocker('Unexpected native activation')
        return dict(strict_loading=True, shape=list(f.shape), projection_shape=list(self.projection_input.shape),
                    native_mean_max_difference=float((self.native-projected).abs().max().cpu()),
                    affine_mean_max_difference=float((affine-projected).abs().max().cpu()), native_projection_executed=True)


def decode(encoded, entry, positions=None):
    """Fresh complete decode and actual timestamp planning, including inside timing."""
    import av
    from research_utils import configure_decoder
    from temporal_refresh_common import snippet_grid
    from ucf101_proxy_data import presentation_order
    frames, pts, identities = [], [], []
    with av.open(io.BytesIO(encoded)) as container:
        stream = container.streams.video[0]; configure_decoder(stream)
        for i, f in enumerate(container.decode(stream)):
            if f.pts is None:
                raise Blocker('Previously screened input lost PTS')
            frames.append(f); pts.append(float(f.pts*f.time_base))
            identities.append(dict(index=i, pts=int(f.pts), time_base=str(f.time_base)))
        frames, pts, identities, reordered = presentation_order(frames, pts, identities)
        grid = snippet_grid(pts)
        grid['frame_identities'] = [[identities[j] for j in row] for row in grid['frame_indices']]
        if grid != entry['snippet_grid'] or len(frames) != entry['decoded_frame_count']:
            raise Blocker('Runtime sampling differs from screened real timestamps')
        positions = list(range(16)) if positions is None else positions
        needed = sorted(set(j for t in positions for j in grid['frame_indices'][t]))
        rgb = {j: frames[j].to_ndarray(format='rgb24') for j in needed}
    return rgb, grid, len(frames)


def read_features(entry, role):
    ledger = read(OUT/f'{role}_collection.json')
    r = ledger[entry['clip']]
    if r['input_sha256'] != entry['sha256'] or r['grid_sha256'] != hashlib.sha256(json.dumps(entry['snippet_grid'], sort_keys=True).encode()).hexdigest():
        raise Blocker('Feature source provenance mismatch')
    p = ROOT/r['path']
    if not p.exists() or sha(p) != r['sha256']:
        raise Blocker(f'Real arrays missing or changed: {p}; no reconstruction from means permitted')
    with np.load(p, allow_pickle=False) as f:
        a = f['x3d_xs']
    if a.shape != (16, 2048) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise Blocker('Invalid genuine feature array')
    return a


def collect(entries, role, encoder):
    if role == 'test':
        seal_check(True)
    path = OUT/f'{role}_collection.json'
    ledger = read(path) if path.exists() else {}
    encoder.phase = role
    for number, v in enumerate(entries):
        if v['role'] != role:
            raise ValueError('Wrong collection role')
        if v['clip'] in ledger:
            read_features(v, role)
            continue
        if role == 'test':
            # Code/static bytes verified once per process; input bytes every clip.
            if not (OUT/'evaluation_exposure.json').exists():
                raise Blocker('Test exposure missing')
        raw = (ROOT/v['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != v['sha256']:
            raise Blocker('Video bytes changed')
        encoder.clip = v['clip']
        start = time.perf_counter()
        frames, grid, count = decode(raw, v)
        with encoder.torch.inference_mode():
            a = encoder.torch.cat([encoder.feature([frames[j] for j in grid['frame_indices'][t]], t)
                                   for t in range(16)]).cpu().numpy()
        encoder.torch.cuda.synchronize()
        p = DATA/'features'/f'{v["clip"]}.npz'; p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, x3d_xs=a)
        ledger[v['clip']] = dict(path=rel(p), sha256=sha(p), input_sha256=v['sha256'],
            grid_sha256=hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest(),
            encoder_identity_sha256=sha(OUT/'encoder_identity.json'), collected_at_utc=now(),
            encoder_starts=16, encoder_completions=16, decoded_frames=count, seconds=time.perf_counter()-start)
        save(path, ledger)
        if (number+1) % 25 == 0:
            print(f'Actual X3D {role} collection {number+1}/{len(entries)}', flush=True)
    return ledger


def smoke(manifest, encoder):
    p = OUT/'training_smoke.json'
    if p.exists():
        return read(p)
    entries = sorted([v for v in manifest['videos'] if v['role'] == 'train'], key=lambda v: rank('smoke', v['clip']))[:2]
    start = time.perf_counter()
    # Smoke arrays are retained as genuine training data, not recomputed for bulk.
    ledger = collect(entries, 'train', encoder)
    seconds = time.perf_counter()-start
    native = encoder.native_check()
    size = sum((ROOT/ledger[v['clip']]['path']).stat().st_size for v in entries)
    result = dict(actual_training_clips=[v['clip'] for v in entries], encoder_calls=32, elapsed_seconds=seconds,
                  projected_collection_seconds=seconds/2*len(manifest['videos']),
                  projected_feature_bytes=size/2*len(manifest['videos']), free_disk_bytes=__import__('shutil').disk_usage(ROOT).free,
                  native_head_verification=native, no_test_features=True, recorded_at_utc=now())
    save(p, result)
    if result['projected_feature_bytes']*2 > result['free_disk_bytes']:
        raise Blocker('Training smoke projects insufficient feature storage')
    return result


def timing(manifest, encoder, head):
    """30 distinct class/group units; repeats remain nested within clips."""
    seal_check(True)
    import gc
    import random
    import pandas as pd
    from temporal_refresh_timing import working_set
    torch = encoder.torch
    gpu_head = {k: torch.tensor(v, dtype=torch.float64, device='cuda') for k, v in head.items()}
    progress = OUT/'timing_progress.json'
    state = read(progress) if progress.exists() else dict(completed=[], trials=[], loads=[])
    entries = {v['clip']: v for v in manifest['videos']}
    encoder.log = False
    try:
        with torch.inference_mode():
            for clip in manifest['timing_clips']:
                if clip in state['completed']:
                    continue
                v = entries[clip]
                start = time.perf_counter(); raw = (ROOT/v['path']).read_bytes(); load_sec = time.perf_counter()-start
                if hashlib.sha256(raw).hexdigest() != v['sha256']:
                    raise Blocker('Timing input hash changed')
                reference = read_features(v, 'test')
                rng = random.Random(int(rank('timing_order', clip)[:16], 16))
                trials = []
                for repetition in range(-2, 5):
                    modes = ['dense16', 'periodicB4']; rng.shuffle(modes)
                    for mode in modes:
                        positions = list(range(16)) if mode == 'dense16' else [0, 4, 8, 12]
                        encoder.clip = clip; encoder.phase = 'timing_warmup' if repetition < 0 else 'timing_measured'
                        gc.collect(); torch.cuda.synchronize()
                        resident = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                        cpu_before, _ = working_set()
                        counts = [encoder.starts, encoder.completions]
                        token = f'{clip}:{repetition}:{mode}:{time.time_ns()}'
                        with (OUT/'timing_attempts.jsonl').open('a', encoding='utf-8') as log:
                            log.write(json.dumps(dict(event='start', token=token, at_utc=now(), expected_calls=len(positions)))+'\n')
                        start = time.perf_counter()
                        frames, grid, decoded = decode(raw, v, positions)
                        features = [encoder.feature([frames[j] for j in grid['frame_indices'][t]], t) for t in positions]
                        weights = np.diff(positions+[16])/16
                        pooled = sum(float(w)*f.to(torch.float64) for w, f in zip(weights, features))
                        # Head/scalers reside on GPU before boundary; native400 overhead still runs.
                        z = ((pooled-gpu_head['means'])/gpu_head['scales']) @ gpu_head['weights'].T+gpu_head['bias']
                        prediction = z.argmax(1)
                        torch.cuda.synchronize(); elapsed = time.perf_counter()-start
                        actual = z.cpu().numpy()[0]
                        ref = pool_logits(np.sum(reference[positions].astype(np.float64)*weights[:, None], axis=0), head)
                        np.testing.assert_allclose(actual, ref, atol=1e-6, rtol=1e-6)
                        cpu_after, cpu_peak = working_set()
                        trials.append(dict(clip=clip, label=v['label'], group=v['group'], mode=mode, repetition=repetition,
                            warmup=repetition < 0, seconds=elapsed, decoded_frames=decoded, rgb_conversions=len(frames),
                            encoder_starts=encoder.starts-counts[0], encoder_completions=encoder.completions-counts[1],
                            resident_cuda_bytes=resident, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                            incremental_peak_cuda_bytes=max(0, torch.cuda.max_memory_allocated()-resident),
                            process_working_set_before=cpu_before, process_working_set_after=cpu_after,
                            process_lifetime_peak_working_set=cpu_peak, prediction=int(prediction.cpu()),
                            reference_logit_max_difference=float(np.max(np.abs(actual-ref))), logits=actual.tolist()))
                        with (OUT/'timing_attempts.jsonl').open('a', encoding='utf-8') as log:
                            log.write(json.dumps(dict(event='complete', token=token, at_utc=now(), calls=len(positions)))+'\n')
                        del frames, features, pooled, z, prediction
                state['trials'].extend(trials); state['completed'].append(clip)
                state['loads'].append(dict(clip=clip, disk_loading_seconds=load_sec, bytes=len(raw)))
                save(progress, state)
                print(f'Real timing complete {len(state["completed"])}/30 groups', flush=True)
    finally:
        encoder.log = True
    pd.DataFrame(state['trials']).to_csv(OUT/'timing_trials.csv', index=False, float_format='%.17g')
    measured = pd.DataFrame([r for r in state['trials'] if not r['warmup']])
    medians = measured.groupby(['label', 'group', 'clip', 'mode']).seconds.agg(['median', 'min', 'max', 'std']).reset_index()
    medians.to_csv(OUT/'timing_per_clip.csv', index=False, float_format='%.17g')
    ratios = medians.pivot(index=['label', 'group', 'clip'], columns='mode', values='median')
    ratios['dense_over_periodic_ratio'] = ratios.dense16/ratios.periodicB4
    ratios.to_csv(OUT/'timing_paired_ratios.csv', float_format='%.17g')
    save(OUT/'timing_summary.json', dict(groups=len(ratios), clips=len(ratios),
        dense_median_seconds=float(ratios.dense16.median()), periodic_median_seconds=float(ratios.periodicB4.median()),
        median_paired_ratio=float(ratios.dense_over_periodic_ratio.median()),
        boundary='encoded bytes in memory -> fresh full decode + actual PTS plan -> selected RGB/transfer -> batch1 FP32 encoder/native400 -> pooled new head -> CUDA-completed logits and prediction',
        repeats='two warmups, five interleaved measurements/mode/clip, one CPU thread; trials not independent sources',
        memory_scope='torch allocated resident/incremental peak; process working set and lifetime peak, not exact per-trial CPU peak',
        offline_selectors='All proxy selectors require 16 dense encodings PLUS exhaustive CPU search; periodic cost is not oracle latency'))


def pool_logits(f, head):
    return ((f-head['means'])/head['scales']) @ np.asarray(head['weights']).T+head['bias']


