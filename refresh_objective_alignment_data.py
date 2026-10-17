"""Identity-gated input inventory, technical screening and sealed collection.

No historical outcome files are used for exclusions. Network and encoder work
are reachable only through explicit, resumable stages in the new runner.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

import numpy as np
from temporal_refresh_common import ROOT, REPS, ASSETS as OLD_ASSETS, now, read, save, sha, relative, inspect_encoded, Budget
from refresh_objective_alignment_utils import verify_seal

ASSETS = ROOT/'data/source/refresh_objective_alignment_v1'


def ranking(source: str, seed: int) -> str:
    return hashlib.sha256(f'refresh_objective_alignment_v1|{seed}|{source}'.encode()).hexdigest()


def inspect_technical(path: Path) -> dict:
    """Retain the v1 criteria; explicitly recognize a missing video stream."""
    try:
        return inspect_encoded(path)
    except IndexError:
        import av
        with av.open(str(path)) as container:
            if len(container.streams.video) != 0:
                raise
        raise ValueError('Encoded candidate contains no video stream') from None


def inventory(output: Path, old: Path, classes: list[str], seed: int) -> dict:
    path = output/'sampling_plan.json'
    if path.exists():
        plan = read(path)
        assert plan['seed'] == seed
        for name, digest in plan['annotation_identities'].items():
            assert sha(ROOT/name) == digest
        return plan
    from temporal_refresh_data import historical_exclusions
    excluded = historical_exclusions(output)  # membership documents only
    train = OLD_ASSETS/'k400_train.csv'
    val = ROOT/'data/source/cache_drift_v2/val.csv'
    urls = ROOT/'data/source/cache_drift_v2/k400_val_path.txt'
    provenance = read(old/'annotation_provenance.json')
    assert sha(train) == provenance['official_train']['sha256']
    assert sha(val) == provenance['official_val']['sha256']
    train_ids = {row['youtube_id'] for row in csv.DictReader(train.open(encoding='utf-8'))}
    blocked = set(excluded['source_ids']) | train_ids
    rows = []
    for row in csv.DictReader(val.open(encoding='utf-8')):
        if row['label'] not in classes or row['youtube_id'] in blocked:
            continue
        source = row['youtube_id']
        rows.append({**row, 'source_video_id': source,
                     'filename': f"{source}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4",
                     'rank': ranking(source, seed)})
    assert len({r['source_video_id'] for r in rows}) == len(rows)
    local = {p.name: relative(p) for p in sorted((ROOT/'data/source').rglob('*.mp4'))}
    archives = []
    for url in urls.read_text().splitlines():
        name = PurePosixPath(url).name
        choices = [ROOT/'data/source/cache_drift_v2/shards'/name, OLD_ASSETS/'shards'/name]
        record = {'url': url, 'path': relative(ASSETS/'shards'/name), 'cached': False}
        for candidate in choices:
            for sidecar in [candidate.with_suffix('.download.json'), candidate.with_name(name+'.identity.json')]:
                if candidate.exists() and sidecar.exists():
                    identity = read(sidecar)
                    assert identity['url'] == url
                    record = dict(url=url, path=relative(candidate), cached=True,
                                  bytes=candidate.stat().st_size, sha256=identity['sha256'])
                    break
            if record['cached']:
                break
        archives.append(record)
    # Traversal is stable and does not determine cohort ranking.
    archives.sort(key=lambda a: (not a['cached'], a['url']))
    plan = dict(created_at_utc=now(), seed=seed, classes=classes,
                annotation_identities={relative(p): sha(p) for p in [train, val, urls]},
                previous_sources=len(excluded['source_ids']), previous_hashes=len(excluded['encoded_sha256']),
                official_training_parent_count=len(train_ids), prior_exclusion_sha256=sha(output/'source_exclusions.json'),
                candidate_rows=sorted(rows, key=lambda r: (r['rank'], r['source_video_id'])),
                candidate_counts=dict(Counter(r['label'] for r in rows)),
                existing_candidate_paths={r['filename']: local[r['filename']] for r in rows if r['filename'] in local},
                archives=archives, cached_archive_bytes=sum(a.get('bytes', 0) for a in archives),
                free_disk_bytes=shutil.disk_usage(ROOT).free,
                estimated_uncached_archive_bytes=(len(archives)-sum(a['cached'] for a in archives))*1_650_000_000,
                ranking_rule='SHA256(UTF8(refresh_objective_alignment_v1|seed|original_source_id)), ascending; class quotas after input-only screening',
                archive_rule='verified cached first then URL lexicographic; traversal never changes source rank',
                retry_rule='same archive, at most three whole-body attempts, count received and uncertain bytes; never replace source for machine/CUDA/network failure',
                rights='No videos redistributed; CVDF/Kinetics public-access provenance does not establish individual video licensing',
                unresolved='Parent exclusion handles adjacent windows, hashes exact bytes; reuploads, near duplicates and all checkpoint pretraining cannot be proved absent')
    save(path, plan)
    return plan


def screen(output: Path, plan: dict, precision: dict, cap_gib: float) -> dict:
    """No RGB or recognition inference here. Decode only technical candidates."""
    progress = output/'eligibility.json'
    state = read(progress) if progress.exists() else dict(events={}, paths=plan['existing_candidate_paths'].copy(), archives=[], eligible={})
    if state.get('complete') and (output/'precision_and_feasibility.json').exists():
        return read(output/'precision_and_feasibility.json')
    budget = Budget(output/'download_accounting.json', int(cap_gib*2**30))
    excluded = read(output/'source_exclusions.json')
    excluded_hashes = set(excluded['encoded_sha256'])
    names = {r['filename']: r for r in plan['candidate_rows']}
    hashes = {v['file_identity']['sha256']: s for s, v in state['eligible'].items()}

    def inspect_available():
        for row in plan['candidate_rows']:
            source, name = row['source_video_id'], row['filename']
            if source in state['events'] or name not in state['paths']:
                continue
            path = ROOT/state['paths'][name]
            digest = sha(path)
            event = {'source_video_id': source, 'label': row['label'], 'rank': row['rank'],
                     'path': relative(path), 'sha256': digest, 'inspected_at_utc': now()}
            try:
                if digest in excluded_hashes or (digest in hashes and hashes[digest] != source):
                    raise ValueError('identical encoded bytes overlap historical cohort or another candidate')
                technical = inspect_technical(path)
            except Exception as exc:
                # Technical input failures only; infrastructure errors propagate.
                import av
                if not isinstance(exc, (ValueError, av.error.FFmpegError)):
                    raise
                event.update(eligible=False, reason=f'{type(exc).__name__}: {exc}')
            else:
                state['eligible'][source] = {**row, 'video_id': path.stem, 'relative_path': relative(path),
                    'recognition_role': 'confirmation', 'original_dataset_split': 'official Kinetics400 validation',
                    'file_identity': {'sha256': digest, 'size_bytes': path.stat().st_size}, **technical}
                hashes[digest] = source
                event.update(eligible=True, reason='input-only technical checks; no encoder/classifier inference')
            state['events'][source] = event
            save(progress, state)

    inspect_available()
    for archive in plan['archives']:
        if archive['url'] in state['archives']:
            continue
        missing = set(names)-set(state['paths'])
        if not missing:
            break
        path = ROOT/archive['path']
        print('Inventory archive', path.name, 'remaining unlocated', len(missing), flush=True)
        if not archive['cached']:
            budget.download(archive['url'], path)
        else:
            assert path.stat().st_size == archive['bytes'] and sha(path) == archive['sha256']
        with tarfile.open(path, mode='r|gz') as tar:
            for member in tar:
                name = PurePosixPath(member.name).name
                if not member.isfile() or name not in missing:
                    continue
                dest = ASSETS/'candidates'/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_suffix('.partial')
                with tar.extractfile(member) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(dest)
                state['paths'][name] = relative(dest)
        state['archives'].append(archive['url'])
        save(progress, state)
        inspect_available()
    for row in plan['candidate_rows']:
        if row['source_video_id'] not in state['events']:
            state['events'][row['source_video_id']] = dict(source_video_id=row['source_video_id'], label=row['label'],
                eligible=False, reason='not found in accessible frozen archive pool', rank=row['rank'])
    counts = {c: sum(v['label'] == c for v in state['eligible'].values()) for c in plan['classes']}
    m = min(counts.values())
    per_class = min(precision['N_requested']//20, 20, m)
    n = per_class*20
    result = {**precision, 'eligible_counts': counts, 'smallest_eligible_class_count': m,
        'n_per_class': per_class, 'N_frozen': n if per_class >= 5 else None,
        'confirmation_feasible': per_class >= 5, 'precision_limited': n < precision['N_requested'],
        'predicted_halfwidth_achieved': float(1.96*np.sqrt(precision['V']/n)) if n else None,
        'minimum_five_per_class_shortfall': {c: max(0, 5-counts[c]) for c in counts if counts[c] < 5},
        'candidate_count': len(names), 'technically_screened': sum('inspected_at_utc' in e for e in state['events'].values()),
        'eligible_total': sum(counts.values()), 'new_received_bytes': budget.state['actual_body_bytes'],
        'new_uncertain_bytes_charged': budget.state['interruption_charge_bytes'],
        'cap_bytes': budget.state['cap_bytes'], 'cached_archive_bytes': plan['cached_archive_bytes'],
        'outcomes_used_for_membership': False, 'screening_complete_at_utc': now()}
    state['complete'] = True
    save(progress, state)
    save(output/'precision_and_feasibility.json', result)
    if per_class >= 5:
        videos = []
        for c in plan['classes']:
            available = sorted((v for v in state['eligible'].values() if v['label'] == c), key=lambda v: (v['rank'], v['source_video_id']))
            videos.extend(available[:per_class])
        assert len({v['source_video_id'] for v in videos}) == n
        assert len({v['file_identity']['sha256'] for v in videos}) == n
        frozen = output/'confirmation_manifest.json'
        contents = dict(classes=plan['classes'], videos=videos, frozen_at_utc=now(), seed=plan['seed'],
                        membership_rule='first n_per_class eligible in frozen source-ID hash order')
        if frozen.exists():
            assert read(frozen)['videos'] == videos
        else:
            save(frozen, contents)
    return result


def collect(output: Path, old: Path, entries: list[dict], device: str) -> dict:
    """Actual unchanged v1 encoders, new arrays/ledger only, strict own seal."""
    seal = read(output/'evaluation_seal.json')
    verify_seal(ROOT, seal)
    assert entries == read(output/'confirmation_manifest.json')['videos']
    ledger_path = output/'confirmation_collection.json'
    ledger = read(ledger_path) if ledger_path.exists() else {'videos': {}}
    if len(ledger['videos']) == len(entries):
        for v in entries:
            saved = ledger['videos'][v['source_video_id']]
            assert sha(ROOT/saved['path']) == saved['sha256']
        return ledger
    # No package changes or downloads. Existing separate environment required.
    for name in ['dependency_setup.json', 'encoder_provenance.json']:
        path = output/name
        if not path.exists():
            shutil.copyfile(old/name, path)
        assert sha(path) == sha(old/name)
    from temporal_refresh_encoders import Encoders, load_rgb_frames
    import torch
    encoders = Encoders(output, device)
    import subprocess
    primitive = (torch.tensor([1., 2.], device='cuda')**2).sum()
    torch.cuda.synchronize()
    save(output/'collection_environment.json', dict(recorded_at_utc=now(),
        torch_version=torch.__version__, torch_version_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(), GPU=torch.cuda.get_device_name(),
        nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
        small_GPU_operation_result=float(primitive.cpu()), checkpoint_and_numerics=encoders.provenance,
        driver_is_distinct_from_torch_CUDA=True, historical_environment_unchanged=True))
    encoders.phase = 'objective_alignment_confirmation'
    for i, entry in enumerate(entries):
        verify_seal(ROOT, seal)
        source = entry['source_video_id']
        if source in ledger['videos']:
            assert sha(ROOT/ledger['videos'][source]['path']) == ledger['videos'][source]['sha256']
            continue
        raw = (ROOT/entry['relative_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['file_identity']['sha256']
        start = time.perf_counter()
        with (output/'collection_events.jsonl').open('a', encoding='utf-8') as event_log:
            event_log.write(json.dumps(dict(source_video_id=source, event='decode_start', at_utc=now()))+'\n')
        frames = load_rgb_frames(raw, entry)
        before = dict(encoders.counts)
        arrays = {}
        with torch.inference_mode():
            for rep in REPS:
                values = [encoders.features([frames[j] for j in indices], rep) for indices in entry['snippet_grid']['frame_indices']]
                array = torch.cat(values, dim=0).cpu().numpy()
                assert array.dtype == np.float32 and array.shape == (16, 2048 if rep == 'x3d_xs' else 512)
                if not np.isfinite(array).all():
                    raise RuntimeError(f'Nonfinite feature collection: {source}; preserve membership and diagnose')
                arrays[rep] = array
        path = ASSETS/'features'/f'{source}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        ledger['videos'][source] = dict(path=relative(path), sha256=sha(path), input_sha256=entry['file_identity']['sha256'],
            snippet_grid_sha256=hashlib.sha256(json.dumps(entry['snippet_grid'], sort_keys=True).encode()).hexdigest(),
            encoder_provenance_sha256=sha(old/'encoder_provenance.json'), collected_at_utc=now(),
            seconds=time.perf_counter()-start, forward_calls={k: v-before.get(k, 0) for k, v in encoders.counts.items() if v-before.get(k, 0)})
        save(ledger_path, ledger)
        with (output/'collection_events.jsonl').open('a', encoding='utf-8') as event_log:
            event_log.write(json.dumps(dict(source_video_id=source, event='retained_complete', at_utc=now(),
                decoded_frames=entry['decoded_frame_count'], forward_calls=ledger['videos'][source]['forward_calls']))+'\n')
        del arrays, frames, values
        if (i+1) % 10 == 0:
            print(f'Actual confirmation collection {i+1}/{len(entries)}', flush=True)
    encoders.audit_handle.close()
    verify_seal(ROOT, seal)
    return ledger
