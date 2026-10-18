"""Official split-1 population, input-only screening and duplicate quarantine."""
from __future__ import annotations
from collections import Counter
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
import zipfile
import numpy as np
from ucf101_proxy_common import *
from temporal_refresh_common import snippet_grid

NAME = re.compile(r'^v_(?P<label>.+)_(?P<group>g\d{2})_c\d{2}\.avi$')


def presentation_order(frames, pts, identities):
    """Order REAL unique PTS, retaining decode ordinals; no timestamp synthesis."""
    a = np.asarray(pts, dtype=np.float64)
    if not len(a) or not np.isfinite(a).all() or len(set(a)) != len(a):
        raise ValueError('Missing/nonfinite/duplicate actual presentation timestamps')
    order = np.argsort(a, kind='stable')
    ordered_ids = [{**identities[int(j)], 'presentation_index': i} for i, j in enumerate(order)]
    return [frames[int(j)] for j in order], a[order].tolist(), ordered_ids, int(np.sum(order != np.arange(len(a))))


def parse_metadata(archive):
    with zipfile.ZipFile(archive) as z:
        names = ['classInd.txt', 'trainlist01.txt', 'testlist01.txt']
        raw = {n: z.read('ucfTrainTestlist/'+n) for n in names}
    classes = []
    for line in raw['classInd.txt'].decode().splitlines():
        if not line.strip():
            continue
        index, label = line.split()
        if int(index) != len(classes)+1:
            raise ValueError('Noncontiguous official class index')
        classes.append(label)
    if len(classes) != 101 or len(set(classes)) != 101:
        raise ValueError('Official population must have all 101 classes')
    rows, roles, seen = [], {}, set()
    for name, role in [('trainlist01.txt', 'official_train'), ('testlist01.txt', 'test')]:
        for line in raw[name].decode().splitlines():
            if not line.strip():
                continue
            bits = line.split()
            path = PurePosixPath(bits[0])
            match = NAME.fullmatch(path.name)
            canonical = path.parent.as_posix()
            if not match or canonical.casefold() != match['label'].casefold() or canonical not in classes:
                raise ValueError(f'Invalid official filename: {line}')
            if len(bits) > 1 and int(bits[1]) != classes.index(canonical)+1:
                raise ValueError('Official train label mismatch')
            key = (canonical, match['group'])
            if key in roles and roles[key] != role:
                raise ValueError(f'Official split GROUP leakage: {key}')
            if bits[0] in seen:
                raise ValueError('Duplicate official clip')
            seen.add(bits[0]); roles[key] = role
            rows.append(dict(label=canonical, filename_class=match['label'], group=match['group'], clip=path.name,
                             official_path=bits[0], official_role=role))
    dev = set()
    for c in classes:
        groups = sorted([g for (label, g), role in roles.items() if label == c and role == 'official_train'],
                        key=lambda g: (rank(c, g), g))
        if len(groups) < 13:
            raise Blocker(f'Too few official training groups for {c}')
        dev.update((c, g) for g in groups[:3])
    for v in rows:
        v['role'] = 'test' if v['official_role'] == 'test' else ('development' if (v['label'], v['group']) in dev else 'train')
        v['clip_rank'] = rank('clip', v['label'], v['group'], v['clip'])
        v['group_rank'] = rank(v['label'], v['group'])
    return dict(classes=classes, clips=sorted(rows, key=lambda v: (v['label'], v['group'], v['clip_rank'])),
                metadata_members={n: hashlib.sha256(b).hexdigest() for n, b in raw.items()},
                groups_by_role=dict(Counter(roles[k] if k not in dev else 'development' for k in roles)),
                role_rule='SHA256(UTF8(str(20261026)+class+gXX)); three smallest official train groups/class; clip rank has separate clip prefix',
                official_group_isolation_verified=True, created_at_utc=now())


def metadata():
    path = OUT/'population_plan.json'
    if path.exists():
        plan = read(path)
        if sha(DATA/'official_splits.zip') != plan['archive_sha256']:
            raise Blocker('Official metadata bytes changed')
        return plan
    page, archive = DATA/'official_page.html', DATA/'official_splits.zip'
    if not page.exists() or not archive.exists():
        network = Network()
        page = network.get(PAGE_URL, page, 2*1024**2)
        archive = network.get(SPLIT_URL, archive, 2*1024**2)
    for p, url in [(page, PAGE_URL), (archive, SPLIT_URL)]:
        identity = read(p.with_name(p.name+'.identity.json'))
        if identity['url'] != url or identity['sha256'] != sha(p):
            raise Blocker('Official metadata identity mismatch')
    html = page.read_text(encoding='utf-8', errors='replace')
    if 'UCF101/UCF101.rar' not in html or 'UCF101TrainTestSplits-RecognitionTask.zip' not in html:
        raise Blocker('Official download links changed; review access before acquisition')
    plan = parse_metadata(archive)
    plan['archive_sha256'] = sha(archive)
    save(path, plan)
    save(OUT/'official_metadata_identities.json', dict(page_url=PAGE_URL, page_sha256=sha(page),
        split_url=SPLIT_URL, split_sha256=sha(archive), members=plan['metadata_members'],
        video_url=DATA_URL, version='UCF101 official 2012 archive, official split 1',
        video_HTTP_identity='Last-Modified Wed, 21 Nov 2012 19:40:04 GMT; Content-Length 6932971618; ETag 50ad2e14-19d3cc062 observed by verified-TLS HEAD',
        access='Public direct official links. No account, click-through or new terms accepted.',
        terms='Official page requests dataset technical-report citation. No explicit video redistribution license found on downloaded page; original video rights remain separate. Research analysis only; no video redistribution.',
        citation='Soomro, Zamir and Shah, UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild, CRCV-TR-12-01, 2012.'))
    return plan


def acquire():
    path = DATA/'UCF101.rar'
    if not path.exists():
        network = Network()
        partial = path.with_name(path.name+'.partial')
        expected = 6932971618
        # Range access avoids the server's observed truncation of one huge response.
        # Verify received prefix against a fresh official ranged response before reuse.
        if partial.exists() and partial.stat().st_size:
            prefix = network.get(DATA_URL, DATA/'segments/prefix.bin', 4096, '0-4095')
            with partial.open('rb') as stream:
                if stream.read(4096) != prefix.read_bytes():
                    raise Blocker('Interrupted archive prefix does not match official bytes')
        else:
            partial.parent.mkdir(parents=True, exist_ok=True); partial.touch()
        segments = []
        while partial.stat().st_size < expected:
            lo = partial.stat().st_size; hi = min(lo+128*1024**2, expected)-1
            segment = network.get(DATA_URL, DATA/f'segments/{lo}-{hi}.bin', hi-lo+1, f'{lo}-{hi}')
            with partial.open('ab') as dest, segment.open('rb') as source:
                shutil.copyfileobj(source, dest)
            segments.append(dict(range=f'{lo}-{hi}', sha256=sha(segment)))
            print(f'Official archive ranged acquisition {partial.stat().st_size}/{expected}', flush=True)
        if partial.stat().st_size != expected:
            raise Blocker('Unexpected archive size')
        partial.replace(path)
        save(path.with_name(path.name+'.identity.json'), dict(url=DATA_URL, sha256=sha(path), bytes=expected,
            downloaded_at_utc=now(), transport='Windows verified-TLS curl; validated Content-Range; initial partial prefix checked',
            resumed_segments=segments, no_new_terms_accepted=True))
    if not path.with_name(path.name+'.identity.json').exists():
        # A user interruption can land after atomic final rename but before the
        # large-file hash is written. Recover from our completed official range
        # ledger; full archive extraction must still pass CRC before screening.
        ledger_path = OUT/'network_accounting.json'
        if not ledger_path.exists() or path.stat().st_size != 6932971618:
            raise Blocker('Dataset bytes lack recoverable acquisition identity')
        requests = read(ledger_path)['requests']
        if not any(r.get('status') == 'complete' and r.get('url') == DATA_URL
                   and r.get('byte_range', '').endswith('-6932971617') for r in requests if r.get('byte_range')):
            raise Blocker('No completed official final-range evidence')
        prefix = DATA/'segments/prefix.bin'
        identity = prefix.with_name(prefix.name+'.identity.json')
        if not prefix.exists() or not identity.exists() or sha(prefix) != read(identity)['sha256']:
            raise Blocker('Interrupted finalization lacks verified official prefix')
        with path.open('rb') as source:
            if source.read(4096) != prefix.read_bytes():
                raise Blocker('Completed archive prefix mismatch')
        save(path.with_name(path.name+'.identity.json'), dict(url=DATA_URL, sha256=sha(path),
            bytes=path.stat().st_size, recovered_at_utc=now(),
            recovery='User interrupted after final rename, before identity hash; complete official range ledger and verified prefix retained. Full extraction CRC remains mandatory before input screening.'))
    if sha(path) != read(path.with_name(path.name+'.identity.json'))['sha256']:
        raise Blocker('Dataset archive hash mismatch')
    save(OUT/'official_video_identity.json', read(path.with_name(path.name+'.identity.json')))
    dest = DATA/'videos'
    marker = DATA/'extraction.json'
    if not marker.exists():
        if shutil.disk_usage(ROOT).free < 20*1024**3:
            raise Blocker('Need 20 GiB free for safe complete extraction and features')
        listing = subprocess.check_output(['tar.exe', '-tf', str(path)], text=True)
        for name in listing.splitlines():
            p = PurePosixPath(name.replace('\\', '/'))
            if p.is_absolute() or '..' in p.parts or ':' in name:
                raise Blocker('Unsafe archive member')
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar.exe', '-xf', str(path), '-C', str(dest)], check=True)
        save(marker, dict(archive_sha256=sha(path), members=len(listing.splitlines()),
                          extracted_at_utc=now(), destination=rel(dest)))
    return dest


def inspect_video(path):
    """Full native AVI/codec decode, valid actual PTS; no duration/codec filter."""
    import av
    from research_utils import configure_decoder
    start = time.perf_counter()
    pts, identities, frames = [], [], []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('No video stream')
        stream = container.streams.video[0]
        configure_decoder(stream)
        codec = stream.codec_context.name
        resolution = [stream.codec_context.width, stream.codec_context.height]
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Missing PTS/timebase')
            pts.append(float(frame.pts*frame.time_base))
            identities.append(dict(index=i, pts=int(frame.pts), time_base=str(frame.time_base)))
            frames.append(frame)
        fps = float(stream.average_rate) if stream.average_rate else None
        frames, pts, identities, reordered = presentation_order(frames, pts, identities)
        grid = snippet_grid(pts)
        grid['frame_identities'] = [[identities[j] for j in row] for row in grid['frame_indices']]
        # Input-only temporal fingerprints, independent of learned features/outcomes.
        from scipy.fft import dctn
        from PIL import Image
        fingerprints = []
        for q in [.1, .5, .9]:
            frame = frames[int(q*(len(frames)-1))]
            grey = Image.fromarray(frame.to_ndarray(format='rgb24')).convert('L').resize((32, 32))
            low = dctn(np.asarray(grey, dtype=np.float64), norm='ortho')[:8, :8].flatten()[1:]
            fingerprints.append(''.join('1' if v > np.median(low) else '0' for v in low))
    return dict(codec=codec, resolution=resolution, reported_fps=fps,
                decoded_frame_count=len(pts), duration_pts_span=pts[-1]-pts[0], snippet_grid=grid,
                perceptual_fingerprint=fingerprints, inspection_seconds=time.perf_counter()-start,
                PTS_presentation_reordered_frames=reordered,
                PTS_policy='sort actual unique PTS for native AVI presentation order, retain original decode ordinal; never synthesize/drop/pad frames',
                decoder=dict(thread_count=1, thread_type='NONE', skip_frame='NONE'))


def require_population(videos, classes):
    counts = {}
    for role, minimum in [('train', 10), ('development', 2), ('test', 3)]:
        counts[role] = {}
        for c in classes:
            group_counts = Counter(v['group'] for v in videos if v['role'] == role and v['label'] == c)
            if any(n != 2 for n in group_counts.values()):
                raise ValueError('Each included group requires exactly two clips')
            counts[role][c] = len(group_counts)
            if len(group_counts) < minimum:
                raise Blocker(f'Population minimum failed: {role} {c}: {len(group_counts)} < {minimum}')
    assigned = {}
    for v in videos:
        key = (v['label'], v['group'])
        if key in assigned and assigned[key] != v['role']:
            raise ValueError('Cross-role group leakage')
        assigned[key] = v['role']
    return counts


def eligibility(plan, directory):
    path = OUT/'eligible_manifest.json'
    if path.exists():
        manifest = read(path)
        require_population(manifest['videos'], plan['classes'])
        return manifest
    import av
    progress = OUT/'eligibility_progress.json'
    state = read(progress) if progress.exists() else dict(plan_sha256=sha(OUT/'population_plan.json'), attempts={}, groups={})
    if state['plan_sha256'] != sha(OUT/'population_plan.json'):
        raise Blocker('Population changed during screening')
    lookup = {p.name: p for p in directory.rglob('*.avi')}
    if len(lookup) != len(list(directory.rglob('*.avi'))):
        raise Blocker('Nonunique official filenames in extracted dataset')
    grouped = {}
    for row in plan['clips']:
        grouped.setdefault((row['label'], row['group']), []).append(row)
    # Finish all training input screening before perceptual threshold calibration.
    ordered = sorted(grouped, key=lambda k: ({'train': 0, 'development': 1, 'test': 2}[grouped[k][0]['role']], k))
    for number, key in enumerate(ordered):
        name = '/'.join(key)
        if name in state['groups']:
            continue
        selected = []
        for row in sorted(grouped[key], key=lambda v: (v['clip_rank'], v['clip'])):
            if len(selected) == 2:
                break
            clip = row['clip']
            if clip not in state['attempts']:
                try:
                    if clip not in lookup:
                        raise ValueError('Official listed clip missing from archive')
                    p = lookup[clip]
                    info = inspect_video(p)
                    state['attempts'][clip] = dict(eligible=True, video={**row, **info,
                        'path': rel(p), 'sha256': sha(p), 'bytes': p.stat().st_size}, inspected_at_utc=now())
                except (ValueError, av.error.FFmpegError) as exc:
                    state['attempts'][clip] = dict(eligible=False, reason=f'{type(exc).__name__}: {exc}', inspected_at_utc=now())
                with (OUT/'technical_inspection_events.jsonl').open('a', encoding='utf-8') as log:
                    log.write(__import__('json').dumps(dict(clip=clip, **state['attempts'][clip]))+'\n')
            event = state['attempts'][clip]
            if event['eligible']:
                selected.append(event['video'])
        state['groups'][name] = dict(included=len(selected) == 2, videos=selected if len(selected) == 2 else [],
                                    reason=None if len(selected) == 2 else 'Fewer than two technically eligible clips; whole group excluded')
        if (number+1) % 25 == 0 or number+1 == len(ordered):
            save(progress, state)
        if (number+1) % 50 == 0:
            print(f'Input-only group screening {number+1}/{len(ordered)}', flush=True)
    videos = [v for g in state['groups'].values() for v in g['videos']]
    save(OUT/'eligibility_exclusions.json', dict(clip_rejections={k: v for k, v in state['attempts'].items() if not v['eligible']},
        group_exclusions={k: v for k, v in state['groups'].items() if not v['included']},
        attempted_clips=len(state['attempts']), complete_group_count=len(state['groups']),
        screening_rule='first two eligible in separate hash order, same group only; stop once two pass; no outcomes',
        scope='Selected technically eligible clips and replacements screened; unattempted candidates not claimed eligible'))
    videos = duplicate_quarantine(videos)
    achieved = {role: {c: len({v['group'] for v in videos if v['role'] == role and v['label'] == c})
                      for c in plan['classes']} for role in ['train', 'development', 'test']}
    save(OUT/'population_feasibility.json', dict(counts=achieved,
        group_totals={r: sum(cs.values()) for r, cs in achieved.items()},
        clip_totals={r: 2*sum(cs.values()) for r, cs in achieved.items()},
        thresholds=dict(train=10, development=2, test=3),
        after_input_eligibility_and_duplicate_quarantine=True, at_utc=now()))
    save(OUT/'screened_population.json', dict(videos=videos, classes=plan['classes'],
        status='input-only screened; population gate not yet passed', at_utc=now()))
    counts = require_population(videos, plan['classes'])
    # Predetermined cost cohort: one group in each of the 30 smallest hashed classes.
    timing = []
    for c in sorted(plan['classes'], key=lambda c: (rank('timing_class', c), c))[:30]:
        candidates = [v for v in videos if v['role'] == 'test' and v['label'] == c]
        g = min({v['group'] for v in candidates}, key=lambda g: (rank('timing_group', c, g), g))
        timing.append(min([v['clip'] for v in candidates if v['group'] == g], key=lambda n: (rank('timing_clip', n), n)))
    result = dict(classes=plan['classes'], videos=videos, counts=counts, timing_clips=timing,
                  created_at_utc=now(), metadata_sha256=sha(OUT/'population_plan.json'),
                  test_features_extracted=False, population='second-dataset replication with disjoint experimental groups')
    save(path, result)
    return result


def duplicate_quarantine(videos):
    """Conservative training-calibrated pHash screening; quarantine connected groups."""
    historical_hashes, historical_ids, evidence = set(), set(), []
    def walk(item):
        if isinstance(item, dict):
            identity = item.get('file_identity', {})
            if identity.get('sha256'):
                historical_hashes.add(identity['sha256'])
            for name in ['source_video_id', 'video_id']:
                if isinstance(item.get(name), str):
                    historical_ids.add(item[name])
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for v in item:
                walk(v)
    for folder in sorted((ROOT/'results').iterdir()):
        if folder == OUT or not folder.is_dir():
            continue
        for name in ['dataset_manifest.json', 'split_manifest.json', 'manifest.json', 'frozen_test_manifest.json',
                     'train_development_manifest.json', 'evaluation_manifest.json', 'confirmation_manifest.json']:
            p = folder/name
            if p.exists():
                walk(read(p)); evidence.append(dict(path=rel(p), sha256=sha(p)))
    # Threshold determined from training-only distinct-group temporal fingerprints.
    train = [v for v in videos if v['role'] == 'train']
    bits = np.array([[int(x, 2) for x in v['perceptual_fingerprint']] for v in videos], dtype=np.uint64)
    def distance(i, j):
        return [int(a ^ b).bit_count() for a, b in zip(bits[i], bits[j])]
    ti = [i for i, v in enumerate(videos) if v['role'] == 'train']
    # Deterministic 256 training clips for calibration only; no model values.
    ti = sorted(ti, key=lambda i: rank('duplicate_calibration', videos[i]['clip']))[:256]
    nearest = []
    for i in ti:
        ds = [max(distance(i, j)) for j in ti if (videos[i]['label'], videos[i]['group']) != (videos[j]['label'], videos[j]['group'])]
        nearest.append(min(ds))
    threshold = min(4, max(0, int(np.percentile(nearest, 1))-1))
    save(OUT/'duplicate_threshold.json', dict(fixed_at_utc=now(), training_calibration_clips=[videos[i]['clip'] for i in ti],
        training_nearest_max_hamming=nearest, threshold_bits_per_frame=threshold, duration_ratio_minimum=.9,
        rule='three 63-bit DCT pHashes at 10/50/90% decoded frames, ALL distances <= threshold; duration ratio >= .9; threshold=min(4,max(0,floor(training first percentile)-1))',
        test_model_outcomes_used=False))
    edges, quarantined = [], set()
    # Strict pairwise screening (~13 million inexpensive uint64 comparisons at nominal N).
    for i, a in enumerate(videos):
        ka = (a['label'], a['group'])
        if a['sha256'] in historical_hashes or a['clip'] in historical_ids or Path(a['clip']).stem in historical_ids:
            quarantined.add(ka)
            edges.append(dict(a=a['clip'], b='historical cohort', reason='known exact hash/source identifier'))
        for j in range(i):
            b = videos[j]; kb = (b['label'], b['group'])
            if ka == kb:
                continue
            exact = a['sha256'] == b['sha256']
            ratio = min(a['duration_pts_span'], b['duration_pts_span'])/max(a['duration_pts_span'], b['duration_pts_span'])
            # Reject on first hash without allocating three Python distance arrays.
            near = (ratio >= .9 and int(bits[i, 0] ^ bits[j, 0]).bit_count() <= threshold
                    and all(v <= threshold for v in distance(i, j)))
            if exact or near:
                edges.append(dict(a=a['clip'], b=b['clip'], reason='exact bytes' if exact else 'conservative temporal pHash candidate',
                                  cross_role=a['role'] != b['role']))
                # Quarantine every conflicting group, even same-role candidates, before inference.
                quarantined.update([ka, kb])
    import csv
    annotations = ROOT/'data/source/temporal_refresh_generalization_v1/k400_train.csv'
    pretrain = list(csv.DictReader(annotations.open(encoding='utf-8')))
    pretrain_ids = {row['youtube_id'] for row in pretrain}
    result = dict(at_utc=now(), historical_manifest_evidence=evidence, historical_hashes=len(historical_hashes),
        historical_source_ids=len(historical_ids), edges=edges, quarantined_groups=[list(k) for k in sorted(quarantined)],
        official_K400_training_annotations=dict(path=rel(annotations), sha256=sha(annotations),
            rows=len(pretrain), parent_IDs=len(pretrain_ids), UCF_parent_identifiers_available=0,
            mapped_UCF_overlap=None, reason='Official UCF filenames supply class/group/clip, no YouTube parent mapping; unknown is not zero overlap'),
        manual_decisions=[], manual_policy='No manual retain overrides; every flagged connected group quarantined in full; roles never moved',
        checkpoint_pretraining='X3D-XS Kinetics400; local official train.csv identities available, UCF filename-to-YouTube parent mapping unavailable',
        unknown='UCF source URLs/parent IDs, Kinetics reuploads, perceptual matches to unavailable pretraining video bytes and unflagged near duplicates remain unknown',
        limitation='Exact historical hashes and available IDs audited; perceptual screen covers experimental UCF groups, not all historical/pretraining video content. Not universally pretraining-clean.')
    save(OUT/'duplicate_and_pretraining_audit.json', result)
    return [v for v in videos if (v['label'], v['group']) not in quarantined]
