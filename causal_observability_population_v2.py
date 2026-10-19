"""Experiment 19.1: genuine INPUT-ONLY population evidence; no model interface.

Synthetic arrays belong only in test_causal_observability_population_v2.py.
Heavy decoder/solver/plot imports are local so the audit has no such side effects.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tarfile
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/causal_observability_population_v2'
LOCAL = ROOT / 'data/source/causal_observability_population_v2'
ORIGINAL = ROOT / 'data/source/causal_observability_v1'
HISTORICAL = ROOT / 'results/causal_observability_v1'
RULE = 'adaptive_history_index_uniform_v2'
SEED = 20261019
REVIEWED = 'e9b7f5422f20083bafaa82569a3bd9ddd429fe08'
ROLES = ['head_train', 'controller_train', 'development']
CONFIGS = {'A': [30, 10, 10], 'B': [25, 10, 5], 'C': [20, 8, 5]}
ARCHIVES = {
    'hmdb51_org.rar': '9e714a0d8b76104d76e932764a7ca636f929fff66279cda3f2e326fa912a328e',
    'test_train_splits.rar': '229c94f845720d01eb3946d39f39292ea962d50a18136484aa47c1eba251d2b7',
}
IMPLEMENTATION = ['causal_observability_population_v2.py', 'run_causal_observability_population_v2.py',
                  'research_utils.py', 'temporal_refresh_common.py', 'ucf101_proxy_data.py',
                  'causal_observability_data.py', 'causal_observability_common.py',
                  'ucf101_proxy_common.py']
REVIEW_FILES = ['README.md', 'causal_observability_common.py', 'causal_observability_data.py',
    'causal_observability_policy.py', 'causal_observability_runtime.py', 'causal_observability_learning.py',
    'causal_observability_analysis.py', 'causal_observability_verify.py', 'run_causal_observability.py',
    'test_causal_observability.py', 'temporal_refresh_common.py', 'ucf101_proxy_data.py',
    'results/ucf101_proxy_replication_v1/findings.md']


class PopulationError(RuntimeError):
    """Missing resource or changed evidence; never silently rescue the population."""


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def check(condition, message):
    if not condition:
        raise PopulationError(message)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def rank(*parts):
    return hashlib.sha256(('|'.join(map(str, (SEED, *parts)))).encode()).hexdigest()


def adaptive_grid(timestamps, identities=None):
    """Only real presentation indices, current or earlier; no synthetic timestamps."""
    pts = np.asarray(timestamps, dtype=np.float64)
    if pts.ndim != 1 or not np.isfinite(pts).all() or not np.all(np.diff(pts) > 0):
        raise ValueError('Missing/nonfinite/duplicate/nonincreasing actual PTS')
    if len(pts) < 19:
        raise ValueError('Fewer than 19 distinct actual presentation frames')
    endpoints = np.floor(np.linspace(3, len(pts) - 1, 16)).astype(int)
    check(len(endpoints) == 16 and np.all(np.diff(endpoints) > 0), 'Endpoint contract failed')
    left_time = np.searchsorted(pts, pts[endpoints] - 47/30, side='left')
    left = np.minimum(left_time, endpoints - 3)
    indices = np.array([np.floor(np.linspace(a, i, 4)).astype(int) for a, i in zip(left, endpoints)])
    check(np.all(np.diff(indices, axis=1) > 0) and np.array_equal(indices[:, -1], endpoints)
          and np.all(indices <= endpoints[:, None]), 'Snippet distinctness/causality failed')
    result = dict(rule=RULE, endpoint_indices=endpoints.tolist(), endpoint_timestamps=pts[endpoints].tolist(),
                  frame_indices=indices.tolist(), actual_timestamps=pts[indices].tolist(),
                  actual_offsets=(pts[endpoints, None] - pts[indices]).tolist(),
                  left_time_indices=left_time.tolist(), left_indices=left.tolist(),
                  history_window_seconds=(pts[endpoints] - pts[left]).tolist(),
                  window_extended=(left < left_time).tolist(), clip_pts_span=float(pts[-1] - pts[0]))
    if identities is not None:
        check(len(identities) == len(pts), 'Frame identity length mismatch')
        result['frame_identities'] = [[identities[j] for j in row] for row in indices]
        result['endpoint_identities'] = [identities[j] for j in endpoints]
    return result


def scalar_validate_frames(record):
    """Independent scalar replay from rational PTS; no decoder or vector sampler."""
    ids = record.get('presentation_identities', [])
    if not record['new_eligible']:
        return
    check(len(ids) >= 19, 'Insufficient retained real identities')
    rational = [Fraction(i['pts']) * Fraction(i['time_base']) for i in ids]
    check(all(b > a for a, b in zip(rational, rational[1:])), 'Nonincreasing rational PTS')
    check(sorted(i['index'] for i in ids) == list(range(record['decoded_frames'])), 'Decode ordinals not a permutation')
    pts = [float(x) for x in rational]
    grid = record['grid']
    check(grid['rule'] == RULE, 'Wrong sampling rule')
    endpoints = [(3 * 15 + t * (len(ids) - 4)) // 15 for t in range(16)]
    check(grid['endpoint_indices'] == endpoints, 'Scalar endpoint replay differs')
    for t, i in enumerate(endpoints):
        threshold = pts[i] - 47/30
        lt = next((j for j, p in enumerate(pts) if p >= threshold), len(pts))
        left = min(lt, i - 3)
        expected = [(left * 3 + k * (i - left)) // 3 for k in range(4)]
        check(grid['frame_indices'][t] == expected and len(set(expected)) == 4, 'Scalar frame replay differs')
        check(expected[-1] == i and all(j <= i for j in expected), 'Future or repeated snippet frame')
        check(grid['frame_identities'][t] == [ids[j] for j in expected], 'Snippet identities changed')
        check(grid['endpoint_identities'][t] == ids[i], 'Endpoint identity changed')
        check(grid['actual_timestamps'][t] == [pts[j] for j in expected], 'Actual times changed')
        check(grid['actual_offsets'][t] == [pts[i]-pts[j] for j in expected], 'Actual offsets changed')
        check(grid['left_time_indices'][t] == lt and grid['left_indices'][t] == left, 'Window indices changed')
        check(grid['window_extended'][t] == (left < lt), 'Window extension changed')
        check(grid['history_window_seconds'][t] == pts[i]-pts[left], 'Window span changed')
    check(grid['endpoint_timestamps'] == [pts[i] for i in endpoints], 'Candidate times changed')
    check(grid['clip_pts_span'] == pts[-1]-pts[0], 'Clip PTS span changed')


def archive_members(path):
    """Read authenticated RAR members as a tar stream; never extract/modify inputs."""
    command = ['tar.exe', '-cf', '-', '@' + str(path)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
                for member in stream:
                    q = PurePosixPath(member.name.replace('\\', '/'))
                    check(not q.is_absolute() and '..' not in q.parts and ':' not in member.name,
                          'Unsafe archive member')
                    if member.isdir():
                        continue
                    check(member.isfile(), 'Nonregular archive member')
                    h = hashlib.sha256()
                    f = stream.extractfile(member)
                    for block in iter(lambda: f.read(2**20), b''):
                        h.update(block)
                    yield q.as_posix(), dict(bytes=member.size, sha256=h.hexdigest())
        finally:
            if process.poll() is None:
                process.stdout.close()
            error = process.stderr.read().decode('utf-8', errors='replace')
            code = process.wait()
        check(code == 0, 'Archive authentication/CRC read failed: ' + error)


def original_metadata():
    from causal_observability_data import source_family
    paths = sorted((ORIGINAL/'splits').rglob('*_test_split1.txt'))
    check(len(paths) == 51, 'Missing official split-1 class files')
    rows, classes = [], []
    for path in paths:
        label = path.name.removesuffix('_test_split1.txt')
        classes.append(label)
        for line in path.read_text(encoding='utf-8').splitlines():
            clip, flag = line.rsplit(None, 1)
            family, title = source_family(clip, label)
            check(int(flag) in (0, 1, 2) and Path(clip).name == clip, 'Invalid annotation')
            rows.append(dict(key=label+'/'+clip, class_name=label, filename=clip,
                             official_flag=int(flag), source_title=title, title_family=family))
    check(len({r['key'] for r in rows}) == len(rows), 'Duplicated official annotations')
    for c in classes:
        count = Counter(r['official_flag'] for r in rows if r['class_name'] == c)
        check(count[1] == 70 and count[2] == 30, 'Unexpected official split-1 membership')
    return classes, rows


def prepare():
    """Verify immutable history and original bytes before new screening."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    baseline_path = OUT/'historical_baseline.json'
    if baseline_path.exists():
        baseline = read(baseline_path)
        verify_hashes(baseline['sha256'])
    else:
        tracked = git('ls-files', 'results').splitlines()
        historical = {p: sha(ROOT/p) for p in tracked if (ROOT/p).is_file() and not p.startswith(rel(OUT)+'/')}
        historical.update({p: sha(ROOT/p) for p in REVIEW_FILES if p != 'README.md'})
        original_records = ['extraction.json', 'eligibility_attempts.jsonl', 'eligibility_progress.json',
                            'eligible_input_population.json', 'official_page.html', 'distribution_README.md',
                            'distribution_metadata.json']
        historical.update({rel(ORIGINAL/n): sha(ORIGINAL/n) for n in original_records if (ORIGINAL/n).exists()})
        baseline = dict(at_utc=now(), initial_checkout='0dd02da818ec41be254b30b56caaf8d612eceb93',
                        actual_main=git('rev-parse', 'origin/main'), reviewed_main=REVIEWED,
                        initial_and_main_trees_equal=git('rev-parse', '0dd02da^{tree}') == git('rev-parse', REVIEWED+'^{tree}'),
                        sha256=historical, review_files={p: sha(ROOT/p) for p in REVIEW_FILES},
                        relevant_newer_main_changes=git('log', '--oneline', REVIEWED+'..origin/main'))
        save(baseline_path, baseline)
    check(shutil.disk_usage(ROOT).free > 3*2**30, 'Need 3GiB free for metadata checkpoints')
    identity_path = OUT/'archive_member_inventory.json'
    if identity_path.exists():
        inventory = read(identity_path)
        verify_hashes(inventory['archive_sha256'])
        verify_hashes(inventory['inner_archive_sha256'])
    else:
        for n, expected in ARCHIVES.items():
            check((ORIGINAL/n).exists() and sha(ORIGINAL/n) == expected, 'Original archive unavailable/changed: '+n)
        outer = dict(archive_members(ORIGINAL/'hmdb51_org.rar'))
        inner, videos = {}, {}
        for n, info in sorted(outer.items()):
            path = ORIGINAL/'class_archives'/n
            check(path.is_file() and sha(path) == info['sha256'], 'Original inner archive mismatch: '+n)
            inner[rel(path)] = info['sha256']
            for key, member in archive_members(path):
                check(key not in videos and key.endswith('.avi'), 'Duplicate/unexpected original AVI member')
                videos[key] = member
            print('Authenticated original archive', n, len(videos), flush=True)
        annotations = dict(archive_members(ORIGINAL/'test_train_splits.rar'))
        actual_annotations = {p.relative_to(ORIGINAL/'splits').as_posix(): p for p in (ORIGINAL/'splits').rglob('*.txt')}
        check(set(actual_annotations) == set(annotations), 'Extracted annotations differ from original archive')
        for key, info in annotations.items():
            check(sha(actual_annotations[key]) == info['sha256'], 'Annotation bytes differ from archive')
        inventory = dict(archive_sha256={rel(ORIGINAL/n): h for n, h in ARCHIVES.items()},
                         inner_archive_sha256=inner, videos=videos, annotations=annotations,
                         method='Read original RAR via bsdtar archive-to-tar stream; SHA256 every member; CRC errors fatal',
                         archive_identity=read(HISTORICAL/'source_metadata_identities.json'))
        save(identity_path, inventory)
    classes, rows = original_metadata()
    check(set(r['key'] for r in rows) == set(inventory['videos']), 'Archive/annotation membership mismatch')
    actual = {p.relative_to(ORIGINAL/'videos').as_posix() for p in (ORIGINAL/'videos').rglob('*.avi')}
    check(actual == set(inventory['videos']), 'Extracted AVI membership differs from original archive')
    return classes, rows, inventory


def verify_hashes(records):
    for p, expected in records.items():
        check((ROOT/p).is_file() and sha(ROOT/p) == expected, 'Missing or changed evidence: '+p)


def screen_clip(path):
    import av
    from PIL import Image
    from scipy.fft import dctn
    from research_utils import configure_decoder
    from temporal_refresh_common import snippet_grid
    from ucf101_proxy_data import presentation_order
    frames, pts, ids = [], [], []
    result = dict(new_eligible=False, old_eligible=False, decoded_frames=0, valid_distinct_pts=0,
                  technical_exclusion=None, old_exclusion=None, fingerprints=[])
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            configure_decoder(stream)
            result.update(codec=stream.codec_context.name, container_duration_seconds=(
                float(container.duration/av.time_base) if container.duration is not None else None))
            for ordinal, frame in enumerate(container.decode(stream)):
                frames.append(frame)
                if frame.pts is None or frame.time_base is None:
                    pts.append(None)
                    ids.append(dict(index=ordinal, pts=None, time_base=None))
                else:
                    pts.append(float(frame.pts*frame.time_base))
                    ids.append(dict(index=ordinal, pts=int(frame.pts), time_base=str(frame.time_base)))
            result['decoded_frames'] = len(frames)
            result['valid_distinct_pts'] = len(set(p for p in pts if p is not None and np.isfinite(p)))
            check(None not in pts, 'Missing actual PTS/time_base')
            frames, pts, ids, reordered = presentation_order(frames, pts, ids)
            result.update(presentation_identities=ids, presentation_reordered_frames=reordered,
                          pts_span=float(pts[-1]-pts[0]))
            for q in [.1, .5, .9]:
                a = frames[int(q*(len(frames)-1))].to_ndarray(format='gray')
                image = Image.fromarray(a).resize((32, 32), resample=Image.Resampling.BICUBIC)
                low = dctn(np.asarray(image, dtype=float), norm='ortho')[:8, :8].ravel()[1:]
                result['fingerprints'].append(''.join('1' if b else '0' for b in low > np.median(low)))
            try:
                snippet_grid(pts)
                result['old_eligible'] = True
            except ValueError as e:
                result['old_exclusion'] = str(e)
            try:
                result['grid'] = adaptive_grid(pts, ids)
                result['new_eligible'] = True
            except ValueError as e:
                result['technical_exclusion'] = str(e)
    except Exception as e:
        result['technical_exclusion'] = type(e).__name__+': '+str(e)
        result['old_exclusion'] = result['old_exclusion'] or result['technical_exclusion']
    return result


def checkpoint_identity():
    return dict(rule=RULE, protocol_sha256=sha(OUT/'protocol.md'),
                code_sha256={p: sha(ROOT/p) for p in IMPLEMENTATION},
                archive_inventory_sha256=sha(OUT/'archive_member_inventory.json'))


def load_journal(path, identity, recover=False):
    records = {}
    if not path.exists():
        return records
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        check(recover, 'Interrupted journal tail; run --resume to preserve/recover')
        archive = path.with_name(path.name+'.interrupted-'+hashlib.sha256(raw).hexdigest()[:12])
        archive.write_bytes(raw)
        raw = raw[:raw.rfind(b'\n')+1]
        path.write_bytes(raw)
    for line in raw.splitlines():
        item = json.loads(line)
        check(item['identity'] == identity, 'Checkpoint implementation/input identity changed')
        check(item['payload_sha256'] == digest(item['payload']), 'Checkpoint payload corrupted')
        record = item['payload']
        check(record['key'] not in records, 'Duplicate checkpoint key')
        records[record['key']] = record
    return records


def screening(rows, inventory):
    identity = checkpoint_identity()
    path = LOCAL/'screening.jsonl'
    completed = load_journal(path, identity, recover=True)
    check(set(completed).issubset(r['key'] for r in rows), 'Foreign screening checkpoint')
    start = time.perf_counter()
    for i, meta in enumerate(rows):
        p = ORIGINAL/'videos'/meta['key']
        expected = inventory['videos'][meta['key']]['sha256']
        check(sha(p) == expected, 'AVI differs from authenticated member: '+meta['key'])
        if meta['key'] in completed:
            check(completed[meta['key']]['sha256'] == expected, 'Checkpoint input differs')
            continue
        tick = time.perf_counter()
        record = dict(**meta, sha256=expected, bytes=p.stat().st_size,
                      **screen_clip(p), decoded_at_utc=now(), input_screen_seconds=time.perf_counter()-tick)
        scalar_validate_frames(record)
        item = dict(identity=identity, payload=record, payload_sha256=digest(record))
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
            f.flush()
        completed[meta['key']] = record
        if (i+1) % 100 == 0 or i+1 == len(rows):
            print(f'Real input-only screening {i+1}/{len(rows)}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    records = [completed[r['key']] for r in rows]
    with (OUT/'screening_evidence.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as f:
            for record in records:
                f.write((json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode())
    save(OUT/'screening_identity.json', dict(identity=identity, clips=len(records),
         evidence_sha256=sha(OUT/'screening_evidence.jsonl.gz'), local_checkpoint=rel(path),
         local_checkpoint_sha256=sha(path), all_records_have_genuine_archive_hash=True))
    return records


def source_edges(rows):
    """All observed links; title links remain explicitly uncertain evidence."""
    edges = []
    for field, kind in [('sha256', 'exact'), ('title_family', 'title')]:
        buckets = defaultdict(list)
        for r in rows:
            buckets[r[field]].append(r)
        for value, members in sorted(buckets.items()):
            for other in members[1:]:
                edges.append(dict(first=members[0]['key'], second=other['key'], kind=kind,
                                  evidence=value, independence='confirmed_same_bytes' if kind == 'exact'
                                  else 'uncertain_shared_normalized_title'))
    good = [r for r in rows if len(r['fingerprints']) == 3 and r.get('pts_span', 0) > 0]
    bits = np.array([[int(h, 2) for h in r['fingerprints']] for r in good], dtype=np.uint64)
    spans = np.array([r['pts_span'] for r in good])
    for i, r in enumerate(good):
        distance = np.bitwise_count(bits[i+1:] ^ bits[i])
        ratio = np.minimum(spans[i+1:], spans[i])/np.maximum(spans[i+1:], spans[i])
        for offset in np.flatnonzero((distance <= 4).all(1) & (ratio >= .9)):
            j = i+1+int(offset)
            edges.append(dict(first=r['key'], second=good[j]['key'], kind='near',
                              hamming=distance[offset].astype(int).tolist(), duration_ratio=float(ratio[offset]),
                              independence='conservative_detectable_visual_similarity'))
    return edges


def components(rows, edges, kinds=('exact', 'near', 'title')):
    parent = {r['key']: r['key'] for r in rows}
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for e in edges:
        if e['kind'] in kinds:
            a, b = root(e['first']), root(e['second'])
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for k in parent:
        groups[root(k)].append(k)
    mapping, members = {}, {}
    for group in groups.values():
        group = sorted(group)
        identity = 'family_'+digest(group)
        members[identity] = group
        mapping.update({k: identity for k in group})
    return mapping, members


def availability(rows, mapping, eligible_field='new_eligible'):
    test_families = {mapping[r['key']] for r in rows if r['official_flag'] == 2}
    train = [r for r in rows if r[eligible_field] and r['official_flag'] == 1 and mapping[r['key']] not in test_families]
    test = [r for r in rows if r[eligible_field] and r['official_flag'] == 2]
    return train, test, test_families


def family_audit(rows):
    edges = source_edges(rows)
    mapping, members = components(rows, edges)
    lookup = {r['key']: r for r in rows}
    by_family_edges = defaultdict(list)
    for i, e in enumerate(edges):
        by_family_edges[mapping[e['first']]].append(i)
    details = []
    conflicts = []
    for family, keys in sorted(members.items()):
        selected = [lookup[k] for k in keys]
        flags = Counter(r['official_flag'] for r in selected)
        conflict = flags[1] > 0 and flags[2] > 0
        excluded = [r['key'] for r in selected if r['official_flag'] == 1 and conflict]
        edge_ids = by_family_edges[family]
        details.append(dict(source_family=family, size=len(keys), classes=sorted({r['class_name'] for r in selected}),
            n_classes=len({r['class_name'] for r in selected}), official_roles={str(f): flags[f] for f in (0, 1, 2)},
            new_eligible=sum(r['new_eligible'] for r in selected), members=keys,
            normalized_titles=sorted({r['title_family'] for r in selected}),
            title_examples=sorted({r['source_title'] for r in selected if r['source_title']})[:10],
            edge_indices=edge_ids, edge_counts=dict(Counter(edges[i]['kind'] for i in edge_ids)),
            train_test_conflict=conflict, training_members_excluded=excluded,
            eligible_training_excluded=sum(lookup[k]['new_eligible'] for k in excluded),
            independence_status='uncertain_original_parent_identifiers_unavailable',
            large_component_review='Conservative component retained intact; broad title may merge distinct sources; no metadata-only split authorized'
                if len(keys) >= 20 else None))
        if conflict:
            conflicts.append(dict(source_family=family, family_size=len(keys), classes='|'.join(details[-1]['classes']),
                official_train=flags[1], official_test=flags[2], official_unused=flags[0],
                eligible_training_excluded=details[-1]['eligible_training_excluded'],
                eligible_test_kept=sum(r['new_eligible'] and r['official_flag'] == 2 for r in selected)))
    sensitivities = {}
    classes = sorted({r['class_name'] for r in rows})
    for name, kinds in [('exact_only', ('exact',)), ('exact_near', ('exact', 'near')),
                        ('title_only', ('title',)), ('conservative_union_PRIMARY', ('exact', 'near', 'title'))]:
        m, groups = components(rows, edges, kinds)
        train, test, _ = availability(rows, m)
        sensitivities[name] = dict(role='primary' if name.endswith('PRIMARY') else 'less_conservative_diagnostic_NOT_sufficient_for_GO',
            components=len(groups), training_available=len(train), test_eligible=len(test),
            per_class={c: dict(training_clips=sum(r['class_name'] == c for r in train),
                              training_families=len({m[r['key']] for r in train if r['class_name'] == c}),
                              test_clips=sum(r['class_name'] == c for r in test),
                              test_families=len({m[r['key']] for r in test if r['class_name'] == c})) for c in classes})
    result = dict(components=details, edges=edges, edge_counts=dict(Counter(e['kind'] for e in edges)),
        missing_title_identifiers=sum(r['source_title'] is None for r in rows), missing_original_parent_URLs=len(rows),
        uncertain_independence_components=len(details), conflict_families=len(conflicts),
        training_members_excluded=sum(len(d['training_members_excluded']) for d in details),
        eligible_training_excluded=sum(d['eligible_training_excluded'] for d in details),
        all_eligible_test_members_kept=True, manual_retain_overrides=[],
        near_rule='unchanged v1: all three 63-bit pHash distances <=4 and PTS-span ratio>=.9; all matching pairs retained',
        threshold_selection='inherited input-only v1 threshold; not optimized on v2 clips or recognition',
        grouping_sensitivities=sensitivities,
        large_components=[d['source_family'] for d in details if d['size'] >= 20],
        top_components=[d['source_family'] for d in sorted(details, key=lambda d: (-d['size'], d['source_family']))[:15]])
    save(OUT/'source_family_audit.json', result)
    write_csv(OUT/'source_conflicts.csv', conflicts)
    return mapping, result


def counts_by_class(rows, classes, mapping):
    train, test, _ = availability(rows, mapping)
    old_train, old_test, _ = availability(rows, mapping, 'old_eligible')
    output = []
    for c in classes:
        group = [r for r in rows if r['class_name'] == c]
        item = dict(class_name=c, annotated=len(group),
            old_eligible=sum(r['old_eligible'] for r in group), new_eligible=sum(r['new_eligible'] for r in group),
            newly_eligible=sum(r['new_eligible'] and not r['old_eligible'] for r in group))
        for f, name in [(1, 'train'), (2, 'test'), (0, 'unused')]:
            subset = [r for r in group if r['official_flag'] == f]
            item['official_'+name] = len(subset)
            item['old_'+name+'_eligible'] = sum(r['old_eligible'] for r in subset)
            item['new_'+name+'_eligible'] = sum(r['new_eligible'] for r in subset)
            item['new_'+name+'_temporal_excluded'] = sum(not r['new_eligible'] for r in subset)
        a = [r for r in train if r['class_name'] == c]
        b = [r for r in test if r['class_name'] == c]
        item.update(training_available=len(a), training_families=len({mapping[r['key']] for r in a}),
            test_families=len({mapping[r['key']] for r in b}),
            training_source_excluded=item['new_train_eligible']-len(a),
            old_training_available_same_v2_graph=sum(r['class_name'] == c for r in old_train),
            old_test_available=sum(r['class_name'] == c for r in old_test))
        output.append(item)
    return output


def necessary_failures(class_counts, quotas):
    failures = []
    for r in class_counts:
        for field, required in [('training_available', sum(quotas)), ('training_families', 9),
                                ('new_test_eligible', 10), ('test_families', 5)]:
            if r[field] < required:
                failures.append(dict(class_name=r['class_name'], constraint=field,
                                     available=r[field], required=required,
                                     proof='Disjoint role minima cannot exceed total available clips/families; fixed test cannot be reassigned'))
    return failures


def validate_assignment(rows, classes, mapping, assignments, quotas, enforce_test=True):
    """Independent counting of actual clips and component support (not MILP matrix)."""
    train, test, test_families = availability(rows, mapping)
    expected = {mapping[r['key']] for r in train}
    check(set(assignments) == expected, 'Assignment must cover exactly all available training families')
    check(all(role in ROLES for role in assignments.values()), 'Invalid training role')
    check(not expected.intersection(test_families), 'Test family assigned to training')
    result = []
    for c in classes:
        counts = dict(class_name=c)
        for role, minimum in zip(ROLES, quotas):
            selected = [r for r in train if r['class_name'] == c and assignments[mapping[r['key']]] == role]
            n_groups = len({mapping[r['key']] for r in selected})
            check(len(selected) >= minimum and n_groups >= 3, 'Independent allocation minimum failed: '+c+'/'+role)
            counts[role+'_clips'] = len(selected)
            counts[role+'_families'] = n_groups
        selected = [r for r in test if r['class_name'] == c]
        counts['test_clips'] = len(selected)
        counts['test_families'] = len({mapping[r['key']] for r in selected})
        if enforce_test:
            check(counts['test_clips'] >= 10 and counts['test_families'] >= 5, 'Fixed test minimum failed: '+c)
        result.append(counts)
    return result


def solve_configuration(rows, classes, mapping, class_counts, name):
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_array
    from scipy.optimize._highspy._core import _Highs
    train, _, _ = availability(rows, mapping)
    families = sorted({mapping[r['key']] for r in train}, key=lambda f: (rank('family_order', f), f))
    family_index = {f: i for i, f in enumerate(families)}
    class_index = {c: i for i, c in enumerate(classes)}
    counts = Counter((class_index[r['class_name']], family_index[mapping[r['key']]]) for r in train)
    nf, nc = len(families), len(classes)
    rr, cc, vv = [], [], []
    for f in range(nf):
        for role in range(3):
            rr.append(f); cc.append(role*nf+f); vv.append(1.)
    for (c, f), n in counts.items():
        for role in range(3):
            for row, value in [(nf+role*nc+c, n), (nf+3*nc+role*nc+c, 1)]:
                rr.append(row); cc.append(role*nf+f); vv.append(float(value))
    a = coo_array((vv, (rr, cc)), shape=(nf+6*nc, 3*nf)).tocsc()
    quotas = CONFIGS[name]
    lb = np.r_[np.ones(nf), np.repeat(quotas, nc), np.full(3*nc, 3.)]
    ub = np.r_[np.ones(nf), np.full(6*nc, np.inf)]
    costs = np.array([int(rank('role_cost', f, role)[:12], 16)/16**12 for role in ROLES for f in families])
    options = dict(time_limit=300., mip_rel_gap=0., threads=1, random_seed=SEED,
                   primal_feasibility_tolerance=1e-7, mip_feasibility_tolerance=1e-7)
    proofs = necessary_failures(class_counts, quotas)
    runs, solutions = [], []
    for repetition in range(2):
        tick = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = milp(costs, integrality=np.ones(3*nf), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(a, lb, ub), options=options)
        assignment = None
        if result.x is not None:
            check(np.max(abs(result.x-np.rint(result.x))) <= 1e-6, 'Nonintegral returned solution')
            x = np.rint(result.x).reshape(3, nf)
            assignment = {f: ROLES[int(np.argmax(x[:, i]))] for i, f in enumerate(families)}
            validate_assignment(rows, classes, mapping, assignment, quotas, enforce_test=False)
        runs.append(dict(repetition=repetition, status=int(result.status), success=bool(result.success),
            message=result.message, seconds=time.perf_counter()-tick,
            objective=None if result.fun is None else float(result.fun),
            mip_gap=None if getattr(result, 'mip_gap', None) is None else float(result.mip_gap),
            mip_dual_bound=None if getattr(result, 'mip_dual_bound', None) is None else float(result.mip_dual_bound),
            mip_node_count=None if getattr(result, 'mip_node_count', None) is None else int(result.mip_node_count),
            warnings=[str(w.message) for w in caught], assignment_digest=digest(assignment)))
        solutions.append(assignment)
    check(runs[0]['status'] == runs[1]['status'] and solutions[0] == solutions[1], 'Solver repeat not deterministic')
    feasible = not proofs and solutions[0] is not None and runs[0]['status'] == 0
    impossible = bool(proofs) or runs[0]['status'] == 2
    status = 'FEASIBLE' if feasible else ('PROVEN_INFEASIBLE' if impossible else 'UNRESOLVED_RESOURCE_LIMIT')
    check(not (feasible and impossible), 'Contradictory solver/certificate evidence')
    independent = validate_assignment(rows, classes, mapping, solutions[0], quotas) if feasible else None
    return dict(configuration=name, minima=dict(zip(ROLES, quotas)), test_clips_minimum=10,
        training_families_per_role_minimum=3, test_families_minimum=5, status=status,
        necessary_condition_failures=proofs, solver=dict(scipy=importlib.metadata.version('scipy'), highs=_Highs().version(),
            options=options, seed=SEED, variables=3*nf, constraints=nf+6*nc, objective='minimum seeded SHA256 role costs',
            matrix_digest=digest(dict(counts=sorted((c, f, n) for (c, f), n in counts.items()), families=families, quotas=quotas)), runs=runs),
        deterministic_repeat_verified=True, assignments=solutions[0] if feasible else None,
        independent_counts=independent)


def write_csv(path, rows):
    check(bool(rows), 'Refuse ambiguous empty CSV without schema: '+str(path))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def retained_records():
    with gzip.open(OUT/'screening_evidence.jsonl.gz', 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def allocation_manifest(rows, mapping, chosen, grid):
    assignments = grid[chosen]['assignments'] if chosen else {}
    train, _, test_families = availability(rows, mapping)
    available = {r['key'] for r in train}
    output = []
    for r in rows:
        family = mapping[r['key']]
        if not r['new_eligible']:
            role, reason = 'excluded_technical', r['technical_exclusion']
        elif r['official_flag'] == 2:
            role, reason = 'test', None
        elif r['official_flag'] == 0:
            role, reason = 'excluded_official_unused', 'Official split-1 unused; no substitution'
        elif family in test_families:
            role, reason = 'excluded_test_family_conflict', 'Conservative official test-family priority'
        elif chosen:
            role, reason = assignments[family], None
        else:
            check(r['key'] in available, 'Unaccounted eligible training clip')
            role, reason = 'unassigned_training_pool', 'No feasible configuration; no study roles invented'
        output.append(dict(key=r['key'], class_name=r['class_name'], filename=r['filename'], sha256=r['sha256'],
                           official_flag=r['official_flag'], source_family=family, role=role, exclusion_reason=reason))
    return dict(configuration=chosen, classes=sorted({r['class_name'] for r in rows}),
                rule=RULE, seed=SEED, videos=output,
                scientific_stage='population_gate_only_no_model_execution',
                no_go_allocation_note='Test population and eligible training pool retained even when no training-role assignment exists')


def validate_leakage(manifest, rows, inventory, mapping, grid):
    """Independent assertions over retained inputs; rejects altered or foreign data."""
    actual = {r['key']: r for r in rows}
    check(len(actual) == len(rows) == len(manifest['videos']), 'Population count mismatch')
    check(set(actual) == set(inventory['videos']), 'Population not original archive membership')
    by_role_family, by_role_hash = defaultdict(set), defaultdict(set)
    selected_test = set()
    for v in manifest['videos']:
        r = actual[v['key']]
        check(v['sha256'] == inventory['videos'][v['key']]['sha256'] == r['sha256'], 'Manifest SHA corruption')
        check(v['class_name'] == r['class_name'] and v['official_flag'] == r['official_flag'], 'Manifest label/flag mutation')
        check(v['source_family'] == mapping[v['key']], 'Manifest component mutation')
        if v['role'] in ROLES + ['test']:
            check(r['new_eligible'], 'Technical exclusion retained')
            check(v['official_flag'] == (2 if v['role'] == 'test' else 1), 'Official test/unused entering training')
            by_role_family[v['source_family']].add(v['role'])
            by_role_hash[v['sha256']].add(v['role'])
        if v['role'] == 'test':
            selected_test.add(v['key'])
        scalar_validate_frames(r)
    check(all(len(s) == 1 for s in by_role_family.values()), 'Conservative family crosses roles')
    check(all(len(s) == 1 for s in by_role_hash.values()), 'Exact duplicate crosses roles')
    check(selected_test == {r['key'] for r in rows if r['new_eligible'] and r['official_flag'] == 2}, 'Official eligible test altered')
    check(manifest['classes'] == sorted({r['class_name'] for r in rows}) and len(manifest['classes']) == 51,
          'Full 51-class membership lost')
    chosen = manifest['configuration']
    if chosen:
        assignments = {v['source_family']: v['role'] for v in manifest['videos'] if v['role'] in ROLES}
        validate_assignment(rows, manifest['classes'], mapping, assignments, CONFIGS[chosen])
        check(assignments == grid[chosen]['assignments'], 'Manifest differs from solved assignment')
    return dict(passed=True, original_annotated_clips=len(rows), exact_duplicate_cross_role_violations=0,
        family_cross_role_violations=0, official_test_or_unused_in_training=0,
        eligible_test_members_unchanged=True, full_class_inventory=51, scalar_actual_frame_replays=sum(r['new_eligible'] for r in rows),
        chosen_training_configuration_verified=chosen,
        no_feasible_configuration_note=None if chosen else 'Training-role minima do not pass; validation does not claim they do',
        recognition_outputs_used=False, official_test_encoder_calls=0,
        input_only_implementation_imports='No torch, model, inference, download or learning calls in v2 execution')


def graph_replay(rows, audit):
    """Independent breadth-first components, separately from union-find builder."""
    adj = {r['key']: set() for r in rows}
    lookup = {r['key']: r for r in rows}
    for e in audit['edges']:
        a, b = lookup[e['first']], lookup[e['second']]
        if e['kind'] == 'exact':
            check(a['sha256'] == b['sha256'] == e['evidence'], 'False exact edge')
        elif e['kind'] == 'title':
            check(a['title_family'] == b['title_family'] == e['evidence'], 'False title edge')
        elif e['kind'] == 'near':
            distances = [(int(x, 2)^int(y, 2)).bit_count() for x, y in zip(a['fingerprints'], b['fingerprints'])]
            ratio = min(a['pts_span'], b['pts_span'])/max(a['pts_span'], b['pts_span'])
            check(len(distances) == 3 and max(distances) <= 4 and ratio >= .9, 'False near duplicate edge')
            check(distances == e['hamming'] and ratio == e['duration_ratio'], 'Near-edge evidence changed')
        else:
            raise PopulationError('Unknown graph link type')
        adj[a['key']].add(b['key']); adj[b['key']].add(a['key'])
    unseen, mapping, groups = set(adj), {}, {}
    while unseen:
        queue = [min(unseen)]; visited = set(queue)
        while queue:
            for k in adj[queue.pop()]:
                if k not in visited:
                    visited.add(k); queue.append(k)
        unseen.difference_update(visited)
        identity = 'family_'+digest(sorted(visited))
        groups[identity] = sorted(visited)
        mapping.update({k: identity for k in visited})
    check(groups == {c['source_family']: c['members'] for c in audit['components']}, 'BFS graph replay differs')
    return mapping


def environment():
    import av
    from scipy.optimize._highspy._core import _Highs
    return dict(at_utc=now(), python=platform.python_version(), platform=platform.platform(),
        executable_environment='.venv_temporal', packages={n: importlib.metadata.version(n) for n in
            ['av', 'numpy', 'scipy', 'Pillow', 'matplotlib']}, highs=_Highs().version(),
        linked_FFmpeg={k: list(v) for k, v in av.library_versions.items()},
        decoder=dict(thread_count=1, thread_type='NONE', skip_frame='NONE', hardware=False),
        GPU_used=False, CUDA_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        model_imports_in_population_process=[n for n in ['torch', 'pytorchvideo', 'causal_observability_runtime'] if n in __import__('sys').modules],
        free_disk_bytes=shutil.disk_usage(ROOT).free, new_dataset_network_bytes=0,
        network_note='No data requests implemented. One git fetch of main; 8MiB conservative metadata reserve, below20GiB.',
        installed_or_upgraded_packages=[])


def historical_verification():
    baseline = read(OUT/'historical_baseline.json')
    verify_hashes(baseline['sha256'])
    old = read(HISTORICAL/'population_design.json')
    with gzip.open(HISTORICAL/'eligibility_attempts.jsonl.gz', 'rt', encoding='utf-8') as f:
        attempts = [json.loads(line) for line in f]
    old_events = [json.loads(line) for line in (HISTORICAL/'encoder_events.jsonl').read_text().splitlines()]
    forbidden = [r for r in old_events if str(r.get('phase', '')).startswith('test')]
    check(not forbidden, 'Unexpected historical official-test encoder evidence')
    check(old['official_clips'] == len(attempts) and old['technically_eligible'] == sum(r['eligible'] for r in attempts),
          'Historical table/journal count disagreement')
    check(old['technical_failures'] == sum(not r['eligible'] for r in attempts), 'Historical failures differ')
    return dict(passed=True, immutable_files_verified=len(baseline['sha256']),
        baseline_sha256=sha(OUT/'historical_baseline.json'),
        original_clips=len(attempts), old_eligible=sum(r['eligible'] for r in attempts),
        old_failures=sum(not r['eligible'] for r in attempts),
        classes_below_original_70=len(old['deficient_classes']), old_catch_test=old['eligible_official_test_per_class']['catch'],
        old_test_encoder_events=len(forbidden), new_model_calls=0,
        previous_training_only_native400_exposure_disclosed=True,
        new_head_or_controller_training=False, historical_recognition_and_final_latency_results=False,
        v1_failure_stage='population_construction_before_hypothesis_testing')


def run_tests():
    """Only labeled synthetic fixtures, CPU-only; never invoke the v1 runner."""
    import sys
    records = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    for module in ['test_causal_observability_population_v2', 'test_causal_observability']:
        tick = time.perf_counter()
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', module, '-v'], cwd=ROOT,
                                env=env, capture_output=True, text=True, encoding='utf-8')
        record = dict(module=module, returncode=result.returncode, seconds=time.perf_counter()-tick,
                      stdout=result.stdout, stderr=result.stderr, fixtures='SYNTHETIC TEST FIXTURE only',
                      CUDA_disabled=True, empirical_training=False, scientific_completion_claim=False,
                      source_sha256=sha(ROOT/(module+'.py')))
        records.append(record)
        print(module, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
    save(OUT/'unit_tests.json', dict(passed=all(r['returncode'] == 0 for r in records), suites=records,
        caveat='Existing regression suite fits tiny CPU synthetic heads/networks; no real HMDB training or model inference'))
    check(all(r['returncode'] == 0 for r in records), 'Unit/regression tests failed; evidence retained')


def analyze(rows, classes, inventory):
    check(len(rows) == len(inventory['videos']), 'Complete screening not available')
    history = historical_verification()
    save(OUT/'historical_integrity.json', history)
    mapping, graph = family_audit(rows)
    class_counts = counts_by_class(rows, classes, mapping)
    write_csv(OUT/'per_class_eligibility.csv', class_counts)
    # Every per-clip grid and full actual PTS identities are in the compressed evidence.
    clips = []
    for r in rows:
        g = r.get('grid', {})
        durations = g.get('history_window_seconds', [])
        clips.append(dict(class_name=r['class_name'], filename=r['filename'], key=r['key'],
            official_status={0: 'unused', 1: 'train', 2: 'test'}[r['official_flag']],
            sha256=r['sha256'], codec=r.get('codec'), decoded_frame_count=r['decoded_frames'],
            valid_distinct_pts_count=r['valid_distinct_pts'], clip_pts_span=r.get('pts_span'),
            old_eligible=r['old_eligible'], new_eligible=r['new_eligible'],
            source_family=mapping[r['key']], title_family=r['title_family'],
            technical_exclusion=r['technical_exclusion'], old_exclusion=r['old_exclusion'],
            endpoint_indices=json.dumps(g.get('endpoint_indices', []), separators=(',', ':')),
            snippet_frame_indices=json.dumps(g.get('frame_indices', []), separators=(',', ':')),
            snippet_min_span=min(durations) if durations else None, snippet_max_span=max(durations) if durations else None,
            extended_candidates=sum(g.get('window_extended', [])), evidence='screening_evidence.jsonl.gz#'+r['key']))
    write_csv(OUT/'per_clip_eligibility.csv', clips)
    with gzip.open(HISTORICAL/'eligibility_attempts.jsonl.gz', 'rt', encoding='utf-8') as f:
        previous = {r['label']+'/'+r['clip']: r for r in map(json.loads, f)}
    differences = [r['key'] for r in rows if r['old_eligible'] != previous[r['key']]['eligible']]
    check(not differences, 'Recomputed historical eligibility differs; investigate before gate')
    extensions = [x for r in rows if r['new_eligible'] for x in r['grid']['window_extended']]
    spans = [x for r in rows if r['new_eligible'] for x in r['grid']['history_window_seconds']]
    nonuniform = sum(r['new_eligible'] and not np.allclose(np.diff(r['grid']['endpoint_timestamps']),
                     np.diff(r['grid']['endpoint_timestamps'])[0], rtol=0, atol=1e-9) for r in rows)
    temporal = dict(rule=RULE, actual_decoded_clips=len(rows), original_annotation_clips=len(inventory['videos']),
        old_eligible=sum(r['old_eligible'] for r in rows), new_eligible=sum(r['new_eligible'] for r in rows),
        newly_eligible=sum(r['new_eligible'] and not r['old_eligible'] for r in rows),
        new_failures=dict(Counter(r['technical_exclusion'] for r in rows if not r['new_eligible'])),
        old_comparison_differences=differences, independent_scalar_actual_frame_replays=sum(r['new_eligible'] for r in rows),
        original_decode_ordinals_retained=True, candidate_count=16, frames_per_snippet=4,
        selected_candidate_windows=len(extensions), extended_windows=sum(extensions),
        history_span_quantiles_seconds=np.percentile(spans, [0, 25, 50, 75, 100]).tolist() if spans else [],
        clips_with_temporally_nonuniform_endpoints=int(nonuniform),
        nonuniform_warning='Index-uniform endpoints may be temporally nonuniform, also through frame-index quantization at constant rate',
        genuine_full_PTS_evidence='screening_evidence.jsonl.gz; integer PTS/time bases and decode ordinals for all valid decoded clips',
        historical_recipe_equivalence_claim=False)
    save(OUT/'temporal_sampling_audit.json', temporal)
    grid = {}
    for name in CONFIGS:
        print('Global family allocation', name, flush=True)
        grid[name] = solve_configuration(rows, classes, mapping, class_counts, name)
        save(OUT/'quota_feasibility.json', grid)
    chosen = next((n for n in CONFIGS if grid[n]['status'] == 'FEASIBLE'), None)
    unresolved = [n for n in CONFIGS if grid[n]['status'] == 'UNRESOLVED_RESOURCE_LIMIT' and (not chosen or n < chosen)]
    decision = 'INCONCLUSIVE_RESOURCE_BLOCKED' if unresolved else ('GO' if chosen else 'NO_GO')
    manifest = allocation_manifest(rows, mapping, chosen, grid)
    save(OUT/'allocation_manifest.json', manifest)
    validation = validate_leakage(manifest, rows, inventory, mapping, grid)
    check(not environment()['model_imports_in_population_process'], 'Population process imported model runtime')
    save(OUT/'leakage_validation.json', validation)
    summary = dict(decision=decision, chosen_configuration=chosen, annotated_clips=len(rows), classes=len(classes),
        old_eligible=temporal['old_eligible'], new_eligible=temporal['new_eligible'],
        new_technical_exclusions=len(rows)-temporal['new_eligible'],
        old_technical_exclusions=len(rows)-temporal['old_eligible'],
        eligible_official_train=sum(r['new_eligible'] and r['official_flag'] == 1 for r in rows),
        eligible_official_test=sum(r['new_eligible'] and r['official_flag'] == 2 for r in rows),
        eligible_official_unused=sum(r['new_eligible'] and r['official_flag'] == 0 for r in rows),
        isolated_training_available=sum(c['training_available'] for c in class_counts),
        excluded_conflicting_families=graph['conflict_families'],
        excluded_eligible_training_clips=graph['eligible_training_excluded'],
        catch=next(c for c in class_counts if c['class_name'] == 'catch'),
        configurations={n: dict(status=g['status'], failed_necessary_constraints=len(g['necessary_condition_failures']),
                                affected_classes=len({p['class_name'] for p in g['necessary_condition_failures']})) for n, g in grid.items()},
        no_model_execution=True, official_test_encoder_calls=0, recognition_findings=None,
        feasibility_thresholds_are_not_power_guarantees=True, unresolved_searches=unresolved)
    save(OUT/'feasibility_summary.json', summary)
    make_figure(class_counts)
    write_reports(summary, class_counts, graph)
    save(OUT/'environment.json', environment())
    return summary


def make_figure(class_counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    labels = [c['class_name'] for c in class_counts]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 17), sharey=True, layout='constrained')
    for ax, role, title in [(axes[0], 'train', 'Official training: temporal eligibility'),
                            (axes[1], 'test', 'Official test: temporal eligibility')]:
        ax.barh(y-.18, [c['old_'+role+'_eligible'] for c in class_counts], height=.34, color='#969da5', label='Historical rule')
        ax.barh(y+.18, [c['new_'+role+'_eligible'] for c in class_counts], height=.34, color='#2879a8', label='Redesigned rule')
        ax.set_title(title, fontsize=12, loc='left', pad=12)
        ax.set_xlabel('Real eligible clips / class')
        ax.legend(loc='lower right', fontsize=9, framealpha=.95)
    a = np.array([c['training_available'] for c in class_counts])
    b = np.array([c['training_source_excluded'] for c in class_counts])
    d = np.array([c['new_train_temporal_excluded'] for c in class_counts])
    axes[2].barh(y, a, color='#31856c', label='Available after family isolation', height=.72)
    axes[2].barh(y, b, left=a, color='#df9846', label='Excluded: test-family relationship', height=.72)
    axes[2].barh(y, d, left=a+b, color='#c7cbd1', label='Excluded: technical/temporal', height=.72)
    for value, style in [(50, ':'), (40, '--'), (33, '-')]:
        axes[2].axvline(value, color='#4d4254', lw=.9, ls=style, alpha=.75)
    axes[2].set_title('Conservative training availability', fontsize=12, loc='left', pad=12)
    axes[2].set_xlabel('Real official-training clips / class\nLines: A total50, B total40, C total33; group minima also required', fontsize=9)
    axes[2].legend(loc='lower left', bbox_to_anchor=(-.02, -.095), fontsize=8, frameon=False)
    axes[0].set_yticks(y, labels, fontsize=9)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=.16)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='y', length=0)
    fig.suptitle('HMDB51 split 1: input-only population feasibility\nOriginal clips; no recognition outcomes or GPU inference', fontsize=17)
    folder = OUT/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'population_feasibility.png', dpi=180, facecolor='white')
    plt.close(fig)


def write_reports(summary, class_counts, graph):
    s = summary
    lines = ['# Experiment 19.1 findings', '', f"Population gate: **{s['decision']}**. No model was trained or executed in this study.", '',
        f"All {s['annotated_clips']:,} original authenticated clips across {s['classes']} classes were freshly decoded. Historical technical eligibility is {s['old_eligible']:,}; redesigned eligibility is {s['new_eligible']:,}. The old rule has {s['old_technical_exclusions']:,} exclusions; the new rule has {s['new_technical_exclusions']:,}. These counts are calculated from retained real frame evidence, not assumed planning totals.", '',
        f"Eligible official train/test/unused: {s['eligible_official_train']:,} / {s['eligible_official_test']:,} / {s['eligible_official_unused']:,}. Conservative test-priority isolation excludes {s['excluded_eligible_training_clips']:,} eligible training clips in {s['excluded_conflicting_families']} train/test-conflicting components, leaving {s['isolated_training_available']:,} available training clips. No eligible official test member is removed or moved.", '',
        f"For catch, old/new eligible test clips are {s['catch']['old_test_eligible']} / {s['catch']['new_test_eligible']}; new eligible training clips are {s['catch']['new_train_eligible']}, with {s['catch']['training_available']} remaining after family quarantine and {s['catch']['training_families']} training / {s['catch']['test_families']} test components.", '',
        '|Configuration|Status|Classes with necessary-condition failures|Failed necessary constraints|', '|---|---|---:|---:|']
    for n, g in s['configurations'].items():
        lines.append(f"|{n}|{g['status']}|{g['affected_classes']}|{g['failed_necessary_constraints']}|")
    lines += ['', f"Selected configuration: {s['chosen_configuration'] or 'none'}. Full solver options, repeated termination states and independent count certificates are in quota_feasibility.json. A timeout is never counted as infeasibility. The minima are engineering thresholds, not power guarantees.", '',
        '## Class-level evidence', '', '|Class|Old eligible|New eligible|Available train|Test clips|Train / test families|', '|---|---:|---:|---:|---:|---:|']
    lines += [f"|{r['class_name']}|{r['old_eligible']}|{r['new_eligible']}|{r['training_available']}|{r['new_test_eligible']}|{r['training_families']} / {r['test_families']}|" for r in class_counts]
    lines += ['', '## Interpretation and threats', '',
        'The temporal amendment recovers input coverage, not demonstrated recognition fidelity. Endpoints are uniform in presentation-frame index and may be nonuniform in time. Windows in very short clips can contain little action evolution. This is not equivalent to the inherited offset recipe and requires a separately reviewed recognition protocol.', '',
        f"The graph contains {len(graph['components'])} components and {len(graph['large_components'])} components with >=20 clips. Shared titles are uncertain proxies, sometimes broad titles/compilations. All original parent URLs/identifiers remain unavailable; different titles do not establish independence. No uncertain large family was split to rescue quotas. pHash is conservative finite-resolution detection, not an exhaustive duplicate guarantee. Exact-only / exact+pHash / title-only counts are labeled sensitivities and cannot justify GO.", '',
        'The historical blocked experiment stopped before hypothesis testing; it is not evidence that learning failed. Its two official-training videos had prior native400 mechanical outputs, disclosed in historical_integrity.json and the original exposure notes. No official HMDB test model output exists in either study. UCF offline Z_pool quality required dense features; its periodic timing does not establish a deployable causal advantage. No new claim about that advantage follows from this audit.', '',
        '## Claim and evidence', '', '|Claim|Evidence and limit|', '|---|---|',
        '|Genuine population and frame contract|archive_member_inventory.json, screening_evidence.jsonl.gz and independent scalar replay; no fabricated frames|',
        '|Technical eligibility change|per_clip_eligibility.csv, per_class_eligibility.csv, temporal_sampling_audit.json; no recognition outcomes|',
        '|Known leakage protection|source_family_audit.json and leakage_validation.json; external parent identity remains uncertain|',
        '|Allocation gate|quota_feasibility.json, independent necessary-count certificates / verified assignments; no power claim|',
        '|Historical preservation|historical_baseline.json and historical_integrity.json; all previous tracked result hashes checked|',
        '|Method accuracy or latency|Unmeasured in Experiment19.1; tests and population audit do not establish either|', '',
        '## Next decision', '']
    if s['decision'] == 'GO':
        lines += ['Review this population protocol first. Only a separate authorization should begin a new51-class affine head, equal-budget H/H+cheap controller training and development selection on the frozen allocated roles, followed by a new pretest seal and actual causal B4 rollouts before privileged diagnostics. This commit stops at the population gate.']
    elif s['decision'] == 'NO_GO':
        lines += ['Do not run the declared full51-class HMDB learning experiment. Sampling recovery cannot resolve the verified conservative source-allocation deficits. Do not split broad families, lower minima further or drop classes after this audit. Select a fresh independent dataset prospectively using the protocol’s declared source independence, full-class availability, genuine video/label access, temporal suitability, compute and licensing framework. No substitute dataset is acquired or selected using recognition outcomes here.']
    else:
        lines += ['Complete the documented genuine resource/solver evidence before deciding feasibility. No inference or new training is authorized.']
    (OUT/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (OUT/'decision.md').write_text(s['decision']+'\n\nSee findings.md and quota_feasibility.json for the evidence. This is a population gate, not a completed adaptive-policy experiment.\n', encoding='utf-8')
    commands = [r'& .\.venv_temporal\Scripts\python.exe -B -m unittest test_causal_observability_population_v2 -v',
                r'& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability_population_v2.py --screen-only --resume',
                r'& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability_population_v2.py --analyze-only --resume',
                r'& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability_population_v2.py --resume',
                r'& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability_population_v2.py --audit-only']
    (OUT/'README.md').write_text('# Experiment19.1 reproduction\n\n'+s['decision']+'; input-only, no GPU.\n\n'
        'Run from repository root in PowerShell. Original authenticated data/source/causal_observability_v1 bytes are required for screening. No download or package installation is implemented. Existing extraction is read-only.\n\n'
        '```powershell\n'+'\n'.join(commands)+'\n```\n\n'
        'The full screening command resumes completed per-clip checkpoints and rejects changed code/protocol/input identities. --analyze-only reads retained actual metadata without decoding; --resume runs feasible pending stages. After finalization, resume performs the same read-only audit. --audit-only reads/hashes evidence and replays saved frame algebra, graph links and allocation constraints; it never downloads, decodes, solves, trains, predicts or regenerates results. It also checks original local bytes when present.\n\n'
        'screening_evidence.jsonl.gz publishes lossless actual frame metadata (no pixels): integer PTS/time bases, presentation/decode order, full candidate grids, offsets, spans, eligibility and clip hashes. The append-only operational checkpoint remains ignored locally, inventoried in screening_identity.json. Public archive_member_inventory.json binds every AVI to original RAR members. SHA256SUMS.json locks all retained public evidence and execution code. No raw videos/checkpoints/archives are added to Git.\n\n'
        'See protocol.md for scientific amendments and stop rules, source_family_audit.json for every component/edge and title uncertainty, quota_feasibility.json for both deterministic solver executions per configuration, and findings.md for actual evidence and limitations.\n', encoding='utf-8')


def audit(require_final=True):
    """Genuinely read-only. No decoder, solver, network, models, or writes."""
    inventory_path = OUT/'SHA256SUMS.json'
    if require_final:
        check(inventory_path.exists(), 'Final evidence inventory missing; run pending stages')
        verify_hashes(read(inventory_path))
    identity = read(OUT/'screening_identity.json')
    verify_hashes(identity['identity']['code_sha256'])
    check(identity['identity']['protocol_sha256'] == sha(OUT/'protocol.md'), 'Protocol changed')
    check(identity['evidence_sha256'] == sha(OUT/'screening_evidence.jsonl.gz'), 'Screening evidence changed')
    inventory = read(OUT/'archive_member_inventory.json')
    rows = retained_records()
    check(len(rows) == identity['clips'] == len(inventory['videos']), 'Incomplete real screening')
    classes, metadata = original_metadata()
    check([{k: r[k] for k in m} for r, m in zip(rows, metadata)] == metadata, 'Saved/original metadata differ')
    graph = read(OUT/'source_family_audit.json')
    mapping = graph_replay(rows, graph)
    class_counts = counts_by_class(rows, classes, mapping)
    grid = read(OUT/'quota_feasibility.json')
    for name, g in grid.items():
        check(necessary_failures(class_counts, CONFIGS[name]) == g['necessary_condition_failures'], 'Infeasibility certificate changed')
        if g['status'] == 'FEASIBLE':
            check(validate_assignment(rows, classes, mapping, g['assignments'], CONFIGS[name]) == g['independent_counts'],
                  'Independent solved counts differ')
        if g['status'] == 'PROVEN_INFEASIBLE':
            check(bool(g['necessary_condition_failures']) or all(r['status'] == 2 for r in g['solver']['runs']),
                  'Infeasibility lacks solver/certificate evidence')
        runs = g['solver']['runs']
        check(runs[0]['status'] == runs[1]['status'] and runs[0]['assignment_digest'] == runs[1]['assignment_digest'],
              'Deterministic repeat receipt differs')
    manifest = read(OUT/'allocation_manifest.json')
    leak = validate_leakage(manifest, rows, inventory, mapping, grid)
    check(leak == read(OUT/'leakage_validation.json'), 'Independent leakage validation differs')
    history = historical_verification()
    check(history == read(OUT/'historical_integrity.json'), 'Historical verification changed')
    # Bind local original members, when retained, without opening a decoder.
    checked = 0
    for r in rows:
        p = ORIGINAL/'videos'/r['key']
        check(p.is_file() and sha(p) == r['sha256'], 'Original local video missing/changed: '+r['key'])
        checked += 1
    verify_hashes(inventory['archive_sha256'])
    verify_hashes(inventory['inner_archive_sha256'])
    for n, info in inventory['annotations'].items():
        check(sha(ORIGINAL/'splits'/n) == info['sha256'], 'Original annotation changed')
    summary = read(OUT/'feasibility_summary.json')
    check(summary['old_eligible'] == sum(r['old_eligible'] for r in rows) and
          summary['new_eligible'] == sum(r['new_eligible'] for r in rows), 'Summary totals changed')
    expected_chosen = next((n for n in CONFIGS if grid[n]['status'] == 'FEASIBLE'), None)
    unresolved = [n for n in CONFIGS if grid[n]['status'] == 'UNRESOLVED_RESOURCE_LIMIT' and (not expected_chosen or n < expected_chosen)]
    expected_decision = 'INCONCLUSIVE_RESOURCE_BLOCKED' if unresolved else ('GO' if expected_chosen else 'NO_GO')
    check(summary['decision'] == expected_decision and summary['chosen_configuration'] == expected_chosen, 'Population decision changed')
    check(read(OUT/'unit_tests.json')['passed'], 'Unit tests not passed')
    check(not environment()['model_imports_in_population_process'], 'Audit imported model runtime')
    return dict(passed=True, decision=expected_decision, genuine_clips_verified=len(rows),
        local_original_AVI_hashes_verified=checked, independent_scalar_frame_replays=sum(r['new_eligible'] for r in rows),
        historical_files_verified=history['immutable_files_verified'], read_only=True,
        downloads=0, decodes=0, solver_calls=0, model_calls=0, regenerated_outputs=0)


def finalize():
    check((OUT/'figure_review.json').exists() and read(OUT/'figure_review.json')['actually_inspected'], 'Rendered figure review required')
    report = audit(require_final=False)
    save(OUT/'verification.json', report)
    save(OUT/'status.json', dict(stage='population_gate', status='verified', decision=report['decision'], at_utc=now(),
        model_training='not_authorized', official_test_inference='not_authorized', GPU='not_used'))
    files = {rel(p): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['SHA256SUMS.json', 'run.lock']}
    files.update({p: sha(ROOT/p) for p in IMPLEMENTATION + ['test_causal_observability_population_v2.py']})
    save(OUT/'SHA256SUMS.json', files)
    return audit()
