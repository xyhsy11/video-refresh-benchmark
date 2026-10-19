"""Input-only HMDB51 eligibility, conservative families and role constraints."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import numpy as np
from causal_observability_common import *


def source_family(name, label):
    token = '_'+label+'_'
    if token not in name:
        return 'unknown:'+name, None
    title = name.rsplit(token, 1)[0]
    return re.sub('[^a-z0-9]+', '', title.casefold()), title


def metadata():
    path = OUT/'official_membership.json'
    if path.exists(): return read(path)
    directory = DATA/'splits'; directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar.exe', '-xf', str(DATA/'test_train_splits.rar'), '-C', str(directory)], check=True)
    rows, classes, seen = [], [], set()
    for p in sorted(directory.rglob('*_test_split1.txt')):
        label = p.name.removesuffix('_test_split1.txt'); classes.append(label)
        counts = Counter()
        for line in p.read_text().splitlines():
            name, flag = line.rsplit(None, 1); flag = int(flag); counts[flag] += 1
            if (label, name) in seen or flag not in [0, 1, 2]: raise ValueError('Invalid official metadata')
            seen.add((label, name)); family, title = source_family(name, label)
            rows.append(dict(label=label, clip=name, official_flag=flag, source_family=family, source_title=title))
        if counts[1] != 70 or counts[2] != 30: raise Blocker('Unexpected official split-1 counts: '+label)
    if len(classes) != 51: raise Blocker('Expected all 51 class annotation files')
    result = dict(at_utc=now(), classes=classes, videos=rows,
        annotation_sha256={rel(p): sha(p) for p in sorted(directory.rglob('*.txt'))},
        source_rule='Prefix before final _class_ token; casefold ASCII-alphanumeric normalization globally, conservative source proxies; unknown prefixes become explicit unknown singleton families',
        source_unknown=sum(v['source_title'] is None for v in rows))
    save(path, result); return result


def extract():
    marker = DATA/'extraction.json'
    if marker.exists():
        if read(marker)['archive_sha256'] != sha(DATA/'hmdb51_org.rar'): raise Blocker('Archive changed')
        return DATA/'videos'
    if shutil.disk_usage(ROOT).free < 20*2**30: raise Blocker('Need 20GiB free before full extraction')
    outer, dest = DATA/'class_archives', DATA/'videos'
    outer.mkdir(parents=True, exist_ok=True); dest.mkdir(exist_ok=True)
    def safe_extract(archive, folder):
        listing = subprocess.check_output(['tar.exe', '-tf', str(archive)], text=True, encoding='utf-8', errors='replace')
        for n in listing.splitlines():
            q = PurePosixPath(n.replace('\\', '/'))
            if q.is_absolute() or '..' in q.parts or ':' in n: raise Blocker('Unsafe archive member')
        subprocess.run(['tar.exe', '-xf', str(archive), '-C', str(folder)], check=True)
    safe_extract(DATA/'hmdb51_org.rar', outer)
    parts = sorted(outer.rglob('*.rar'))
    if len(parts) != 51: raise Blocker('Expected 51 original class archives')
    for part in parts:
        safe_extract(part, dest)
    save(marker, dict(at_utc=now(), archive_sha256=sha(DATA/'hmdb51_org.rar'),
        class_archives={rel(p): sha(p) for p in parts}, AVI_files=len(list(dest.rglob('*.avi'))), CRC_extraction_passed=True))
    return dest


def inspect(path):
    import av
    from research_utils import configure_decoder
    from temporal_refresh_common import snippet_grid
    from ucf101_proxy_data import presentation_order
    from scipy.fft import dctn
    from PIL import Image
    frames, pts, ids = [], [], []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]; configure_decoder(stream)
        codec = stream.codec_context.name
        for i, frame in enumerate(c.decode(stream)):
            if frame.pts is None or frame.time_base is None: raise ValueError('Missing actual PTS')
            frames.append(frame); pts.append(float(frame.pts*frame.time_base))
            ids.append(dict(index=i, pts=int(frame.pts), time_base=str(frame.time_base)))
        frames, pts, ids, reordered = presentation_order(frames, pts, ids)
        grid = snippet_grid(pts)
        grid['frame_identities'] = [[ids[j] for j in row] for row in grid['frame_indices']]
        fingerprints = []
        for q in [.1, .5, .9]:
            a = frames[int(q*(len(frames)-1))].to_ndarray(format='gray')
            low = dctn(np.asarray(Image.fromarray(a).resize((32, 32)), dtype=float), norm='ortho')[:8, :8].ravel()[1:]
            bits = low > np.median(low); fingerprints.append(''.join(map(lambda x:'1' if x else '0', bits)))
    return dict(codec=codec, decoded_frames=len(pts), pts_span=pts[-1]-pts[0], snippet_grid=grid,
        fingerprints=fingerprints, presentation_reordered_frames=reordered)


def isolated_partition(rows, classes):
    """Exact quotas via family-level MILP; never split a connected family."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    families = sorted({r['source_family'] for r in rows}, key=lambda f: rank('partition', f))
    counts = np.zeros((len(classes), len(families)))
    for r in rows: counts[classes.index(r['label']), families.index(r['source_family'])] += 1
    if not np.all(counts.sum(1) == 70):
        raise Blocker('Exact 45/15/10 training quotas require 70 eligible, isolated official train clips/class')
    nf = len(families); a = np.zeros((nf+3*len(classes), 3*nf))
    for i in range(nf): a[i, [i, nf+i, 2*nf+i]] = 1
    targets = [*np.ones(nf)]
    for j, quota in enumerate([45, 15, 10]):
        a[nf+j*len(classes):nf+(j+1)*len(classes), j*nf:(j+1)*nf] = counts
        targets.extend([quota]*len(classes))
    cost = np.array([int(rank('role_cost', family, j)[:12], 16)/16**12 for j in range(3) for family in families])
    result = milp(cost, integrality=np.ones(3*nf), bounds=Bounds(0, 1),
        constraints=LinearConstraint(a, targets, targets), options={'time_limit': 120, 'mip_rel_gap': 0})
    if not result.success: raise Blocker('Family-level exact-quota assignment infeasible or unproven within fixed solver limit')
    assignment = np.rint(result.x).reshape(3, nf)
    np.testing.assert_allclose(a @ assignment.ravel(), targets, atol=0, rtol=0)
    roles = ['head_train', 'controller_train', 'development']
    return {f: roles[int(assignment[:, i].argmax())] for i, f in enumerate(families)}


def eligibility(plan):
    target = OUT/'eligible_manifest.json'
    if target.exists():
        result=read(target);validate_manifest(result)
        if result['classes']!=plan['classes']:raise Blocker('Official class order changed')
        return result
    if (OUT/'population_design.json').exists() and (DATA/'eligible_input_population.json').exists():
        previous=read(OUT/'population_design.json')
        if previous['deficient_classes']:
            raise Blocker(f'Fixed population design remains infeasible in {len(previous["deficient_classes"])} classes; original completed screening retained. A new population/candidate rule needs an explicit pre-outcome amendment, not silent resume changes.')
    dest = extract(); progress = DATA/'eligibility_progress.json'
    attempts = read(progress) if progress.exists() else {}
    journal = DATA/'eligibility_attempts.jsonl'
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            r = json.loads(line); attempts[r['label']+'/'+r['clip']] = r
    lookup = {(p.parent.name, p.name): p for p in dest.rglob('*.avi')}
    if len(lookup) < len(plan['videos']): raise Blocker('Original extracted archive lacks metadata clips')
    for i, v in enumerate(plan['videos']):
        key = v['label']+'/'+v['clip']
        if key in attempts: continue
        path = lookup.get((v['label'], v['clip']))
        try:
            if path is None: raise ValueError('Missing original clip')
            record = dict(**v, path=rel(path), sha256=sha(path), **inspect(path), eligible=True)
        except Exception as exc:
            record = dict(**v, eligible=False, reason=type(exc).__name__+': '+str(exc))
        attempts[key] = record
        with journal.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        if (i+1) % 100 == 0: print(f'Input-only HMDB screening {i+1}/{len(plan["videos"])}', flush=True)
    save(progress, attempts)
    rows = list(attempts.values())
    # Families/exact hashes are joined globally, including metadata role0, before roles.
    parent = {r['source_family']: r['source_family'] for r in rows}
    def root(f):
        while parent[f] != f: parent[f] = parent[parent[f]]; f = parent[f]
        return f
    def join(a, b):
        a, b = root(a), root(b)
        if a != b: parent[max(a, b)] = min(a, b)
    by_hash = defaultdict(list)
    for r in rows:
        if r.get('sha256'): by_hash[r['sha256']].append(r)
    exact_edges = []
    for group in by_hash.values():
        for r in group[1:]:
            join(group[0]['source_family'], r['source_family']); exact_edges.append([group[0]['clip'], r['clip']])
    # Fixed conservative three-frame pHash screen, calibrated/described on train input only.
    # All three Hamming distances<=4, duration ratio>=.9; no outcome/manual retain override.
    good = [r for r in rows if r['eligible']]
    bits = np.array([[int(q, 2) for q in r['fingerprints']] for r in good], dtype=np.uint64)
    near_edges = []
    for i, r in enumerate(good):
        for j in range(i):
            s = good[j]
            if root(r['source_family']) == root(s['source_family']): continue
            if min(r['pts_span'], s['pts_span'])/max(r['pts_span'], s['pts_span']) < .9: continue
            ds = [int(a ^ b).bit_count() for a, b in zip(bits[i], bits[j])]
            if max(ds) <= 4:
                join(r['source_family'], s['source_family']); near_edges.append(dict(first=r['clip'], second=s['clip'], hamming=ds))
    for r in rows: r['source_family'] = root(r['source_family'])
    test_families = {r['source_family'] for r in rows if r['official_flag'] == 2}
    excluded, retained = [], []
    for r in rows:
        if not r['eligible']: excluded.append(dict(label=r['label'], clip=r['clip'], reason=r['reason'])); continue
        if r['official_flag'] == 0: continue
        if r['official_flag'] == 1 and r['source_family'] in test_families:
            excluded.append(dict(label=r['label'], clip=r['clip'], reason='Official test family takes priority; exclude related training member')); continue
        retained.append(r)
    train = [r for r in retained if r['official_flag'] == 1]
    counts = {c: sum(r['label'] == c for r in train) for c in plan['classes']}
    save(OUT/'eligibility_exclusions.json', excluded)
    save(OUT/'duplicate_and_source_audit.json', dict(at_utc=now(),exact_edges=exact_edges,near_edges=near_edges,
        fixed_pHash_rule='all three 63-bit hashes <=4 bit difference; duration ratio>=.9; conservative connected families; fixed before model outcomes',
        training_calibration_population=sum(r['official_flag']==1 for r in good),manual_retain_overrides=[],
        source_titles_missing=plan['source_unknown'],external_parent_source_IDs_missing=len(rows),unknown_overlap='Actual source URLs/YouTube IDs and checkpoint pretraining mappings unavailable; title proxies do not establish universal independence',
        test_priority_policy='Keep eligible official test membership; exclude conflicting official training members, never move data across roles'))
    summary = dict(at_utc=now(),classes=plan['classes'],official_clips=len(rows),technically_eligible=sum(r['eligible'] for r in rows),
        exact_required_per_class=dict(head_train=45,controller_train=15,development=10),isolated_training_per_class=counts,
        eligible_official_test_per_class={c:sum(r['label']==c and r['official_flag']==2 for r in retained) for c in plan['classes']},
        deficient_classes=[c for c,n in counts.items() if n<70],technical_failures=sum(not r['eligible'] for r in rows),
        test_model_calls=0)
    save(OUT/'population_design.json',summary)
    save(DATA/'eligible_input_population.json',dict(classes=plan['classes'],videos=retained))
    if summary['deficient_classes']:
        raise Blocker(f'Population design: {len(summary["deficient_classes"])} classes have fewer than 70 technically eligible, source-isolated official training clips; exact45/15/10 and all51 cannot both be satisfied. No model-outcome substitution permitted.')
    assignment = isolated_partition(train,plan['classes'])
    for r in retained: r['role']='test' if r['official_flag']==2 else assignment[r['source_family']]
    result = dict(classes=plan['classes'],videos=retained,at_utc=now())
    validate_manifest(result);save(target,result);return result


def assert_roles(rows):
    families, hashes = {}, {}
    for r in rows:
        for key, registry in [(r['source_family'],families),(r['sha256'],hashes)]:
            if key in registry and registry[key] != r['role']: raise ValueError('Cross-role family/duplicate leakage')
            registry[key]=r['role']


def validate_manifest(manifest):
    classes=manifest['classes'];rows=manifest['videos']
    if len(classes)!=51 or len(set(classes))!=51:raise Blocker('Fixed experiment requires all51 classes')
    assert_roles(rows)
    for c in classes:
        for role,quota in [('head_train',45),('controller_train',15),('development',10)]:
            if sum(v['label']==c and v['role']==role for v in rows)!=quota:
                raise Blocker('Cached manifest violates fixed population quotas')
        if not any(v['label']==c and v['role']=='test' for v in rows):raise Blocker('Class lacks eligible official test support')
    for v in rows:
        expected=2 if v['role']=='test' else 1
        if v['official_flag']!=expected:raise Blocker('Official source roles changed')
