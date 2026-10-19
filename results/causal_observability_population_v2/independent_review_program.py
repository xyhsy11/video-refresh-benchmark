"""Read-only REAL retained-metadata review, independent scalar pHash pair search.

Run from repository root: python -B results/causal_observability_population_v2/independent_review_program.py
No decoding, models, optimization, files written, or synthetic research evidence.
"""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json

OUT = Path(__file__).resolve().parent


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    with gzip.open(OUT/'screening_evidence.jsonl.gz', 'rt', encoding='utf-8') as f:
        rows = list(map(json.loads, f))
    audit = load('source_family_audit.json')
    saved = {tuple(sorted([e['first'], e['second']])) for e in audit['edges'] if e['kind'] == 'near'}
    expected = set()
    fingerprints = [[int(h, 2) for h in r['fingerprints']] for r in rows]
    pairs_checked = 0
    for i, a in enumerate(rows):
        if len(fingerprints[i]) != 3 or a.get('pts_span', 0) <= 0:
            continue
        for j in range(i):
            b = rows[j]
            if len(fingerprints[j]) != 3 or b.get('pts_span', 0) <= 0:
                continue
            pairs_checked += 1
            if (fingerprints[i][0]^fingerprints[j][0]).bit_count() > 4:
                continue
            if min(a['pts_span'], b['pts_span'])/max(a['pts_span'], b['pts_span']) < .9:
                continue
            if all((x^y).bit_count() <= 4 for x, y in zip(fingerprints[i], fingerprints[j])):
                expected.add(tuple(sorted([a['key'], b['key']])))
    assert saved == expected, 'Independent all-pairs pHash edges differ'
    lookup = {r['key']:r for r in rows}
    family_for = {k:c['source_family'] for c in audit['components'] for k in c['members']}
    for field in ['sha256', 'title_family']:
        grouped = defaultdict(set)
        for r in rows:
            grouped[r[field]].add(family_for[r['key']])
        assert all(len(s) == 1 for s in grouped.values()), 'Missing exact/title connectivity'
    manifest = load('allocation_manifest.json')
    expected_test = {r['key'] for r in rows if r['new_eligible'] and r['official_flag'] == 2}
    assert {r['key'] for r in manifest['videos'] if r['role'] == 'test'} == expected_test
    classes = sorted({r['class_name'] for r in rows})
    assert len(classes) == 51
    forbidden_training_families = {family_for[r['key']] for r in rows if r['official_flag'] == 2}
    available = [r for r in rows if r['new_eligible'] and r['official_flag'] == 1
                 and family_for[r['key']] not in forbidden_training_families]
    count = Counter(r['class_name'] for r in available)
    counts = {}
    for c in classes:
        groups = {family_for[r['key']] for r in available if r['class_name'] == c}
        test = [lookup[k] for k in expected_test if lookup[k]['class_name'] == c]
        counts[c] = dict(train=count[c], train_families=len(groups), test=len(test),
                         test_families=len({family_for[r['key']] for r in test}))
    independent_bad = {}
    for config, minima in [('A', (30, 10, 10)), ('B', (25, 10, 5)), ('C', (20, 8, 5))]:
        bad = sorted(c for c, v in counts.items() if v['train'] < sum(minima) or v['train_families'] < 9
                     or v['test'] < 10 or v['test_families'] < 5)
        saved_bad = sorted({r['class_name'] for r in load('quota_feasibility.json')[config]['necessary_condition_failures']})
        assert bad == saved_bad, 'Independent impossibility certificate differs'
        independent_bad[config] = bad
    result = dict(passed=True, scope='REAL retained metadata; independent scalar all-pairs pHash and necessary-count review',
        original_clips=len(rows), all_pairs_checked=pairs_checked, exact_near_edges=len(expected),
        missing_near_edges=0, extra_near_edges=0, eligible_test_unchanged=len(expected_test),
        independent_infeasible_classes=independent_bad, per_class_counts=counts,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decodes=0, model_calls=0, file_writes=0)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
