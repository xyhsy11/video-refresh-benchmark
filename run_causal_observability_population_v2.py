"""Experiment19.1 CPU input-only screening and population gate; never X3D."""
from __future__ import annotations
import argparse
import json
import os
import sys

# Set before numpy/scipy imports. No CUDA runtime is imported by this runner.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'

import causal_observability_population_v2 as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--screen-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--finalize-reviewed', action='store_true')
    args = parser.parse_args()
    if args.audit_only or (p.OUT/'SHA256SUMS.json').exists():
        print(json.dumps(p.audit(), indent=2))
        return 0
    p.OUT.mkdir(parents=True, exist_ok=True)
    lock = p.OUT/'run.lock'
    try:
        with lock.open('x') as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        raise p.PopulationError('Run lock exists; preserve evidence and confirm recorded PID exited before removing stale lock')
    try:
        if args.finalize_reviewed:
            print(json.dumps(p.finalize(), indent=2))
            return 0
        classes, metadata, inventory = p.prepare()
        p.save(p.OUT/'environment.json', p.environment())
        if args.analyze_only:
            rows = p.retained_records()
            p.check(len(rows) == len(metadata), 'Screening incomplete; resume full screening first')
        else:
            rows = p.screening(metadata, inventory)
        if args.screen_only:
            p.save(p.OUT/'status.json', dict(stage='screening', status='complete', clips=len(rows), at_utc=p.now()))
            return 0
        p.run_tests()
        summary = p.analyze(rows, classes, inventory)
        p.save(p.OUT/'status.json', dict(stage='population_gate', status='awaiting_rendered_figure_review',
                                       decision=summary['decision'], at_utc=p.now()))
        print(json.dumps(summary, indent=2))
        print('Input-only evidence complete; rendered PNG inspection and --finalize-reviewed remain.')
        return 0
    except Exception as e:
        p.save(p.OUT/'execution_blocker.json', dict(at_utc=p.now(), exception=type(e).__name__, reason=str(e),
            decision='INCONCLUSIVE_RESOURCE_BLOCKED', pending='Resume genuine input-only screening/analysis; no model execution'))
        p.save(p.OUT/'status.json', dict(stage='incomplete', status='blocked', decision='INCONCLUSIVE_RESOURCE_BLOCKED', reason=str(e)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.exit(main())
