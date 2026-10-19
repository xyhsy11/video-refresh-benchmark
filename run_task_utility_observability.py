"""CPU-only Experiment22; never launches a video encoder, decoder or controller."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
import sys
import task_utility_observability_study as s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--finalize-reviewed',action='store_true')
    args=parser.parse_args();s.GUARD.install()
    if args.audit_only or (s.OUT/'verification.json').exists():
        print(json.dumps(s.audit(),indent=2));return
    if args.finalize_reviewed:
        s.finalize();return
    s.OUT.mkdir(parents=True,exist_ok=True);s.DATA.mkdir(parents=True,exist_ok=True)
    lock=s.OUT/'run.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        s.extract()
        s.save(s.OUT/'status.json',dict(stage='targets',status='complete',scope=s.SCOPE))
        s.target_verification();s.train_probes()
        s.save(s.OUT/'status.json',dict(stage='CPU_probes',status='complete',scope=s.SCOPE))
        s.analyze_predictions();s.bootstrap();s.alignment_and_headroom();s.costs();s.figures()
        s.save(s.OUT/'status.json',dict(stage='figures',status='awaiting_rendered_review',scope=s.SCOPE))
        print('CPU analysis complete. Review genuine evidence/figures, write findings/decision, then --finalize-reviewed.',flush=True)
    except Exception as e:
        s.save(s.OUT/'incomplete_execution.json',dict(at_utc=s.now(),exception=repr(e),resume='python -B run_task_utility_observability.py --resume'))
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
