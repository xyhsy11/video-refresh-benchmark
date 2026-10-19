"""Resume real UCF DEVELOPMENT causal policy execution, never official test."""
from __future__ import annotations
import argparse
import json
import os
import sys
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import ucf101_causal_development_common as c
import ucf101_causal_development as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--finalize-reviewed',action='store_true');args=parser.parse_args()
    c.GUARD.install()
    if args.audit_only or (c.OUT/'SHA256SUMS.json').exists():
        print(json.dumps(study.audit(),indent=2));return 0
    c.OUT.mkdir(parents=True,exist_ok=True);lock=c.OUT/'run.lock'
    try:
        with lock.open('x') as f:f.write(str(os.getpid()))
    except FileExistsError:raise c.Blocker('Run lock exists: verify PID has exited before removing stale lock; preserve attempts')
    try:
        if args.finalize_reviewed:
            c.require(c.read(c.OUT/'figure_review.json')['actually_inspected'],'Rendered figure review missing')
            # Audit in a fresh CPU process, avoiding the training/GPU process entirely.
            r=__import__('subprocess').run([sys.executable,'-B','-c',
                'import json; import ucf101_causal_development_common as c; c.GUARD.install(); import ucf101_causal_development as s; print(json.dumps(s.audit(final=False)))'],capture_output=True,text=True,encoding='utf-8')
            c.require(r.returncode==0,'Read-only verification failed: '+r.stderr)
            c.save(c.OUT/'verification.json',json.loads(r.stdout));c.save(c.OUT/'status.json',dict(stage='final',status='verified',scope='DEVELOPMENT/EXPLORATORY',at_utc=c.now()))
            inventory={c.rel(p):c.sha(p) for p in c.OUT.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS.json','run.lock']}
            inventory.update({p:c.sha(c.ROOT/p) for p in c.CODE+['test_ucf101_causal_development.py']});c.save(c.OUT/'SHA256SUMS.json',inventory)
            print(r.stdout);return 0
        manifest,head=c.prepare();study.unit_tests()
        c.save(c.OUT/'status.json',dict(stage='real_training_smoke',status='running',scope='DEVELOPMENT/EXPLORATORY'))
        encoder=study.encoder_factory();study.smoke(manifest,encoder,head)
        c.save(c.OUT/'status.json',dict(stage='training_selection',status='running'))
        selection=study.train(manifest,head);study.freeze(manifest)
        c.save(c.OUT/'status.json',dict(stage='actual_causal_execution',status='running'))
        study.run_quality(manifest,encoder,head,selection)
        c.save(c.OUT/'status.json',dict(stage='dense_diagnostics',status='running'))
        study.privileged_diagnostics(manifest,encoder,head,selection)
        c.save(c.OUT/'status.json',dict(stage='actual_timing',status='running'))
        study.timing(manifest,encoder,head,selection)
        study.analysis(manifest,head);c.save(c.OUT/'execution_accounting.json',study.accounting());study.report(manifest);study.figures()
        c.save(c.OUT/'status.json',dict(stage='figures',status='awaiting_rendered_review',scope='DEVELOPMENT/EXPLORATORY'))
        print('Actual DEVELOPMENT evidence complete; inspect figures then --finalize-reviewed.',flush=True);return 0
    except Exception as e:
        c.save(c.OUT/'execution_accounting.json',study.accounting())
        c.save(c.OUT/'execution_blocker.json',dict(at_utc=c.now(),exception=repr(e),scope='DEVELOPMENT/EXPLORATORY',
            resume=r'& .\.venv_temporal\Scripts\python.exe -B .\run_ucf101_causal_development.py --resume'))
        c.save(c.OUT/'status.json',dict(stage='execution',status='blocked_or_incomplete',reason=repr(e)));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':sys.exit(main())
