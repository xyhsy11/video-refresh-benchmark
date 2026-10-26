"""Post-run fitting audits and handoff only; never executes empirical fitting."""
import argparse,json,os,subprocess,sys,time
from native_head_value_confirmation import read,save,sha
from native_head_confirmation_state import totals,metered
from run_native_head_value_confirmation import ROOT,OUT,run_lock,event,now


def timed_command(args,label,outer_meter=False):
    before=totals();start=time.perf_counter()
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-B',*args]
    if outer_meter:
        with run_lock(),metered('cpu',label,command=['python','-B',*args]):
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    else:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    wall=time.perf_counter()-start;after=totals()
    accounted=sum(after[k]-before[k] for k in ('collection_seconds','cpu_seconds','timing_seconds'))
    with run_lock():
        event('task1-delivery-command-residual',cpu_seconds=max(0,wall-accounted),label=label,
            command=['python','-B',*args],exit_code=result.returncode,
            basis='Measured subprocess wall minus already metered exclusive phase wall; no wait/sleep time')
    output=OUT/'task1'/(label+'.txt');output.write_bytes(result.stdout)
    record=dict(command=['python','-B',*args],exit_code=result.returncode,elapsed_seconds=wall,
        output=output.relative_to(OUT).as_posix(),output_sha256=sha(output),label=label)
    print(json.dumps(record),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observed-control-seconds',type=float,required=True)
    args=parser.parse_args()
    if args.observed_control_seconds<0:raise ValueError('Control wall must be nonnegative')
    with run_lock():
        status=read(OUT/'task1_status.json')
        if status['technical']=='RUNNING' or totals()['uncertain_operations']:raise RuntimeError('Production run must stop cleanly before post-run audits')
        edges=read(OUT/'task1/production_process_time.json')
        event('task1-production-process-edge-wall',cpu_seconds=edges['startup_wall_seconds']+edges['post_session_upper_seconds'],
            startup_wall_seconds=edges['startup_wall_seconds'],post_session_upper_seconds=edges['post_session_upper_seconds'],
            receipt_sha256=sha(OUT/'task1/production_process_time.json'),
            basis='Measured OS-birth-to-session-start wall plus conservative session-end-to-first-stopped-observation upper bound. No model calls added.')
        event('task1-observed-external-control-work',cpu_seconds=args.observed_control_seconds,
            basis='Sum of measured inspection/monitor command wall from tool receipts; excludes all sleeps, production execution and already metered validation. Git publication is outside fitting-work accounting.')
    records=[]
    commands=[(['audit_native_head_task1_checkpoints.py'],'task1-final-checkpoint-audit',False),
        (['verify_native_head_task1_resume.py'],'task1-final-resume-check',False)]
    if (OUT/'fit_oof_state_predictions.npz').exists():commands.append((['report_native_head_task1_diagnostics.py'],'task1-final-development-diagnostics',False))
    # Focused boundary tests need no active production operation. Their entire
    # measured subprocess wall is charged by timed_command after completion.
    commands.extend([(['-m','unittest','test_native_head_task1','-v'],'task1-final-focused-tests',False),
        (['validate_release.py'],'task1-final-release-validation',True),
        (['propose_native_head_task1_resources.py'],'task1-final-resource-proposal',False)])
    for command,label,outer in commands:
        record=timed_command(command,label,outer);records.append(record)
        save(OUT/'task1/publication_checks.json',dict(at_utc=now(),commands=records,
            passed=all(r['exit_code']==0 for r in records),controller_sha256=sha(__file__)))
        if record['exit_code'] and label not in ('task1-final-focused-tests','task1-final-release-validation'):
            raise RuntimeError('Post-run audit/report failed; inspect preserved output before continuing: '+label)
    # This last command binds a completed-ledger snapshot, so do not append a
    # timing event after it; its own preparation is already internally metered.
    result=subprocess.run([sys.executable,'-B','write_native_head_task1_handoff.py'],cwd=ROOT)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
