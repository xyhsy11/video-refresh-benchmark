"""Exercise safe control boundaries using authentic saved Task 1 artifacts."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
import fit_plan_native_head_confirmation as task
from native_head_value_confirmation import read,save,folds,VIEWS
from native_head_confirmation_state import totals,metered
from run_native_head_value_confirmation import run_lock


def verify():
    before=totals();rows=read(task.OUT/'val_manifest.json.gz');fs=folds(rows)
    found=[f for f in range(4) if (task.TASK/'folds'/f'{f}.json').exists()]
    predictions={t:{v:np.full(len(rows)*12,np.nan) for v in VIEWS} for t in ('advantage','drift')}
    with (patch.object(task.NativeEncoder,'forward',side_effect=AssertionError('unexpected native execution')),
          patch.object(task,'fit_models',side_effect=AssertionError('unexpected probe refit'))):
        for f in found:
            if not task.load_fold(f,fs,predictions,rows):raise AssertionError('completed fold not reused')
            hold=np.flatnonzero(np.repeat(fs==f,12))
            with np.load(task.TASK/'folds'/f'{f}.npz',allow_pickle=False) as saved:
                for target,views in predictions.items():
                    for view,a in views.items():np.testing.assert_array_equal(a[hold],saved[target+'_'+view])
        # Use actual certified input and reference hashes; no fabricated effects.
        actual=read(task.OUT/'smoke_manifest.json')[0]
        with tempfile.TemporaryDirectory() as folder,patch.object(task,'TASK',Path(folder)):
            first=task.Session(10)
            if not task.authenticate_reference(actual):raise AssertionError('authentic cached reference missing')
            first.next_operation='next pending source after authentic cached-source verification'
            first.checkpoint('RUNNING')
            try:first.before(first.next_operation,11)
            except task.PausedSession:pass
            else:raise AssertionError('safe pause not reached')
            saved=read(Path(folder)/'sessions'/(first.id+'.json'))
            if saved['status']!='PAUSED_SESSION' or saved['counters']!=before:raise AssertionError('pause counters changed')
            second=task.Session(10)
            if second.record['counters_before']!=before:raise AssertionError('new control session reset counters')
            if not task.authenticate_reference(actual):raise AssertionError('cached reference not reusable after pause')
    after=totals()
    if before!=after:raise AssertionError('control replay changed empirical or cumulative counters')
    return dict(passed=True,completed_folds_reloaded=found,heldout_source_rows_per_fold=800,
        heldout_state_rows_per_fold=9600,native_forwards=0,probe_fits=0,cumulative_counters_unchanged=True,
        safe_pause_and_new_session_counter_preservation=True,
        scope='Control tests on authentic saved references/folds; no new six-hour production session, model execution, refitting, or evaluation. Temporary control receipts are not empirical experiment results.')


if __name__=='__main__':
    with run_lock(),threadpool_limits(limits=1),metered('cpu','task1-authentic-resume-verification'):
        result=verify();save(task.OUT/'task1/resume_verification.json',result);print(json.dumps(result,indent=2))
