"""Bounded genuine reference-throughput diagnosis on the fixed fitting smoke.

Repeats real decode and all native forwards. Never uses cached vectors as work
or opens EVAL. Saved smoke arrays are used only for an exact post-run comparison.
"""
import time
import numpy as np
from native_head_value_confirmation import read,save,sha,rank,CHECKPOINT_SHA
from native_head_confirmation_runtime import NativeEncoder,decode,atomic_npz,feature_path
from native_head_confirmation_state import metered,snapshot_accounting
from run_native_head_value_confirmation import ROOT,OUT,CACHE,event,now,run_lock


def profile():
    if (OUT/'evaluation_exposure.json').exists() or (OUT/'evaluation_seal.json').exists():
        raise RuntimeError('Reference-throughput diagnosis is fitting-only and pre-seal')
    rows=read(OUT/'smoke_manifest.json')
    if len(rows)!=8 or read(OUT/'native_smoke_verification.json')['genuine_native_forwards']!=128:
        raise RuntimeError('Completed genuine native fitting smoke required')
    destination=CACHE/'reference_throughput'/rank('throughput-attempt',now())
    receipts=[]
    with metered('collection','genuine-normal-reference-model-setup'):encoder=NativeEncoder()
    assert encoder.validate_native is False
    for row in rows:
        key=rank('feature',row['filename']);target=destination/(key+'.npz')
        started=time.perf_counter();before=(encoder.starts,encoder.completions)
        with metered('collection','genuine-normal-reference-profile',source=row['component'],planned_encoder_calls=16):
            try:
                path=ROOT/row['path']
                if sha(path)!=row['encoded_sha256']:raise RuntimeError('Authenticated fitting RGB changed')
                pts,frames=decode(path.read_bytes(),pixels=True,needed=set(sum(row['grid']['frame_indices'],[])))
                if pts!=row['pts']:raise RuntimeError('Fitting PTS changed')
                features=[];logits=[];thumbnails=[]
                for indices in row['grid']['frame_indices']:
                    f,z=encoder.forward([frames[i][0] for i in indices],'PROBE_FIT_THROUGHPUT')
                    features.append(f);logits.append(z);thumbnails.append(frames[indices[-1]][1])
                atomic_npz(target,features=np.asarray(features),logits=np.asarray(logits),thumbnails=np.asarray(thumbnails))
                save(target.with_suffix('.json'),dict(source=row['filename'],encoded_sha256=row['encoded_sha256'],
                    checkpoint_sha256=CHECKPOINT_SHA,profile_implementation_sha256=sha(__file__),array_sha256=sha(target),
                    native_validation_inside_measurement=False,cached_vector_replay=False))
            finally:
                event('normal_reference_profile_attempt',source=row['component'],phase='PROBE_FIT_THROUGHPUT',
                    encoder_starts=encoder.starts-before[0],encoder_completions=encoder.completions-before[1])
        elapsed=time.perf_counter()-started
        with metered('cpu','normal-reference-exact-array-comparison'):
            with np.load(target,allow_pickle=False) as actual,np.load(feature_path(row),allow_pickle=False) as original:
                matches={name:bool(np.array_equal(actual[name],original[name])) for name in ('features','logits','thumbnails')}
            if not all(matches.values()):raise RuntimeError('Normal reference profile differs from authenticated smoke arrays')
        receipts.append(dict(filename=row['filename'],source=row['component'],elapsed_seconds=elapsed,
            array_path=target.relative_to(ROOT).as_posix(),array_sha256=sha(target),exact_array_matches=matches,
            native_forwards=encoder.starts-before[0],completed=encoder.completions-before[1]))
    result=dict(at_utc=now(),sources=8,genuine_native_forwards=encoder.starts,completed=encoder.completions,
        total_seconds=sum(r['elapsed_seconds'] for r in receipts),receipts=receipts,
        implementation_sha256=sha(__file__),cached_vector_replay_used_for_throughput=False,
        interpretation='Actual ordinary reference extraction without smoke-only affine assertions; same frozen RGB/grid/native model/threads and exact saved-array parity. This is additional fitting-only validation, not EVAL or a scientific effect.')
    save(OUT/'normal_reference_throughput_diagnosis.json',result)
    snapshot_accounting()
    print({k:v for k,v in result.items() if k!='receipts'})
    return result


if __name__=='__main__':
    with run_lock():profile()
