"""Batch historical blob audit plus independent sealed-artifact arithmetic.

Same historical content check as the standalone verifier, in one Git process.
No endpoint implementation is imported. Useful for the complete HMDB audit.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import verify_continuation_value_gap as v


def main():
    start=time.perf_counter();out=v.OUT
    amended=v.read(out/'amendments/mechanical_001.json')['changed_files']
    for n,h in v.read(out/'design_seal.json')['sha256'].items():
        if n in amended:
            a=amended[n]
            assert a['original_sha256']==h and v.sha(ROOT/a['preserved_original'])==h
            assert v.sha(ROOT/n)==a['amended_sha256']
        else:assert v.sha(ROOT/n)==h
    for n,h in v.read(out/'input_manifest.json').items():assert v.sha(ROOT/n)==h
    historical=v.read(out/'historical_git_blobs.json')['lines']
    current=subprocess.check_output(['git','ls-files','-s','results'],cwd=ROOT,text=True).splitlines()
    assert set(historical).issubset(current)
    pairs=[line.split('\t',1) for line in historical];names=[name for left,name in pairs]
    process=subprocess.run(['git','hash-object','--stdin-paths'],cwd=ROOT,input='\n'.join(names)+'\n',capture_output=True,text=True,check=True)
    hashes=process.stdout.splitlines();assert len(hashes)==len(pairs)
    for (left,name),h in zip(pairs,hashes):assert h==left.split()[1], 'Changed historical content: '+name
    result=dict(passed=True,historical_result_files_unchanged=len(pairs),historical_hashing='git hash-object --stdin-paths; same per-path filters')
    for name in ['ucf','hmdb']:
        if (out/name/'estimates.json').exists():result[name]=v.empirical(name)
    result['seconds']=time.perf_counter()-start
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
