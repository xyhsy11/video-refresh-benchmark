"""Guarded recovery: never repeat the single predeclared HMDB head attempt."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/continuation_value_gap_v1'


def main():
    ledger=OUT/'exposure_resource_ledger.jsonl'
    events=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    starts=[e for e in events if e['kind']=='HMDB_head_training_start']
    fit=OUT/'hmdb_head_fit.json';head=OUT/'hmdb_head.json'
    if starts:
        accepted=fit.exists() and head.exists() and json.loads(fit.read_text(encoding='utf-8')).get('accepted') is True
        if not accepted:
            raise SystemExit('VERIFICATION_FAILED: the one predeclared head attempt started but has no accepted retained head. Preserve evidence; no automatic optimization retry is authorized.')
    from continuation_value_gap_hmdb import run
    run()


if __name__=='__main__':main()
