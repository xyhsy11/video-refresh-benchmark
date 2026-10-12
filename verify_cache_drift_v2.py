"""Focused v2 software fixtures; synthetic rows never enter research outputs."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pandas as pd

from analyze_cache_drift import PREDICTORS, TARGET, focused_checks, metrics_for_videos
from analyze_cache_drift_v2 import apply_fitted, fit_models, paired_comparisons, predict
from cache_drift_v2_utils import SEED, pilot_sources, validate_split
from collect_cache_drift import AGE_INPUTS, MOTION_INPUTS, interval_inputs
from prepare_cache_drift_v2 import download_shard
from research_utils import ROOT, now, write_json


def run_checks(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    checks = focused_checks()
    rng = np.random.default_rng(SEED)
    train = pd.DataFrame(rng.uniform(size=(30,10)), columns=AGE_INPUTS+MOTION_INPUTS)
    train[MOTION_INPUTS[2]] = 1
    train[TARGET] = rng.uniform(0,0.6,30)
    test = train.iloc[:6].copy()
    models = fit_models(train)
    changed = test.copy()
    changed[TARGET] = 1e12
    for name, state in models.items():
        np.testing.assert_array_equal(apply_fitted(test,state),apply_fitted(changed,state))
        assert not set(state["columns"]) & {TARGET,"video_id","source_video_id"}
    assert models["motion_only"]["scales"][2] == 1
    checks["v2_train_only_fit_and_held_target_isolation"] = True
    checks["v2_constant_columns_and_input_whitelist"] = True
    fixture = test.copy()
    fixture["video_id"] = [f"fixture{i//2}" for i in range(6)]
    fixture["source_video_id"] = fixture.video_id
    fixture["observation_id"] = [f"software_fixture:{i}" for i in range(6)]
    fixture["label"] = "software fixture"
    fixture["experiment_split"] = "test"
    fixture["candidate_ordinal"] = [1,2]*3
    fixture["cached_candidate_ordinal"] = 0
    predictions = predict(fixture,models)
    assert len(predictions)==24
    metrics = metrics_for_videos(predictions)
    pairs, intervals = paired_comparisons(metrics)
    primary = pairs.loc[(pairs.candidate=="age_plus_motion")&(pairs.reference=="age_only")].sort_values("video_id")
    indices=np.random.default_rng(SEED).integers(0,3,size=(10000,3))
    expected=np.percentile(primary.absolute_mae_improvement.to_numpy()[indices].mean(axis=1),[2.5,97.5],method="linear")
    np.testing.assert_allclose(expected,[intervals[0]["ci95_low"],intervals[0]["ci95_high"]],atol=1e-14)
    assert intervals==paired_comparisons(metrics)[1]
    checks["paired_whole_source_video_bootstrap_matches_independent_calculation"] = True
    checks["exactly_one_prediction_per_fixture_per_method"] = True
    near_ties = pd.DataFrame({"prediction": [0.1, np.nextafter(0.1, np.inf), np.nextafter(0.1, -np.inf), 0.2]})
    near_ties.to_csv(directory/"synthetic_rank_roundtrip.csv", index=False, float_format="%.17g")
    restored = pd.read_csv(directory/"synthetic_rank_roundtrip.csv", float_precision="round_trip")
    np.testing.assert_array_equal(near_ties.prediction.to_numpy(), restored.prediction.to_numpy())
    np.testing.assert_array_equal(near_ties.prediction.rank().to_numpy(), restored.prediction.rank().to_numpy())
    checks["csv_roundtrip_preserves_exact_predictions_and_rank_ties"] = True
    zero_metrics=metrics.copy()
    zero_metrics["mae"]=0.0
    zp,zi=paired_comparisons(zero_metrics)
    assert zp.percentage_mae_improvement.isna().all() and all(x["top_two_positive_gains_pct_of_net"] is None for x in zi)
    checks["zero_denominator_percentages_and_gain_concentration_undefined"] = True
    missing=[{"decoded_frame_index":i,"mv_available":False,"mean_magnitude_px":0.0,"p90_magnitude_px":0.0} for i in range(8)]
    observed_zero=[{**row,"mv_available":True} for row in missing]
    assert interval_inputs(missing,0,4)["interval_mv_available_fraction"]==0
    assert interval_inputs(observed_zero,0,4)["interval_mv_available_fraction"]==1
    checks["all_missing_motion_remains_distinguishable_from_observed_zero"] = True
    def entry(source,digest):
        return {"source_video_id":source,"file_identity":{"sha256":digest},"codec":"h264","duration_seconds":10,
                "decoded_frame_count":16,"candidate_indices":list(range(16)),"experiment_split":"train"}
    for case in [[entry("same", "one"),entry("same", "two")],
                 [entry("one","same"),entry("two","same")], [entry(sorted(pilot_sources())[0],"new")]]:
        try:
            validate_split({"videos":case},check_files=False,complete=False)
        except AssertionError:
            pass
        else:
            raise AssertionError("Overlap or pilot source was accepted")
    checks["split_source_hash_and_pilot_overlap_rejected"] = True

    # Real local HTTP Range behavior, on explicitly synthetic non-video bytes.
    payload=b"software-only resumable fixture"*5000
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200); self.send_header("Content-Length",str(len(payload))); self.send_header("ETag",'"fixture"'); self.end_headers()
        def do_GET(self):
            value=self.headers["Range"]
            requests.append(value)
            start,end=map(int,value.removeprefix("bytes=").split("-"))
            self.send_response(206); self.send_header("Content-Range",f"bytes {start}-{end}/{len(payload)}"); self.end_headers()
            self.wfile.write(payload[start:end+1])
        def log_message(self,*args):
            pass
    server=HTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    url=f"http://127.0.0.1:{server.server_address[1]}/fixture"
    destination=directory/"synthetic_resume_fixture.bin"
    chunks=destination.with_name(destination.name+".chunks")
    chunks.mkdir(exist_ok=True)
    write_json(chunks/"remote.json",{"url":url,"size_bytes":len(payload),"etag":'"fixture"'})
    (chunks/"0000.part").write_bytes(payload[:127])
    result=download_shard(url,destination,1)
    assert destination.read_bytes()==payload and requests==[f"bytes=127-{len(payload)-1}"]
    assert result["sha256"]==hashlib.sha256(payload).hexdigest()
    assert download_shard(url,destination,1)["reused"]
    server.shutdown(); server.server_close()
    checks["http_byte_range_resume_and_existing_archive_reuse"] = True
    result={"passed":True,"checked_at_utc":now(),"fixture_scope":"software-only synthetic rows/bytes; no experimental evidence", "checks":checks}
    write_json(directory/"software_checks.json",result)
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/cache_drift_v2_checks"))
    args=parser.parse_args()
    if (args.output/"software_checks.json").exists():
        from collect_cache_drift import fresh_directory
        args.output=fresh_directory(args.output)
    result=run_checks(args.output)
    print(f"Software verification PASS: {len(result['checks'])} checks; synthetic fixtures isolated in {args.output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
