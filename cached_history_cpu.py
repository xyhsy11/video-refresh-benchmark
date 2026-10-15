"""Preloaded cached-observation CPU components, distinct from video latency."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

import cached_history_signal_utils as u
from research_utils import now

MODES = ["history_T", "baseline_scale_RF1", "augmented_history_scale_RF1"]


@dataclass(frozen=True, slots=True)
class Observation:
    last_refresh: int
    cached_scores: np.ndarray
    baseline_row: np.ndarray


def preload(observations: pd.DataFrame, sources: list[str]) -> list[list[Observation]]:
    videos = []
    for source in sources:
        group = observations[observations.source_video_id == source].sort_values("t")
        assert len(group) == 9
        videos.append([Observation(int(row.last_refresh), row[u.CACHED_COLUMNS].to_numpy(dtype=np.float64),
                                   row[u.C_COLUMNS].to_numpy(dtype=np.float64)) for _, row in group.iterrows()])
    return videos


def prepare_predictors(folder: Path, registry: dict) -> dict:
    result = {}
    for method in ["C_RF400", "C_plus_T_RF400"]:
        state, _, forest = u.previous.load_model(folder, registry["models"][method])
        result[method] = (forest, np.asarray(state["means"]), np.asarray(state["scales"]))
    return result


def component_pass(videos: list[list[Observation]], predictors: dict, mode: str):
    """Timed path: no assertions/logging/I/O/fitting; one video state at a time."""
    last = None
    if mode == "baseline_scale_RF1":
        forest, mean, scale = predictors["C_RF400"]
        for video in videos:
            for obs in video:
                last = forest.predict(((obs.baseline_row-mean)/scale).reshape(1, -1))
    else:
        if mode == "augmented_history_scale_RF1":
            forest, mean, scale = predictors["C_plus_T_RF400"]
        for video in videos:
            history = u.CachedHistory()
            for obs in video:
                temporal = history.advance(obs.last_refresh, obs.cached_scores)
                if mode == "history_T":
                    last = temporal
                else:
                    augmented = np.concatenate((obs.baseline_row, temporal))
                    last = forest.predict(((augmented-mean)/scale).reshape(1, -1))
    return last


def run(observations: pd.DataFrame, sources: list[str], folder: Path, registry: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    # All input access/model loading and correctness checks precede warmups/timing.
    videos, predictors = preload(observations, sources), prepare_predictors(folder, registry)
    for video, source in zip(videos, sources):
        h = u.CachedHistory()
        expected = observations[observations.source_video_id == source].sort_values("t")
        for obs, (_, row) in zip(video, expected.iterrows()):
            t = h.advance(obs.last_refresh, obs.cached_scores)
            np.testing.assert_allclose(t, row[u.T_COLUMNS].to_numpy(dtype=float), atol=u.TOL, rtol=0)
    rng = np.random.Generator(np.random.PCG64(seed))
    warmup_orders, measured_orders, rows = [], [], []
    for _ in range(2):
        order = rng.permutation(MODES).tolist()
        warmup_orders.append(order)
        for mode in order:
            component_pass(videos, predictors, mode)
    for run_id in range(1, 6):
        order = rng.permutation(MODES).tolist()
        measured_orders.append({"run_id": run_id, "order": order})
        for ordinal, mode in enumerate(order):
            start = time.perf_counter()
            last = component_pass(videos, predictors, mode)
            elapsed = time.perf_counter()-start
            assert last is not None  # deliberately after stopping the timer
            rows.append({"mode": mode, "run_id": run_id, "execution_ordinal": ordinal+1,
                "videos": 10, "opportunities": 90, "distinct_history_updates": 0 if mode == MODES[1] else 30,
                "rf_batch_size_one_calls": 0 if mode == MODES[0] else 90,
                "total_ms": elapsed*1000, "us_per_opportunity": elapsed*1e6/90})
    metadata = {"recorded_at_utc": now(), "sample_source_ids": sources, "warmup_passes": 2,
        "measured_passes": 5, "seed": seed, "rng": "PCG64", "warmup_execution_order": warmup_orders,
        "measured_execution_order": measured_orders, "execution": "one process,sequential;RF n_jobs1;batch size1",
        "boundary": "perf_counter around one component pass over10 preloaded videos/90 chronological opportunities",
        "includes": {MODES[0]: "per-video state allocation/reset,all30 distinct history updates,90 T accesses;T recomputed only for new cached events",
                     MODES[1]: "56-column scaling and90 individual RF400 predict calls,including sklearn prediction validation",
                     MODES[2]: "same history updates/reset,T construction,100-column concatenation/scaling,90 individual RF400 predict calls"},
        "excludes": ["file/model loading", "fitting", "logging", "correctness assertions", "video decoding", "RGB extraction",
                     "codec-vector extraction", "full recognition execution"],
        "model_choice": "final development models;sampled training sources used only for CPU component work,not heldout quality claims",
        "history_softmax_evaluations_per_pass": 30, "history_feature_assemblies_per_pass": 30,
        "timing_note": "component CPU timings only;no subtraction from old pilot or end-to-end adaptive speedup claim",
        "history_state": u.CachedHistory().memory_accounting()}
    return pd.DataFrame(rows), metadata
