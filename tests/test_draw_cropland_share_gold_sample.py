from __future__ import annotations

import json

import pandas as pd
import pytest

from training.draw_cropland_share_gold_sample import (
    _excluded_cells,
    build_sample_file,
    draw_sample,
    load_qualifying_with_timing,
    with_dnbr_evidence,
)
from training.spatial_features import grid_keys


@pytest.fixture
def meta_and_impact(tmp_path):
    """Full-detail source rows (as the national parquets carry) plus a matching
    impact-report CSV (as report_cropland_share_impact.py writes), covering
    qualifying and non-qualifying rows in more than one state."""
    rows = pd.DataFrame([
        # Gujarat, qualifies
        {"latitude": 23.2, "longitude": 72.6, "acq_date": "2026-03-01", "acq_time": "0130", "satellite": "N",
         "label": "wildfire", "landcover_class": 20},
        # Karnataka, qualifies
        {"latitude": 15.3, "longitude": 75.7, "acq_date": "2026-03-02", "acq_time": "0230", "satellite": "N",
         "label": "wildfire", "landcover_class": 30},
        # Andhra Pradesh, qualifies
        {"latitude": 15.9, "longitude": 78.5, "acq_date": "2026-03-03", "acq_time": "0330", "satellite": "1",
         "label": "wildfire", "landcover_class": 20},
        # Rajasthan, qualifies
        {"latitude": 26.9, "longitude": 75.8, "acq_date": "2026-03-04", "acq_time": "0430", "satellite": "N",
         "label": "wildfire", "landcover_class": 30},
        # Gujarat, does NOT qualify (low cropland share)
        {"latitude": 23.3, "longitude": 72.7, "acq_date": "2026-03-05", "acq_time": "0530", "satellite": "N",
         "label": "wildfire", "landcover_class": 20},
        # tree cover -- never a candidate regardless of share
        {"latitude": 23.4, "longitude": 72.8, "acq_date": "2026-03-06", "acq_time": "0630", "satellite": "N",
         "label": "wildfire", "landcover_class": 10},
        # not wildfire -- excluded regardless of share
        {"latitude": 23.5, "longitude": 72.9, "acq_date": "2026-03-07", "acq_time": "0730", "satellite": "N",
         "label": "industrial", "landcover_class": 20},
    ])
    source = tmp_path / "source.parquet"
    rows.to_parquet(source, index=False)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"sources": [{"path": str(source)}]}))

    # report_cropland_share_impact.py's own load_wildfire_shrub_grass already drops tree
    # cover (10) and non-wildfire rows before computing shares -- match that here.
    shrub_grass_wildfire = rows[(rows["label"] == "wildfire") & (rows["landcover_class"].isin([20, 30]))]
    impact = shrub_grass_wildfire[["latitude", "longitude", "landcover_class"]].copy()
    impact["cropland_share_1km"] = [0.9, 0.8, 0.7, 0.6, 0.2]
    impact["would_reclassify"] = impact["cropland_share_1km"] > 0.5
    impact_path = tmp_path / "impact.csv"
    impact.to_csv(impact_path, index=False)
    return meta, impact_path


def test_load_qualifying_with_timing_attaches_full_detail_to_qualifying_rows(meta_and_impact):
    meta, impact_path = meta_and_impact
    candidates = load_qualifying_with_timing(impact_path, meta_path=meta)
    assert len(candidates) == 4  # the 4 qualifying wildfire, non-tree-cover rows
    assert set(candidates["acq_date"]) == {"2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"}
    assert "acq_time" in candidates.columns and "cropland_share_1km" in candidates.columns


def test_excluded_cells_reads_existing_gold_files(tmp_path):
    gold_a = tmp_path / "a.csv"
    pd.DataFrame({"latitude": [23.2], "longitude": [72.6]}).to_csv(gold_a, index=False)
    gold_b = tmp_path / "b.csv"  # missing file -- should be skipped, not error
    cells = _excluded_cells([str(gold_a), str(gold_b)])
    glat, glon = grid_keys(pd.DataFrame({"latitude": [23.2], "longitude": [72.6]}))
    assert (int(glat.iloc[0]), int(glon.iloc[0])) in cells
    assert len(cells) == 1


def test_draw_sample_excludes_cells_and_spreads_across_priority_states(meta_and_impact):
    meta, impact_path = meta_and_impact
    candidates = load_qualifying_with_timing(impact_path, meta_path=meta)
    picked = draw_sample(candidates, n=4, seed=1, priority_states=["Gujarat", "Karnataka", "Andhra Pradesh", "Rajasthan"])
    assert len(picked) == 4
    assert set(picked["_state"]) == {"Gujarat", "Karnataka", "Andhra Pradesh", "Rajasthan"}  # one from each


def test_draw_sample_respects_excluded_cells(meta_and_impact):
    meta, impact_path = meta_and_impact
    candidates = load_qualifying_with_timing(impact_path, meta_path=meta)
    glat, glon = grid_keys(pd.DataFrame({"latitude": [23.2], "longitude": [72.6]}))
    excluded = {(int(glat.iloc[0]), int(glon.iloc[0]))}  # the Gujarat qualifying row's cell
    picked = draw_sample(candidates, n=4, seed=1, priority_states=["Gujarat", "Karnataka", "Andhra Pradesh", "Rajasthan"],
                         exclude_cells=excluded)
    assert "Gujarat" not in set(picked["_state"])
    assert len(picked) == 3  # only 3 states had a qualifying, non-excluded candidate


def test_draw_sample_stops_at_n_even_with_more_candidates(meta_and_impact):
    meta, impact_path = meta_and_impact
    candidates = load_qualifying_with_timing(impact_path, meta_path=meta)
    picked = draw_sample(candidates, n=2, seed=1, priority_states=["Gujarat", "Karnataka", "Andhra Pradesh", "Rajasthan"])
    assert len(picked) == 2


def test_build_sample_file_has_no_label_or_cropland_share_and_correct_ist(meta_and_impact):
    meta, impact_path = meta_and_impact
    candidates = load_qualifying_with_timing(impact_path, meta_path=meta)
    picked = draw_sample(candidates, n=4, seed=1, priority_states=["Gujarat", "Karnataka", "Andhra Pradesh", "Rajasthan"])
    sample = build_sample_file(picked)

    assert list(sample.columns) == [
        "sample_id", "latitude", "longitude", "acq_date", "acq_time", "acq_time_ist", "maps_link",
        "state", "verified_label", "structure_seen", "confidence",
    ]
    assert "cropland_share_1km" not in sample.columns and "landcover_class" not in sample.columns
    assert (sample["verified_label"] == "").all()
    assert sample["sample_id"].tolist() == [f"CS{i:03d}" for i in range(1, len(sample) + 1)]
    row = sample[sample["acq_time"] == "0130"].iloc[0]
    assert row["acq_time_ist"] == "07:00"  # 01:30 UTC + 5:30 = 07:00 IST, same day
    assert row["maps_link"] == f"https://www.google.com/maps?q={row['latitude']},{row['longitude']}&t=k"


def test_build_sample_file_ist_rolls_into_next_day():
    picked = pd.DataFrame([{
        "latitude": 22.0, "longitude": 70.0, "acq_date": "2026-03-01", "acq_time": "2200", "satellite": "N",
        "_state": "Gujarat",
    }])
    sample = build_sample_file(picked)
    assert sample.iloc[0]["acq_time_ist"] == "03:30"  # 22:00 UTC + 5:30 -> 03:30 the next day


def test_with_dnbr_evidence_calls_check_point_per_row_and_merges_results(monkeypatch):
    import training.draw_cropland_share_gold_sample as mod

    monkeypatch.setattr(mod.dnbr_check, "get_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr(mod.dnbr_check, "get_token", lambda session, cid, secret: "tok")
    calls = []

    def fake_check_point(session, token, lat, lon, acq_date):
        calls.append((lat, lon, acq_date))
        return {"status": "ok", "dnbr": 0.32, "burn_evidence": "low severity", "nbr_before": 0.4, "nbr_after": 0.08}

    monkeypatch.setattr(mod.dnbr_check, "check_point", fake_check_point)
    sample = pd.DataFrame([
        {"sample_id": "CS001", "latitude": 23.2, "longitude": 72.6, "acq_date": "2026-03-01"},
        {"sample_id": "CS002", "latitude": 15.3, "longitude": 75.7, "acq_date": "2026-03-02"},
    ])
    out = with_dnbr_evidence(sample, pause=0)
    assert len(calls) == 2 and calls[0] == (23.2, 72.6, "2026-03-01")
    assert list(out["dnbr"]) == [0.32, 0.32]
    assert list(out["burn_evidence"]) == ["low severity", "low severity"]
