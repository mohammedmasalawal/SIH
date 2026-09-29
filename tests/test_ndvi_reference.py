from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from training.ndvi_reference import (
    AMPLITUDE_CROP_MIN,
    DEKAD_DAYS,
    N_DEKADS,
    calibration_rows,
    classify_series,
    confusion_table,
    parse_dekads,
    score_agreement,
)

WINDOW_START = datetime(2026, 1, 1)


def _dekads(values: list[float | None]) -> list[dict]:
    """values[i] (or None to simulate a cloud-dropped/missing dekad) at grid slot i."""
    assert len(values) == N_DEKADS
    return [
        {"date": (WINDOW_START + timedelta(days=i * DEKAD_DAYS)).strftime("%Y-%m-%d"), "ndvi": v, "frac_valid": 0.9}
        for i, v in enumerate(values) if v is not None
    ]


def test_single_cropping_cycle_with_bare_soil_trough_is_crop_like():
    # bare (0.15) -> one strong green-up to 0.75 around dekad 16 -> back to bare
    values = [0.15] * 10 + [0.15, 0.30, 0.55, 0.72, 0.75, 0.72, 0.55, 0.30] + [0.15] * 18
    result = classify_series(_dekads(values), WINDOW_START)
    assert result["classification"] == "crop_like"
    assert result["n_peaks"] == 1
    assert result["amplitude"] >= 0.35
    assert result["trough"] <= 0.25


def test_double_cropping_two_peaks_is_crop_like_regardless_of_trough_depth():
    # two green-up cycles, troughs only moderate (not bare-soil) between/around them
    base = [0.35] * N_DEKADS
    for center in (6, 22):
        for offset, bump in zip(range(-3, 4), [0.05, 0.15, 0.30, 0.40, 0.30, 0.15, 0.05]):
            base[center + offset] = 0.35 + bump
    result = classify_series(_dekads(base), WINDOW_START)
    assert result["classification"] == "crop_like"
    assert result["n_peaks"] >= 2


def test_dense_forest_high_flat_ndvi_is_natural_like():
    rng = np.random.default_rng(0)
    values = list(0.72 + 0.03 * np.sin(np.linspace(0, 2 * np.pi, N_DEKADS)) + rng.normal(0, 0.005, N_DEKADS))
    result = classify_series(_dekads(values), WINDOW_START)
    assert result["classification"] == "natural_like"
    assert result["amplitude"] < AMPLITUDE_CROP_MIN


def test_single_monsoon_peak_with_shallow_trough_is_natural_like():
    """Amplitude alone clears the crop bar, but the trough never drops to bare soil
    (semi-arid shrubland/grassland stays dormant-green, not bare, in the dry season) --
    this is exactly the case the trough condition exists to separate from farmland."""
    base = [0.32] * 10 + [0.32, 0.45, 0.60, 0.70, 0.72, 0.70, 0.60, 0.45] + [0.32] * 18
    result = classify_series(_dekads(base), WINDOW_START)
    assert result["n_peaks"] == 1
    assert result["amplitude"] >= 0.35   # clears the amplitude bar
    assert result["trough"] > 0.25       # but never crashes to bare soil
    assert result["classification"] == "natural_like"


def test_too_few_valid_dekads_is_insufficient_data():
    values = [None] * N_DEKADS
    for i in (2, 10, 20):
        values[i] = 0.5
    result = classify_series(_dekads(values), WINDOW_START)
    assert result["classification"] == "insufficient_data"
    assert result["n_peaks"] is None


def test_interior_gap_is_interpolated_but_edges_are_never_extrapolated():
    values = [0.2] * 10 + [0.2, 0.5, None, 0.5, 0.2] + [0.2] * 21  # one interior gap mid-peak
    result = classify_series(_dekads(values), WINDOW_START)
    assert result["classification"] != "insufficient_data"  # still enough valid dekads
    assert result["n_peaks"] == 1  # the gap is interpolated, not treated as two separate peaks


def test_parse_dekads_drops_cloudy_intervals_and_sorts_by_date():
    payload = {"data": [
        {"interval": {"from": "2026-02-01T00:00:00Z"}, "outputs": {"ndvi": {"bands": {"B0": {"stats": {
            "sampleCount": 100, "noDataCount": 80, "mean": 0.4}}}}}},  # only 20% valid -> dropped
        {"interval": {"from": "2026-01-01T00:00:00Z"}, "outputs": {"ndvi": {"bands": {"B0": {"stats": {
            "sampleCount": 100, "noDataCount": 5, "mean": 0.3}}}}}},
        {"interval": {"from": "2026-01-11T00:00:00Z"}, "outputs": {"ndvi": {"bands": {"B0": {"stats": {
            "sampleCount": 100, "noDataCount": 0, "mean": 0.5}}}}}},
    ]}
    result = parse_dekads(payload)
    assert [r["date"] for r in result] == ["2026-01-01", "2026-01-11"]
    assert result[0]["ndvi"] == 0.3


def test_score_agreement_excludes_insufficient_data():
    reference = pd.Series(["crop_like", "natural_like", "insufficient_data", "crop_like"])
    rule = pd.Series(["crop_like", "crop_like", "crop_like", "crop_like"])  # 1 right, 1 wrong, 1 excluded, 1 right
    score = score_agreement(rule, reference)
    assert score["n_total"] == 4 and score["n_scored"] == 3
    assert score["agreement_pct"] == pytest.approx(200 / 3, abs=0.1)  # 2 of 3 scored rows agree


def test_score_agreement_all_insufficient_data_is_none():
    reference = pd.Series(["insufficient_data", "insufficient_data"])
    rule = pd.Series(["crop_like", "natural_like"])
    score = score_agreement(rule, reference)
    assert score["agreement_pct"] is None and score["n_scored"] == 0


def test_confusion_table_rows_are_hand_labels_columns_are_reference():
    expected = pd.Series(["agricultural burning", "wildfire", "wildfire"])
    actual = pd.Series(["crop_like", "natural_like", "crop_like"])
    table = confusion_table(expected, actual)
    assert table.loc["wildfire", "crop_like"] == 1
    assert table.loc["wildfire", "natural_like"] == 1
    assert table.loc["agricultural burning", "crop_like"] == 1


def test_calibration_rows_keeps_only_decided_agri_wildfire_and_maps_expected_class(tmp_path):
    gold = tmp_path / "gold.csv"
    pd.DataFrame([
        {"sample_id": "G001", "verified_label": "agricultural burning"},
        {"sample_id": "G002", "verified_label": "wildfire"},
        {"sample_id": "G003", "verified_label": "industrial"},   # excluded
        {"sample_id": "G004", "verified_label": "uncertain"},    # excluded
    ]).to_csv(gold, index=False)
    sample = tmp_path / "sample.csv"
    pd.DataFrame([
        {"sample_id": "G001", "latitude": 22.1, "longitude": 69.1, "acq_date": "2026-01-01"},
        {"sample_id": "G002", "latitude": 22.2, "longitude": 69.2, "acq_date": "2026-02-01"},
        {"sample_id": "G003", "latitude": 22.3, "longitude": 69.3, "acq_date": "2026-03-01"},
        {"sample_id": "G004", "latitude": 22.4, "longitude": 69.4, "acq_date": "2026-04-01"},
    ]).to_csv(sample, index=False)
    rows = calibration_rows(gold, sample)
    assert sorted(rows["sample_id"]) == ["G001", "G002"]
    assert dict(zip(rows["sample_id"], rows["expected_class"])) == {"G001": "crop_like", "G002": "natural_like"}


def test_calibration_rows_fails_loudly_on_unmatched_sample_id(tmp_path):
    gold = tmp_path / "gold.csv"
    pd.DataFrame([{"sample_id": "G999", "verified_label": "wildfire"}]).to_csv(gold, index=False)
    sample = tmp_path / "sample.csv"
    pd.DataFrame([{"sample_id": "G001", "latitude": 1.0, "longitude": 1.0, "acq_date": "2026-01-01"}]).to_csv(sample, index=False)
    with pytest.raises(SystemExit):
        calibration_rows(gold, sample)
