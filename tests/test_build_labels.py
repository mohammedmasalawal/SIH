from __future__ import annotations

from pathlib import Path

from training.build_labels import build_labels

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def test_build_labels_end_to_end_against_sample_fixtures(tmp_path):
    output_csv = tmp_path / "labeled.csv"
    result = build_labels(
        SAMPLE_DIR / "firms_sample.csv",
        SAMPLE_DIR / "industrial_sample.geojson",
        SAMPLE_DIR / "facilities_sample.geojson",
        output_csv,
    )

    clustered = result[(result["latitude"] == 22.30) & (result["longitude"] == 69.85)]
    assert len(clustered) == 3
    assert (clustered["label"] == "industrial").all()

    far_one_off = result[(result["latitude"] == 21.95) & (result["longitude"] == 70.10)]
    assert len(far_one_off) == 1
    assert (far_one_off["label"] == "unknown").all()

    assert output_csv.exists()
