from __future__ import annotations


def test_config_imports_standalone():
    import backend.config  # noqa: F401


def test_full_import_chain_has_no_cycles():
    import backend.api.routes  # noqa: F401
    import backend.classification.classifier  # noqa: F401
    import backend.classification.features  # noqa: F401
    import backend.classification.model_classifier  # noqa: F401
    import backend.classification.rules  # noqa: F401


def test_contract_columns_matches_expected_order():
    from backend.config import CONTRACT_COLUMNS

    expected = [
        "latitude", "longitude", "acq_date", "acq_time", "satellite", "bright_ti4", "bright_ti5", "frp",
        "confidence", "daynight", "dist_to_industrial_m", "osm_industrial_tag",
        "dist_to_flare_capable_m", "nearest_flare_facility_type",
        "dist_to_heat_industry_m", "nearest_heat_facility_type",
        "dist_to_coal_mine_m", "nearest_coal_source",
        "dist_to_brick_kiln_m",
        "landcover_class", "recurrence_count", "first_seen", "last_seen",
        "is_anomalous", "label", "label_source",
    ]
    assert CONTRACT_COLUMNS == expected
