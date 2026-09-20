# Sample fixtures

`firms_sample.csv`, `industrial_sample.geojson`, and `facilities_sample.geojson` are
**synthetic demo/smoke-test inputs**, not real FIRMS/OSM/GEM data and never a real
training set. They exist so `training/build_labels.py` (and the tests in `tests/`)
are runnable without live API keys or a live FIRMS pull.

Shape: a 3-day recurring cluster (`22.30, 69.85`) 163m from a `cement_plant`
facility — resolves to `industrial` under `backend/classification/rules.py` — plus
one far one-off detection that resolves to `unknown`.

Run it:

```powershell
python -m training.build_labels --firms data/sample/firms_sample.csv --industrial data/sample/industrial_sample.geojson --facilities data/sample/facilities_sample.geojson --output training/data/labeled_hotspots_demo.csv
```

This fixture is deliberately too small and too single-class to usefully feed
`training/train_model.py` (only one label survives excluding `unknown`, and
XGBoost needs at least two). It demonstrates the ingestion → labeling step only;
`train_model.py`/`evaluate.py` need a real, larger `labeled_hotspots.csv` from an
actual FIRMS pull (see the root `README.md` and `training/README.md`).

`data/external/gem_facilities_india.csv` is separate — real GEM master data, not a
demo fixture.
