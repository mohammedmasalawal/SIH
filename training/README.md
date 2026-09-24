# Training pipeline (offline, run by hand)

Run in order from the repo root:

```powershell
python -m training.build_labels --firms data/raw/your_pull.csv --industrial data/context/osm_industrial.geojson --facilities data/context/gem_facilities.geojson --output training/data/labeled_hotspots.csv
python -m training.train_model --labeled-csv training/data/labeled_hotspots.csv
python -m training.evaluate --labeled-csv training/data/labeled_hotspots.csv
```

1. **`build_labels.py`** — joins a FIRMS pull to OSM/GEM context and applies the weak
   rules in `backend/classification/rules.py`, producing `labeled_hotspots.csv`
   (the `CONTRACT_COLUMNS` shape defined in `backend/config.py`).
2. **`train_model.py`** — excludes `label == "unknown"` rows, splits by whole
   ~375m grid cell (never by row — see `spatial_block_split`), and fits an
   XGBoost classifier on `backend/classification/features.py`'s feature vectors,
   writing `training/artifacts/model_400k.pkl`.
3. **`evaluate.py`** — reproduces the same spatial-block split against the same
   CSV + seed recorded in the model's own metadata and scores it, writing
   `training/artifacts/eval_summary.json`.

## Why the labels are "weak" / "silver," not ground truth

Every label in `labeled_hotspots.csv` comes from `rules.py` — simple, deliberately
conservative distance/recurrence/landcover checks against a hotspot's context, not
from human review of imagery or incident reports. A hotspot the rules can't
confidently place lands as `unknown` rather than being forced into a guess, and
`unknown` rows are excluded from training entirely (`is_industrial`,
`is_gas_flare`, etc. are precision-oriented, not recall-oriented — they'd rather
abstain than mislabel).

Consequences for anything built on top of this:
- Never train or evaluate on a random row split. `recurrence_count`, `first_seen`,
  `last_seen`, and `is_anomalous` are computed per physical site (grid cell) across
  the *entire* input pull, so a random split would put the same site's rows on
  both sides of train/test and leak the answer. `spatial_block_split` splits by
  whole grid cell specifically to prevent this.
- A trained model's accuracy is only as good as the rules that generated its
  labels — it will reproduce whatever bias or blind spot the rules have (e.g. the
  known Reliance-refinery large-facility mismatch documented in the root
  `README.md`), not correct it.
- `daynight` is intentionally never a rule input or a model feature signal beyond
  what's implicit in `is_anomalous`'s day/night-separated baseline — it exists as
  an independent sanity check on whether the rules (and by extension the labels
  the model trains on) are behaving physically sensibly.
