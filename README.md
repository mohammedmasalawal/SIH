# AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources

An AI-enabled geospatial monitoring system for identifying, classifying, and tracking industrial fires and persistent thermal sources using NASA FIRMS, OpenStreetMap (OSM), and satellite imagery.

## Challenge Context

Industrial facilities such as oil refineries, petrochemical complexes, thermal power plants, steel industries, mining areas, and LNG terminals produce thermal signatures that can be detected from space. Accidental fires, gas leaks, explosions, and abnormal heat events can also threaten critical infrastructure, public safety, and the environment.

NASA FIRMS provides valuable near-real-time thermal anomaly detections, but a thermal hotspot alone does not explain its cause. The same type of detection may represent an industrial fire, gas flare, agricultural burning, mining activity, or wildfire. This project addresses that gap by combining thermal observations with geographic context, industrial infrastructure data, land-cover information, and satellite imagery.

## Organization

**National Technical Research Organisation (NTRO)**

- [Organization details](https://sih-2026-explorer-pearl.vercel.app/organizations/national-technical-research-organisation-ntro/)
- [Official SIH portal](https://sih.gov.in/sih2026PS)
- **Submission deadline:** 20 September 2026

## Objectives

- Detect and monitor thermal anomalies over a selected area of interest.
- Classify anomalies as industrial fires, persistent industrial heat sources, gas flares, agricultural burns, mining activity, wildfires, or other events.
- Separate industrial fires from forest fires and other natural fires.
- Identify persistent thermal sources using repeated observations over time.
- Correlate hotspots with nearby facilities, roads, settlements, water bodies, and land-cover classes.
- Present results as an actionable GIS layer over an interactive map.

## Proposed Solution

The system will combine four kinds of evidence:

1. **Thermal anomaly data** from NASA FIRMS, including hotspot location, acquisition time, confidence, and brightness-temperature attributes where available.
2. **Infrastructure and geographic context** from OSM and other authoritative datasets, including industrial plants, refineries, power stations, mines, airports, roads, and settlements.
3. **Land-cover and environmental context** to distinguish industrial areas from forests, croplands, grasslands, and urban regions.
4. **Satellite imagery and temporal history** for visual confirmation, feature extraction, change detection, and persistence analysis.

### High-Level Pipeline

```text
NASA FIRMS + OSM + Land Cover + Satellite Imagery
			 |
		 Data Harmonization
			 |
	  Spatial Join + Temporal Aggregation
			 |
	     Feature Engineering / Embeddings
			 |
	      AI Event Classification Model
			 |
       Persistence, Severity, and Confidence Scoring
			 |
	     GIS API and Interactive Map Overlay
```

## Core Features

### Hotspot ingestion and normalization

- Import FIRMS detections for the selected region and time window.
- Normalize coordinate reference systems, timestamps, confidence values, and source metadata.
- Remove duplicates and flag incomplete or low-confidence observations.

### Context-aware classification

For each hotspot, construct a local context using:

- Distance to mapped industrial facilities and infrastructure.
- Land-cover and vegetation class.
- Hotspot density and recurrence over time.
- Brightness temperature, confidence, and observation frequency.
- Satellite-image features and visible change around the detection.
- Nearby settlements, roads, water bodies, and protected areas.

The classifier should return both a predicted category and a confidence score so that uncertain cases can be reviewed by an analyst.

### Persistent-source detection

Repeated detections at the same location can indicate a flare, furnace, kiln, power-generation source, mining operation, or another persistent thermal source. A temporal aggregation layer will group observations spatially and calculate recurrence, duration, intensity, and recent activity.

### GIS visualization and analyst workflow

The map interface should support:

- Thermal hotspot overlays with category-based styling.
- Facility and infrastructure layers from OSM.
- Time-range filtering and playback of historical detections.
- Popups containing source data, classification, confidence, persistence, and last-seen time.
- Filtering by event type, confidence, severity, and facility proximity.
- Export of filtered events and map-ready geospatial data.

## Expected Deliverables

1. A working model that classifies industrial fires separately from forest fires and other natural fires.
2. A persistent thermal-source detection and monitoring workflow.
3. A GIS-based data store for normalized detections, features, predictions, and event history.
4. An interactive map that displays model output as overlays on geographic and satellite layers.
5. An API or service layer for ingesting data and serving event results to the frontend.
6. Evaluation results covering classification quality, false positives, detection latency, and spatial accuracy.
7. Documentation covering data sources, model limitations, reproducibility, and responsible use.

## Suggested System Architecture

| Layer | Responsibility |
| --- | --- |
| Data ingestion | Fetch FIRMS observations and geographic datasets on a schedule or on demand. |
| Geospatial processing | Reproject, validate, spatially join, and index points and polygons. |
| Feature store | Keep derived environmental, infrastructure, temporal, and imagery features. |
| ML inference | Classify events and produce confidence and severity scores. |
| Persistence engine | Group repeat detections and identify recurring thermal sources. |
| GIS backend | Store and serve events as GeoJSON or vector tiles through a queryable API. |
| Web application | Visualize overlays, timelines, filters, imagery, and analyst details. |

## Technology Direction

The implementation can be assembled from open geospatial and machine-learning tooling, for example:

- **Data and APIs:** NASA FIRMS, OSM/Overpass, satellite imagery providers, GeoJSON
- **Geospatial processing:** Python, GeoPandas, Rasterio, GDAL, Shapely
- **Machine learning:** scikit-learn and/or a deep-learning framework suited to the available labeled imagery
- **Spatial storage:** PostgreSQL with PostGIS
- **Backend:** FastAPI or an equivalent geospatial API service
- **Visualization:** MapLibre GL JS, Leaflet, or another map library with raster and vector overlay support

Technology choices should remain modular so that restricted or rate-limited data sources can be replaced without changing the classification workflow.

## Evaluation Plan

Performance should be measured against a labeled validation set and, where possible, independent incident records:

- Precision, recall, F1-score, and confusion matrix for event classes.
- False-positive rate near industrial facilities and in natural-fire regions.
- Spatial distance between predicted events and confirmed source locations.
- Detection latency from observation availability to classification.
- Persistence accuracy for recurring thermal sources.
- Map and API response time for operational use.

## Responsible Use and Limitations

This system is intended for situational awareness and decision support. Satellite revisit time, cloud cover, spatial resolution, incomplete facility inventories, geolocation uncertainty, and mislabeled training data can affect results. Predictions should be treated as confidence-scored indicators and verified against current imagery, ground reports, or authorized operational sources before emergency action.

## Project Status

This repository currently contains the project brief. Implementation details, datasets, model experiments, backend services, and the web GIS application will be added as development progresses.

## References

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [Download the full SIH 2026 problem statement](https://sih-2026-explorer-pearl.vercel.app/downloads/01_SIH_2026_MASTER_PROBLEM_STATEMENTS.pdf)
