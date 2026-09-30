"""Guard for the large read-only inputs that live ingestion needs: the OSM extract, the industrial-
context parquet built from it, and the ESA WorldCover tiles. None of them is in git.

    python -m training.data_manifest check               # files exist and sizes match data/MANIFEST.json (fast)
    python -m training.data_manifest check --checksums   # ...and SHA-256 too (minutes)
    python -m training.data_manifest build               # (re)write data/MANIFEST.json from what is on disk

training/run_ingest.cmd runs `check` first. On a mismatch it prints what is wrong, pings the
heartbeat's failure URL if one is set (HEARTBEAT_URL in the environment or .env; a healthchecks-
style `<url>/fail` request) and exits non-zero, so no scheduled run ever starts on a half-restored
data folder.

The manifest also records where each file came from, so a lost file can be fetched again:
see README "Data folder safety".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from backend.config import DATA_DIR, NATIONAL_INDUSTRIAL_CONTEXT_PATH, NATIONAL_WORLDCOVER_DIR, ROOT_DIR
from backend.ingestion.firms_client import _env

MANIFEST_PATH = ROOT_DIR / "data" / "MANIFEST.json"
OSM_PBF = DATA_DIR / "osm" / "india-latest.osm.pbf"
WORLDCOVER_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}"


NOTES = {
    "osm/india-latest.osm.pbf": "Geofabrik india-260923.osm.pbf (md5 47de076bd6e2d3aca761551d19bdf517, verified), "
                                "https://download.geofabrik.de/asia/india-260923.osm.pbf. Replaced the 2026-09-20 extract "
                                "(1,707,348,267 bytes), lost on 2026-09-30 and no longer served by Geofabrik.",
    "osm/india_industrial_context.parquet": "Rebuilt from that extract with extract_osm_industrial_context, unchanged: "
                                            "37,930 features (the lost file: 37,908). Solar/wind plants are excluded when it is "
                                            "loaded, not in the file.",
    "raw/worldcover_india/*.tif": "102 ESA WorldCover v200 2021 tiles, re-downloaded from the public bucket on 2026-09-30; "
                                  "sizes match the bucket (and the 12 pre-deletion sizes on record); landcover recomputed for a "
                                  "random 20,000 stored detections matches exactly.",
}


def tracked_files() -> list[Path]:
    """The files the manifest covers: the OSM extract, its industrial-context parquet, every WorldCover tile."""
    tiles = sorted(Path(NATIONAL_WORLDCOVER_DIR).glob("*.tif"))
    return [OSM_PBF, Path(NATIONAL_INDUSTRIAL_CONTEXT_PATH), *tiles]


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def build(notes: dict | None = None, manifest_path: Path | None = None) -> dict:
    manifest_path = manifest_path or MANIFEST_PATH
    files = {}
    for path in tracked_files():
        if not path.exists():
            raise FileNotFoundError(f"cannot build a manifest: {path} is missing")
        entry = {"size": path.stat().st_size, "sha256": sha256(path)}
        if path.suffix == ".tif":
            entry["source"] = WORLDCOVER_URL.format(name=path.name)
        files[_relative(path)] = entry
    manifest = {"about": "Large inputs not in git; checked before every live ingest (training/data_manifest.py).",
                "relative_to": "AGNINETRA_DATA_DIR (default: <repo>/data)", "notes": NOTES if notes is None else notes, "files": files}
    Path(manifest_path).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def check(manifest_path: Path | None = None, *, checksums: bool = False) -> list[str]:
    """Problems found (empty = all inputs present and matching)."""
    manifest_path = manifest_path or MANIFEST_PATH
    if not Path(manifest_path).exists():
        return [f"{manifest_path} is missing"]
    manifest = json.loads(Path(manifest_path).read_text())
    problems = []
    for rel, entry in manifest["files"].items():
        path = DATA_DIR / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif path.stat().st_size != entry["size"]:
            problems.append(f"size differs: {rel} is {path.stat().st_size:,} bytes, manifest says {entry['size']:,}")
        elif checksums and sha256(path) != entry["sha256"]:
            problems.append(f"checksum differs: {rel}")
    on_disk = {_relative(p) for p in tracked_files() if p.exists()}
    extra = sorted(on_disk - set(manifest["files"]))
    problems += [f"not in the manifest: {rel}" for rel in extra]
    return problems


def ping_failure(message: str) -> bool:
    """Ping the heartbeat failure URL if HEARTBEAT_URL is set; True if a ping was sent."""
    base = _env("HEARTBEAT_URL").rstrip("/")
    if not base:
        return False
    try:
        request = urllib.request.Request(base + "/fail", data=message.encode("utf-8", "replace")[:10000], method="POST")
        urllib.request.urlopen(request, timeout=15).read()
        return True
    except Exception as error:  # a dead heartbeat must not hide the real problem
        print(f"(heartbeat ping failed: {error})", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "build"])
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        manifest = build()
        print(f"wrote {MANIFEST_PATH}: {len(manifest['files'])} files")
        return 0
    problems = check(checksums=args.checksums)
    if not problems:
        print(f"data inputs OK ({len(json.loads(MANIFEST_PATH.read_text())['files'])} files match data/MANIFEST.json)")
        return 0
    shown = "\n  ".join(problems[:15]) + (f"\n  ... and {len(problems) - 15} more" if len(problems) > 15 else "")
    message = (f"INGEST NOT STARTED: {len(problems)} input problem(s) under {DATA_DIR}:\n  {shown}\n"
               "Restore them (README 'Data folder safety') and re-run.")
    print(message, file=sys.stderr)
    pinged = ping_failure(message)
    print("heartbeat failure ping sent" if pinged else "no HEARTBEAT_URL set: no heartbeat ping sent", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
