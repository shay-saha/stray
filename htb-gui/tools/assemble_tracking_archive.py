#!/usr/bin/env python3
"""Bundle the atlas's published animal observations into one chronological CSV."""

import argparse
import csv
import gzip
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from prepare_migration_tracks import TRACK_HEADER, unix_seconds


CURATED_SPECIES = {
    "Waved Albatross": "Waved albatross",
    "Osprey": "Osprey",
    "Black-tailed Godwit": "Black-tailed godwit",
    "African Cuckoo": "African cuckoo",
    "White-fronted Goose": "Greater white-fronted goose",
    "Caribou": "Caribou",
    "Green Sea Turtle": "Green sea turtle",
    "Blue Whale": "Blue whale",
    "California Sea Lion": "California sea lion",
    "Straw-coloured Fruit Bat": "Straw-coloured fruit bat",
}

# Filename stem, common name, expected taxon, and minimum interval in minutes.
RAW_SOURCES = (
    ("gull-2018", "Lesser black-backed gull", "Larus fuscus", 60),
    ("harrier-antwerp", "Western marsh harrier", "Circus aeruginosus", 30),
    ("harrier-waterland", "Western marsh harrier", "Circus aeruginosus", 30),
    ("crane-sweden", "Common crane", "Grus grus", 60),
    ("osprey", "Osprey", "Pandion haliaetus", 120),
    ("goose", "Greater white-fronted goose", "Anser albifrons", 120),
    ("caribou", "Caribou", "Rangifer tarandus", 120),
    ("sea-lion", "California sea lion", "Zalophus californianus", 120),
    ("cuckoo", "African cuckoo", "Cuculus gularis", 120),
    ("godwit", "Black-tailed godwit", "Limosa limosa", 120),
)


def safe_name(name):
    return "".join(char if " " <= char <= "~" and char not in ',"' else " " for char in name).strip()


def load_normalized(path, tracks):
    with path.open(newline="", encoding="ascii") as source:
        for row in csv.DictReader(source):
            tracks[row["species"], row["bird_id"]][int(row["unix_seconds"])] = (
                float(row["latitude"]), float(row["longitude"])
            )


def load_curated(path, tracks, primary_species):
    with path.open(encoding="utf-8") as source:
        archive = json.load(source)
    for study in archive["studies"]:
        species = CURATED_SPECIES.get(study["species"])
        if not species or species in primary_species or not study["license"].startswith("CC0"):
            continue
        for animal in study["tracks"]:
            label = safe_name(f'{study["id"]}/{animal["id"]}')
            for lon, lat, millis in animal["points"]:
                tracks[species, label].setdefault(int(millis // 1000), (lat, lon))


def load_raw(path, tracks, stem, species, taxon, interval):
    open_source = gzip.open if path.suffix == ".gz" else open
    start = unix_seconds("2018-07-01") if stem == "gull-2018" else 0
    end = unix_seconds("2018-11-16") if stem == "gull-2018" else unix_seconds("2100-01-01")
    buckets = defaultdict(dict)
    with open_source(path, "rt", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        for row in reader:
            if (row.get("visible") or "").lower() in {"false", "0"} or any(
                (row.get(flag) or "").lower() in {"true", "1"}
                for flag in ("manually-marked-outlier", "algorithm-marked-outlier", "import-marked-outlier")
            ):
                continue
            row_taxon = (row.get("individual-taxon-canonical-name") or taxon).casefold()
            if not row_taxon.startswith(taxon.casefold()):
                continue
            try:
                time = unix_seconds(row["timestamp"])
                lat = float(row["location-lat"])
                lon = float(row["location-long"])
                bird = safe_name(row["individual-local-identifier"])
            except (KeyError, AttributeError, ValueError, TypeError, OverflowError):
                continue
            if not bird or not (start <= time < end and -90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            fixes = buckets[bird]
            bucket = time // (interval * 60)
            if bucket not in fixes or time < fixes[bucket][0]:
                fixes[bucket] = time, lat, lon
    retained = 0
    for bird, fixes in buckets.items():
        points = {time: (lat, lon) for time, lat, lon in fixes.values()}
        if can_draw_trail(points):
            tracks[species, f"{stem}/{bird}"] = points
            retained += 1
    return retained


def can_draw_trail(fixes):
    points = sorted(fixes.items())
    for (ta, a), (tb, b) in zip(points, points[1:]):
        hours = (tb - ta) / 3600
        if not 0 < hours <= 24:
            continue
        lat1, lat2 = math.radians(a[0]), math.radians(b[0])
        longitude_delta = math.radians(b[1] - a[1])
        haversine = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(longitude_delta / 2) ** 2
        distance = 12742 * math.asin(min(1, math.sqrt(haversine)))
        if 0.1 <= distance <= hours * 200:
            return True
    return False


def write_archive(path, tracks):
    summary = defaultdict(lambda: [0, 0])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", newline="", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(TRACK_HEADER)
            for (species, bird), fixes in sorted(tracks.items()):
                if not can_draw_trail(fixes):
                    continue
                for time, (lat, lon) in sorted(fixes.items()):
                    writer.writerow((species, bird, time, lat, lon))
                summary[species][0] += 1
                summary[species][1] += len(fixes)
        if not summary:
            raise ValueError("no animals have adjacent observations with valid movement")
        os.chmod(temporary, 0o644)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("stork_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tracks = defaultdict(dict)
    load_normalized(args.stork_csv, tracks)
    primary = {"White stork"}
    for stem, species, taxon, interval in RAW_SOURCES:
        for extension in (".csv.gz", ".csv"):
            path = args.source_directory / (stem + extension)
            if path.is_file():
                print(f"Reading {path.name}", flush=True)
                if load_raw(path, tracks, stem, species, taxon, interval):
                    primary.add(species)
                break
    load_curated(args.source_directory / "curated.json", tracks, primary)
    summary = write_archive(args.output, tracks)
    for species, (animals, fixes) in summary.items():
        print(f"{species}: {animals} animals, {fixes:,} fixes")


if __name__ == "__main__":
    main()
