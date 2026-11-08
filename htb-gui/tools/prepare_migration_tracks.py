#!/usr/bin/env python3
"""Convert UTC bird GPS observations to the atlas's offline migration CSV."""

import argparse
import csv
import gzip
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


TRACK_HEADER = ("species", "bird_id", "unix_seconds", "latitude", "longitude")
SOURCE_COLUMNS = {
    "species": ("species", "individual-taxon-canonical-name", "scientific_name"),
    "bird": ("bird_id", "individual-local-identifier", "individual_id", "name", "id"),
    "time": ("unix_seconds", "timestamp", "datetime", "time"),
    "latitude": ("latitude", "location-lat", "lat"),
    "longitude": ("longitude", "location-long", "lon", "long"),
}
GAP_SECONDS = 12 * 60 * 60


def unix_seconds(value):
    """Return whole UTC seconds. A source timestamp without an offset is UTC."""
    try:
        return math.floor(float(value))
    except ValueError:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return math.floor(timestamp.timestamp())


def runtime_label(value):
    value = value.strip()
    if not value or any(not 32 <= ord(char) <= 126 or char in ',"' for char in value):
        raise ValueError(f"expected a printable ASCII label without commas or quotes, got {value!r}")
    return value


def resolve_columns(fieldnames, args):
    if not fieldnames:
        raise ValueError("the source CSV has no header")
    result = {}
    for field, aliases in SOURCE_COLUMNS.items():
        if field == "species" and args.species is not None:
            continue
        requested = getattr(args, f"{field}_column")
        column = requested or next((alias for alias in aliases if alias in fieldnames), None)
        if column not in fieldnames:
            raise ValueError(f"missing {field} column. Use --{field}-column to choose a source column")
        result[field] = column
    return result


def read_tracks(args):
    tracks = defaultdict(dict)
    skipped = 0
    species_override = runtime_label(args.species) if args.species is not None else None
    open_source = gzip.open if args.source.suffix == ".gz" else open
    with open_source(args.source, "rt", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = resolve_columns(reader.fieldnames, args)
        for row in reader:
            if (row.get("visible") or "").lower() in {"false", "0"} or any(
                (row.get(flag) or "").lower() in {"true", "1"}
                for flag in ("manually-marked-outlier", "algorithm-marked-outlier")
            ):
                skipped += 1
                continue
            try:
                second = unix_seconds(row[columns["time"]])
                latitude = float(row[columns["latitude"]])
                longitude = float(row[columns["longitude"]])
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    raise ValueError("location outside geographic bounds")
            except (ValueError, TypeError, OverflowError, OSError):
                skipped += 1
                continue
            if (args.start is not None and second < args.start) or (args.end is not None and second >= args.end):
                continue
            try:
                species = species_override or runtime_label(row[columns["species"]] or "")
                bird = runtime_label(row[columns["bird"]] or "")
            except ValueError as error:
                raise ValueError(f"line {reader.line_num}: {error}") from error
            # A duplicate second has no duration for playback. Keep the first observation in the source.
            tracks[species, bird].setdefault(second, (latitude, longitude))
    return tracks, skipped


def select_fixes(fixes, interval_seconds):
    points = sorted(fixes.items())
    if len(points) < 2:
        return
    yield points[0]
    last_saved = points[0][0]
    for index, point in enumerate(points[1:-1], start=1):
        second = point[0]
        next_second = points[index + 1][0]
        previous_second = points[index - 1][0]
        if (second - last_saved >= interval_seconds or second - previous_second > GAP_SECONDS
                or next_second - second > GAP_SECONDS):
            yield point
            last_saved = second
    yield points[-1]


def write_tracks(output, tracks, interval_seconds):
    count = 0
    bird_count = 0
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", newline="", dir=output.parent,
                                         delete=False) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.writer(temporary, lineterminator="\n")
            writer.writerow(TRACK_HEADER)
            for (species, bird), fixes in sorted(tracks.items()):
                if len(fixes) < 2:
                    continue
                bird_count += 1
                for second, (latitude, longitude) in select_fixes(fixes, interval_seconds):
                    writer.writerow((species, bird, second, latitude, longitude))
                    count += 1
        if count == 0:
            raise ValueError("the selection has no birds with two valid fixes")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return count, bird_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="plain or gzip-compressed CSV of published GPS observations")
    parser.add_argument("output", type=Path, help="destination CSV for Wildlife Atlas")
    parser.add_argument("--species", help="use this common name for every row in a single-species source")
    parser.add_argument("--start", help="inclusive UTC ISO date or datetime")
    parser.add_argument("--end", help="exclusive UTC ISO date or datetime")
    parser.add_argument("--min-interval-minutes", type=int, default=0,
                        help="thin dense fixes while keeping endpoints and both sides of gaps over 12 hours")
    for field in SOURCE_COLUMNS:
        parser.add_argument(f"--{field}-column", help=f"source header for {field}")
    args = parser.parse_args()
    if args.min_interval_minutes < 0 or args.min_interval_minutes >= GAP_SECONDS // 60:
        parser.error(f"--min-interval-minutes must be between 0 and {GAP_SECONDS // 60 - 1}")
    if args.source.resolve() == args.output.resolve():
        parser.error("the source and output files must be different")
    try:
        args.start = unix_seconds(args.start) if args.start is not None else None
        args.end = unix_seconds(args.end) if args.end is not None else None
        if args.start is not None and args.end is not None and args.start >= args.end:
            raise ValueError("--start must be before --end")
        tracks, skipped = read_tracks(args)
        count, birds = write_tracks(args.output, tracks, args.min_interval_minutes * 60)
    except (ValueError, OverflowError, OSError, csv.Error) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Wrote {count:,} fixes from {birds:,} birds to {args.output}")
    if skipped:
        print(f"Skipped {skipped:,} source rows with invalid or flagged observations", file=sys.stderr)


if __name__ == "__main__":
    main()
