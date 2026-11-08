import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from wildcast.migration_data import DAY, canonicalize, match_targets, past_indices, prepare
from wildcast.data import read_events


def test_migration_csv_preserves_distinct_numeric_identifier_strings(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("timestamp,location-lat,location-long,individual-local-identifier,individual-taxon-canonical-name,study-name,sensor-type\n"
                    "2020-01-01T00:00:00Z,30,20,00029,bird,study,gps\n"
                    "2020-01-02T00:00:00Z,31,21,29,bird,study,gps\n")
    f = read_events(path, "source", preserve_identifiers=True)
    assert f.animal.nunique() == 2
    assert f.animal.str.endswith(":00029").any()


def test_past_memory_crosses_gaps_without_using_future():
    times = np.array([0, 2, 100, 365, 366]) * DAY
    ix, mask = past_indices(times, 4, np.array([0, 1, 266, 364, 365]))
    assert (times[ix[mask]] <= times[4] - np.array([0, 1, 266, 364, 365])[mask] * DAY).all()
    assert mask[-1] and times[ix[-1]] == 0
    ix, mask = past_indices(times, 2, np.array([1., 90.]))
    assert not mask[0]  # No invented position inside the 98-day gap.


def test_future_matching_retains_observed_time_and_datagaps():
    times = np.array([0, 1.02, 7.1, 35]) * DAY
    ix, mask = match_targets(times, 0, [1, 7, 14, 30])
    assert mask.tolist() == [True, True, False, False]
    assert times[ix[0]] != DAY


def test_shared_stationary_coordinates_do_not_link_different_animals():
    dates = pd.date_range("2019-01-01", periods=8, freq="h", tz="UTC")
    rows = []
    for source in ["one", "two"]:
        for t in dates:
            rows.append(dict(animal=f"{source}:study:bird:1", species="bird", source=source,
                             study=source, timestamp=t, lat=30., lon=20.))
    _, mapping, links = canonicalize(pd.DataFrame(rows))
    assert len(set(mapping.values())) == 2 and not links


def test_local_numbers_in_separate_studies_within_one_package_are_not_identity():
    rows = []
    for study, offset in [("first", 0), ("second", 10)]:
        for i, t in enumerate(pd.date_range("2019-01-01", periods=8, freq="h", tz="UTC")):
            rows.append(dict(animal=f"source:{study}:bird:1", species="bird", source="source",
                             study=study, timestamp=t, lat=30 + offset + i / 100, lon=20.))
    _, mapping, links = canonicalize(pd.DataFrame(rows))
    assert len(set(mapping.values())) == 2 and not links


def test_moving_cross_source_aliases_stay_in_one_physical_group():
    dates = pd.date_range("2019-01-01", periods=8, freq="h", tz="UTC")
    rows = []
    for source in ["one", "two"]:
        for i, t in enumerate(dates):
            rows.append(dict(animal=f"{source}:study:bird:1", species="bird", source=source,
                             study=source, timestamp=t, lat=30 + i / 100, lon=20.))
    f, mapping, links = canonicalize(pd.DataFrame(rows))
    assert len(set(mapping.values())) == 1 and len(f) == 8
    assert links[0]["evidence"] == "exact_species_time_GPS_overlap"


def test_calendar_purge_and_sealed_fresh_test(tmp_path):
    rows = []
    # Include a deterministic fresh held-out identity and a temporal identity.
    chosen = {}
    for i in range(100):
        name = f"source:study:bird:{i}"
        bucket = int(hashlib.sha256(f"migration-v1:1701:{name}".encode()).hexdigest()[:12], 16) % 100
        if bucket < 15: chosen.setdefault("test", name)
        if bucket >= 30: chosen.setdefault("temporal", name)
    dates = pd.date_range("2019-01-01", "2022-05-01", freq="D", tz="UTC")
    for j, name in enumerate(chosen.values()):
        for i, t in enumerate(dates):
            rows.append(dict(animal=name, species="bird", source="source", study="study",
                timestamp=t, lat=20 + j * 5 + np.sin(i / 50), lon=30 + i / 1000))
    path = tmp_path / "observations.parquet"
    pd.DataFrame(rows).to_parquet(path)
    old = tmp_path / "old.json"; old.write_text(json.dumps({"animals": {}}))
    output = tmp_path / "cache"
    manifest = prepare([path], output, existing_manifest=old)
    with np.load(output / "cases.npz") as a:
        test = a["split"] >= 3
        assert test.any() and not a["target_mask"][test].any()
        assert not a["y"][test].any() and not a["target_latlon"][test].any()
        assert not a["target_time"][test].any()
        train = a["split"] == 0
        assert (a["target_time"][train][a["target_mask"][train]] < pd.Timestamp("2021-01-01", tz="UTC").timestamp()).all()
        assert (a["history_time"][a["history_mask"]] <= np.broadcast_to(a["origin"][:, 2, None], a["history_time"].shape)[a["history_mask"]]).all()
        assert a["history"].shape[-1] == len(manifest["history_names"])
        assert a["static"].shape[-1] == len(manifest["static_names"])
    with pytest.raises(FileExistsError): prepare([path], output, existing_manifest=old)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"fold_groups": {}, "freshness_exclusion_animals": [chosen["test"]],
                                    "ambiguous": [{"animals": [chosen["test"]]}]}))
    conservative = prepare([path], tmp_path / "conservative", existing_manifest=old, reference_audit=reference)
    assert conservative["assignments"][chosen["test"]] != "test_individual"
    assert chosen["test"] in conservative["ambiguous_identity_records"]
