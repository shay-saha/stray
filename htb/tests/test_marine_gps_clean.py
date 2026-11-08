import csv
import json

import pandas as pd
import pytest

from wildcast.marine_gps_clean import clean, clean_chunk, exact_fix_links, identity_audit
from wildcast.marine_intake import sha256


def raw_row(**values):
    return {"timestamp": "2020-01-01 12:00:00.000", "location-lat": "-7.1000", "location-long": "73.01000",
            "sensor-type": "gps", "individual-local-identifier": "0001", "tag-local-identifier": "0009",
            "individual-taxon-canonical-name": "Chelonia mydas", "study-name": "Chagos",
            "visible": "true", "gps:satellite-count": "7", "wc:residual": "12.3", **values}


def declaration(path=None):
    return {"path": str(path) if path else "original.csv", "sha256": sha256(path) if path else "a" * 64,
            "spec": {"source_id": "release", "processing": "observed", "time_standard": "UTC_or_GPS"},
            "evidence": [{"url": "https://example.org/source", "claim": "GPS observed fixes"}]}


def write_csv(path, values):
    columns = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(values)


def test_quality_and_processing_flags_are_not_lost_or_promoted():
    rows = [raw_row(), raw_row(modelled="true"), raw_row(interpolated="true"), raw_row(smoothed="true"),
            raw_row(**{"sensor-type": "argos-doppler-shift"}), raw_row(visible="false"),
            raw_row(**{"algorithm-marked-outlier": "true"}), raw_row(**{"location-lat": "999"})]
    frame, report = clean_chunk(pd.DataFrame(rows).fillna(""), "original.csv", declaration(), [], "a" * 64)
    assert len(frame) == 1 and report["input_rows"] == 8
    assert frame.timestamp_utc.isna().all()
    assert frame.time_standard_status.iloc[0] == "UTC_or_GPS"
    assert "timestamp" not in frame
    assert frame.timestamp_nominal.iloc[0] == pd.Timestamp("2020-01-01T12:00:00Z")
    assert frame.individual_id_reported.iloc[0] == "0001"
    assert json.loads(frame.reported_fields_json.iloc[0])["wc:residual"] == "12.3"
    assert frame.longitude_reported.iloc[0] == "73.01000"


def test_reported_deployment_end_excludes_floating_tag_and_qc_needs_evidence():
    rows = [raw_row(), raw_row(timestamp="2020-01-03 00:00:00"), raw_row(**{"gps:satellite-count": "5"})]
    reference = [{"animal-id": "0001", "animal-taxon": "Chelonia mydas", "tag-id": "0009",
                  "deploy-on-date": "2020-01-01", "deploy-off-date": "2020-01-02", "animal-sex": "f"}]
    dec = declaration()
    dec["require_deployment_bounds"] = True
    dec["quality_rules"] = [{"column": "gps:satellite-count", "operator": ">", "value": 5,
                              "evidence": "source explicitly requires satellite count > 5"}]
    frame, report = clean_chunk(pd.DataFrame(rows), "original.csv", dec, reference, "a" * 64)
    assert len(frame) == 1
    assert report["outside_or_unresolved_deployment_rows"] == 1
    assert report["quality_rule:gps:satellite-count"] == 1
    assert json.loads(frame.reference_fields_json.iloc[0])["animal-sex"] == "f"
    dec["quality_rules"][0].pop("evidence")
    with pytest.raises(ValueError, match="evidence"):
        clean_chunk(pd.DataFrame(rows), "original.csv", dec, reference, "a" * 64)


def test_future_clock_data_held_without_correction():
    rows = pd.DataFrame([raw_row(timestamp="2058-07-07 12:00:00"), raw_row(timestamp="2020-01-01T12:00:00Z")])
    frame, report = clean_chunk(rows, "original.csv", declaration(), [], "a" * 64)
    assert len(frame) == 1 and report["future_clock_rows"] == 1
    assert frame.timestamp_utc.notna().all()
    assert frame.timestamp_reported.iloc[0] == "2020-01-01T12:00:00Z"


def test_new_core_holds_single_fixes_checks_hash_and_never_overwrites(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    path = root / "gps.csv"
    write_csv(path, [raw_row(), raw_row(timestamp="2020-01-02"),
                     raw_row(**{"individual-local-identifier": "deployment-only"})])
    plan = {"files": [declaration(path)]}
    result = clean([root], plan, tmp_path / "core", chunksize=1)
    assert result["rows"] == 2 and result["animals"] == 1
    frame = pd.read_parquet(tmp_path / "core/processed/observations.parquet")
    assert frame.source_record.tolist() == [1, 2]
    assert "timestamp" not in frame
    with pytest.raises(ValueError, match="empty"):
        clean([root], plan, tmp_path / "core")
    plan["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        clean([root], plan, tmp_path / "bad")


def test_exact_fix_identity_and_sealed_cohort_carryover(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    path = root / "gps.csv"
    rows = [raw_row(timestamp=f"2020-01-01 {hour:02}:00:00", **{"location-long": str(73 + hour / 100)}) for hour in range(6)]
    write_csv(path, rows)
    core = tmp_path / "core"
    clean([root], {"files": [declaration(path)]}, core)
    new = pd.read_parquet(core / "processed/observations.parquet")
    old = new[["timestamp_nominal", "lat", "lon", "animal", "species", "study", "source"]].rename(columns={"timestamp_nominal": "timestamp"})
    old["animal"], old["source"], old["study"] = "old:study:Chelonia mydas:linked", "old", "old:study"
    old_path = tmp_path / "old.parquet"; old.to_parquet(old_path)
    links, _ = exact_fix_links(core / "processed/observations.parquet", [old_path])
    assert len(links) == 1 and links[0]["shared_fixes"] == 6
    manifest = tmp_path / "frozen.json"
    old_name = old.animal.iloc[0]
    manifest.write_text(json.dumps({"identity_mapping": {old_name: old_name}, "exclusion_groups": {old_name: old_name},
                                    "assignments": {old_name: "test_individual"}}))
    audit = identity_audit(core, [old_path], [], manifest)
    binding = audit["new_animal_cohort_bindings"][0]
    assert binding["status"] == "existing_sealed_identity_or_risk_group"
    assert binding["existing_cohort_members"][0]["assignment"] == "test_individual"
    assert len(audit["freshness_exclusion_animals"]) == 1
    assert json.loads(manifest.read_text())["assignments"][old_name] == "test_individual"


def test_stationary_colocated_fixes_do_not_prove_identity(tmp_path):
    times = pd.date_range("2020-01-01", periods=6, freq="h", tz="UTC")
    new = pd.DataFrame({"timestamp_nominal": times, "lat": 1., "lon": 2., "species": "seal", "animal": "new"})
    old = new.rename(columns={"timestamp_nominal": "timestamp"}).assign(animal="old")
    new.to_parquet(tmp_path / "new.parquet"); old.to_parquet(tmp_path / "old.parquet")
    links, _ = exact_fix_links(tmp_path / "new.parquet", [tmp_path / "old.parquet"])
    assert links == []
