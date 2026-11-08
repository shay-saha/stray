import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildcast.breadth_clean import _eligible_times, _endpoint_support, availability_audit, clean, named_species_key, prepare_policy, source_risks
from wildcast.marine_intake import sha256


def row(**kwargs):
    return {"timestamp": "2020-01-01 00:00:00", "location-lat": "1.0000", "location-long": "2.00000",
            "individual-local-identifier": "001", "tag-local-identifier": "090", "sensor-type": "gps",
            "study-name": "study", "individual-taxon-canonical-name": "Example species", **kwargs}


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def acquisition(tmp_path, rows, sensor_metadata=None):
    root = tmp_path / "raw"; directory = root / "source1"
    (directory / "originals").mkdir(parents=True)
    original = directory / "originals/events.csv"
    write(original, rows)
    part = directory / "events__part0001.csv"; part.write_bytes(original.read_bytes())
    md5 = {"checkSumAlgorithm": "MD5", "value": hashlib.md5(original.read_bytes()).hexdigest()}
    (directory / "README.txt").write_text("GPS locations are original fixes.\nTerms of Use\nData Attributes\nmodelled: interpolated locations.\n")
    source = {"id": "source1", "acquisition_complete": True, "metadata": {"dc.rights.uri": ["https://creativecommons.org/publicdomain/zero/1.0/"]},
              "files": [{"name": "events.csv", "stored_as": "originals/events.csv", "role": "events", "status": "complete",
                         "bytes": original.stat().st_size, "sha256": sha256(original), "published_checksum": md5,
                         "derived": [{"chunks": [{"name": part.name, "rows": len(rows), "sha256": sha256(part)}]}]}]}
    (directory / "source.json").write_text(json.dumps(source))
    plan = {"format": "wildcast-breadth-metadata-plan-v1", "selected_sources": [{"id": "source1", "title": "Observed GPS movement",
             "rights_uris": ["https://creativecommons.org/publicdomain/zero/1.0/"], "sensors_metadata": sensor_metadata or ["GPS"],
             "files": [{"name": "events.csv", "bytes": original.stat().st_size, "published_checksum": md5, "included_for_intake": True}]}]}
    path = tmp_path / "plan.json"; path.write_text(json.dumps(plan))
    return root, path, directory


def test_document_processing_hold_does_not_confuse_dictionary_or_negation():
    assert source_risks("GPS movement", "These are original fixes and not the interpolated positions used in the paper.") == []
    assert "source_specific_processed_positions" in source_risks("GPS movement", "The locations were interpolated to hourly intervals.")
    assert "processing_model_in_title_requires_raw_fix_evidence" in source_risks("State-space analysis of tracks", "GPS tags attached")


def test_streamed_flags_clock_quarantine_strings_and_no_legacy_timestamp(tmp_path):
    rows = [row(), row(timestamp="2020-01-02"), row(timestamp="2058-01-01"),
            row(**{"individual-local-identifier": "002"}), row(timestamp="2020-01-02", **{"individual-local-identifier": "002"}),
            row(modelled="true", **{"individual-local-identifier": "processed"})]
    root, plan, directory = acquisition(tmp_path, rows)
    policy = tmp_path / "policy.json"; prepare_policy(root, plan, policy)
    result = clean(policy, tmp_path / "clean", workers=1, chunksize=1)
    frame = pd.read_parquet(tmp_path / "clean/processed/observations.parquet")
    assert result["rows"] == 2 and result["source_animal_ids"] == 1
    assert frame.individual_id_reported.tolist() == ["002", "002"]
    assert "timestamp" not in frame and frame.timestamp_utc.isna().all()
    assert json.loads(frame.reported_fields_json.iloc[0])["location-long"] == "2.00000"
    assert frame.original_source_record.tolist() == [4, 5]
    assert any(reason == "source_clock_anomaly" for reason in result["held_animals"].values())
    assert sha256(directory / "originals/events.csv") == frame.original_source_sha256.iloc[0]
    with pytest.raises(ValueError, match="immutable"):
        clean(policy, tmp_path / "clean", workers=1)


def test_explicit_single_sensor_fallback_is_provenance_not_fabricated_raw_field(tmp_path):
    rows = [row(), row(timestamp="2020-01-02")]
    for r in rows:
        r.pop("sensor-type"); r.pop("study-name")
    root, plan, _ = acquisition(tmp_path, rows)
    policy = tmp_path / "policy.json"; prepare_policy(root, plan, policy)
    clean(policy, tmp_path / "clean", workers=1)
    frame = pd.read_parquet(tmp_path / "clean/processed/observations.parquet")
    assert frame.method_evidence.eq("single_GPS_source_or_explicit_GPS_file").all()
    assert frame.sensor_reported.eq("").all()
    for raw in frame.reported_fields_json.map(json.loads):
        assert "sensor-type" not in raw and not any(k.startswith("__intake") for k in raw)


def test_mixed_sensor_without_row_method_and_hardware_only_identity_are_held(tmp_path):
    rows = [row(), row(timestamp="2020-01-02")]
    for r in rows: r.pop("sensor-type")
    root, plan, _ = acquisition(tmp_path, rows, ["GPS", "Argos"])
    policy = prepare_policy(root, plan, tmp_path / "policy.json")
    assert not policy["jobs"]
    assert any(v["reason"] == "mixed_or_unknown_sensor_without_row_method" for v in policy["held"])
    # A tag is not a documented animal ID: do not join redeployments as one track.
    rows = [row()]
    rows[0].pop("individual-local-identifier")
    root2 = tmp_path / "second"; root2.mkdir()
    raw2, plan2, _ = acquisition(root2, rows)
    policy2 = prepare_policy(raw2, plan2, tmp_path / "policy2.json")
    assert not policy2["jobs"]


def test_pin_rejects_original_or_policy_changes(tmp_path):
    root, plan, directory = acquisition(tmp_path, [row(), row(timestamp="2020-01-02")])
    policy = tmp_path / "policy.json"; prepare_policy(root, plan, policy)
    (directory / "originals/events.csv").write_text("changed")
    with pytest.raises(ValueError, match="Original source bytes"):
        clean(policy, tmp_path / "clean", workers=1)


def test_long_availability_matches_existing_time_law_and_short_tracks_have_real_short_support():
    from wildcast.migration_data import DAY, LAGS, match_targets, past_indices
    assert np.array_equal(LAGS, __import__("wildcast.breadth_clean", fromlist=["LONG_LAGS"]).LONG_LAGS)
    times = pd.date_range("2019-01-01", periods=120 * 4, freq="6h", tz="UTC").asi8 / 1e9
    # Independent application of the actual corpus routines to this irregular
    # time law checks exact tie handling and cadence, not coordinate labels.
    expected, targets, last = 0, np.zeros(7, int), -np.inf
    for i in range(len(times)):
        if i < 8 or times[i]-times[0] < 7*DAY or times[i]-last < 7*DAY: continue
        _, mask = match_targets(times, i, [1, 3, 7, 14, 30, 60, 90])
        _, history_mask = past_indices(times, i)
        if mask.sum() >= 2 and history_mask.sum() >= 5:
            expected += 1; targets += mask; last = times[i]
    origins, counts = _eligible_times(times)
    assert origins == expected and np.array_equal(counts, targets)
    brief = pd.date_range("2019-01-01", periods=24 * 12, freq="5min", tz="UTC").asi8 / 1e9
    assert _eligible_times(brief)[0] == 0
    assert _eligible_times(brief, short=True)[0] > 0
    assert _eligible_times(times, cutoff=pd.Timestamp("2018-01-01", tz="UTC").timestamp())[0] == 0


def test_availability_report_separates_short_support_and_calendar_cutoff(tmp_path):
    dates = pd.date_range("2022-01-01", periods=24*12, freq="5min", tz="UTC")
    path = tmp_path / "times.parquet"
    pd.DataFrame({"animal": "source:short", "species": "Short taxon", "timestamp_nominal": dates}).to_parquet(path)
    report = availability_audit([path], tmp_path / "availability.json")
    for key, value in {"long_anytime_taxa": 0, "long_pre_cutoff_taxa": 0,
                       "short_anytime_taxa": 1, "short_pre_cutoff_taxa": 0}.items():
        assert report["counts"][key] == value


def test_taxonomic_count_does_not_promote_unknowns_hybrids_or_subspecies():
    assert named_species_key(" Canis  lupus familiaris ") == "Canis lupus"
    assert named_species_key("Unknown") == ""
    assert named_species_key("Anser anser x Anser domesticus") == ""
    assert named_species_key("Larus sp.") == ""


def test_one_hour_endpoint_support_does_not_require_three_hour_labels_or_scale_daily_data():
    brief = pd.date_range("2019-01-01", periods=3*12, freq="5min", tz="UTC").asi8 / 1e9
    assert _endpoint_support(brief, 1/24) > 0
    assert _eligible_times(brief, short=True)[0] == 0
    assert _endpoint_support(brief, .5) == 0
    assert _endpoint_support(brief, 1/24, long_history=True) == 0
    daily = pd.date_range("2019-01-01", periods=30, freq="D", tz="UTC").asi8 / 1e9
    assert _endpoint_support(daily, 1/24) == 0


def test_actual_executor_byte_exact_unknown_rows_to_parallel_clean(tmp_path):
    root, plan_path, directory = acquisition(tmp_path, [row(), row(timestamp="2020-01-02")])
    script = Path(__file__).resolve().parents[1] / "scripts/migration_breadth_execute.py"
    spec = importlib.util.spec_from_file_location("breadth_execute_integration", script)
    executor = importlib.util.module_from_spec(spec); spec.loader.exec_module(executor)
    source_path = directory / "source.json"; source = json.loads(source_path.read_text())
    original = directory / "originals/events.csv"; record = source["files"][0]
    storage = executor.Storage(tmp_path, 10_000_000, 0)
    intake = executor.Intake({"quotas": {"maximum_original_file_bytes": 10_000_000}}, tmp_path, storage)
    record["derived"] = intake.extract(original, record, directory)
    assert record["derived"][0]["chunks"][0]["rows"] is None
    assert (directory / "events.csv").is_symlink()
    source_path.write_text(json.dumps(source))
    policy_path = tmp_path / "policy.json"
    policy = prepare_policy(root, plan_path, policy_path)
    assert policy["jobs"][0]["events"][0]["source_record_offset"] == 0
    result = clean(policy_path, tmp_path / "clean", workers=2, chunksize=1)
    assert result["rows"] == 2
    actual = pd.read_parquet(tmp_path / "clean/processed/observations.parquet")
    assert actual.original_source_record.tolist() == [1, 2]


def test_byte_exact_cp1252_is_explicit_and_malformed_extra_fields_are_not_silently_shifted(tmp_path):
    root, plan_path, directory = acquisition(tmp_path, [row(**{"individual-local-identifier": "café001"}),
                                                       row(timestamp="2020-01-02", **{"individual-local-identifier": "café001"})])
    source_path = directory / "source.json"; source = json.loads(source_path.read_text());record=source["files"][0]
    original=directory / record["stored_as"];original.write_bytes(original.read_text().encode("cp1252"))
    chunk=directory / record["derived"][0]["chunks"][0]["name"];chunk.write_bytes(original.read_bytes())
    record.update(bytes=original.stat().st_size,sha256=sha256(original),published_checksum={"checkSumAlgorithm":"MD5","value":hashlib.md5(original.read_bytes()).hexdigest()})
    record["derived"][0]["chunks"][0].update(sha256=sha256(chunk),rows=None)
    source_path.write_text(json.dumps(source))
    plan=json.loads(plan_path.read_text());plan["selected_sources"][0]["files"][0].update(bytes=record["bytes"],published_checksum=record["published_checksum"]);plan_path.write_text(json.dumps(plan))
    policy=prepare_policy(root,plan_path,tmp_path/"policy.json")
    assert policy["jobs"][0]["events"][0]["encoding"]=="cp1252"
    clean(tmp_path/"policy.json",tmp_path/"clean",workers=1)
    actual=pd.read_parquet(tmp_path/"clean/processed/observations.parquet")
    assert actual.individual_id_reported.eq("café001").all()
    from wildcast.breadth_clean import _csv_batches
    bad=tmp_path/"malformed.csv";bad.write_text("a,b\n1,2,3\n")
    with pytest.raises(pd.errors.ParserWarning):
        list(_csv_batches(bad,10,"utf-8-sig"))


def test_ancillary_duplicate_blank_headers_do_not_abort_real_gps_source(tmp_path):
    root, plan_path, directory = acquisition(tmp_path, [row(), row(timestamp="2020-01-02")])
    ancillary=directory/"originals/vegetation.csv";ancillary.write_text("Site,X,Y,,,\nA,1,2,,,\n")
    linked=directory/"vegetation.csv";linked.symlink_to(Path("originals/vegetation.csv"))
    md5={"checkSumAlgorithm":"MD5","value":hashlib.md5(ancillary.read_bytes()).hexdigest()}
    source_path=directory/"source.json";source=json.loads(source_path.read_text())
    source["files"].append({"name":"vegetation.csv","stored_as":"originals/vegetation.csv","role":"events","status":"complete",
                            "bytes":ancillary.stat().st_size,"sha256":sha256(ancillary),"published_checksum":md5,
                            "derived":[{"chunks":[{"name":"vegetation.csv","sha256":sha256(ancillary),"rows":None}]}]})
    source_path.write_text(json.dumps(source))
    plan=json.loads(plan_path.read_text());plan["selected_sources"][0]["files"].append({"name":"vegetation.csv","bytes":ancillary.stat().st_size,"published_checksum":md5,"included_for_intake":True});plan_path.write_text(json.dumps(plan))
    policy=prepare_policy(root,plan_path,tmp_path/"policy.json")
    assert len(policy["jobs"])==1 and len(policy["jobs"][0]["events"])==1
    assert any(v["reason"]=="unsupported_event_schema" for v in policy["held"])


def test_unknown_timezone_suffix_is_not_silently_stripped_into_a_valid_clock(tmp_path):
    values=[row(),row(timestamp="2020-01-02"),row(timestamp="2020-01-02 18:00:0S"),
            row(**{"individual-local-identifier":"safe002"}),row(timestamp="2020-01-02",**{"individual-local-identifier":"safe002"})]
    root,plan_path,_=acquisition(tmp_path,values)
    prepare_policy(root,plan_path,tmp_path/"policy.json")
    result=clean(tmp_path/"policy.json",tmp_path/"clean",workers=1,chunksize=2)
    actual=pd.read_parquet(tmp_path/"clean/processed/observations.parquet")
    assert actual.individual_id_reported.eq("safe002").all()
    assert result["held_animals"]["source1:study:Example species:001"]=="source_clock_anomaly"
    assert not any("__intake" in k for r in actual.reported_fields_json.map(json.loads) for k in r)
