import csv
import json

import numpy as np
import pandas as pd
import pytest

from wildcast.marine_intake import SourceSpec, census, ingest, read_observations, sha256


def table(tmp_path, rows, name="events.csv"):
    path = tmp_path / name
    headers = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def fix(**updates):
    return {"timestamp": "2021-02-03 12:30:00.000", "location-lat": "-7.1000",
            "location-long": "73.00000", "individual-local-identifier": "0007",
            "tag-local-identifier": "00009", "sensor-type": "gps",
            "individual-taxon-canonical-name": "Chelonia mydas", "visible": "true", **updates}


def spec(**updates):
    return SourceSpec(source_id="release-1", processing="observed", timezone="UTC", **updates)


def test_original_strings_ids_and_source_flags_survive(tmp_path):
    rows = [fix(**{"wc:residual": "14.2", "gps:satellite-count": "7"}),
            fix(visible="false"), fix(**{"algorithm-marked-outlier": "true"}),
            fix(**{"location-lat": "999.000"}), fix(**{"visible": "unreviewed"})]
    frame = read_observations(table(tmp_path, rows), spec())
    assert len(frame) == 5
    assert frame.gps_observation_eligible.tolist() == [True, False, False, False, False]
    assert frame.individual_id_reported.tolist() == ["0007"] * 5
    assert frame.tag_id_reported.tolist() == ["00009"] * 5
    assert frame.reported_latitude.iloc[0] == "-7.1000"
    raw = json.loads(frame.reported_fields_json.iloc[0])
    assert raw["wc:residual"] == "14.2" and raw["gps:satellite-count"] == "7"
    assert raw["location-long"] == "73.00000"
    assert frame.timestamp_utc.iloc[0] == pd.Timestamp("2021-02-03T12:30:00Z")


@pytest.mark.parametrize("processing", ["modelled", "interpolated", "smoothed", "state_space", "kalman", "unknown"])
def test_processed_or_unresolved_positions_never_become_gps_truth(tmp_path, processing):
    frame = read_observations(table(tmp_path, [fix()]),
                              SourceSpec("processed", processing=processing, timezone="UTC"))
    assert not frame.gps_observation_eligible.any()
    assert not frame.is_direct_observation.any()


def test_row_modelled_overrides_file_claim_and_false_does_not_override_ssm(tmp_path):
    path = table(tmp_path, [fix(modelled="true"), fix(modelled="false")])
    frame = read_observations(path, spec())
    assert frame.processing.tolist() == ["modelled", "observed"]
    assert frame.gps_observation_eligible.tolist() == [False, True]
    frame = read_observations(path, SourceSpec("ssm", processing="state_space", timezone="UTC"))
    assert not frame.gps_observation_eligible.any()


def test_argos_diagnostics_are_reported_geometry_not_invented_error(tmp_path):
    rows = [fix(**{"sensor-type": "argos-doppler-shift", "argos:lc": lc,
                   "argos:semi-major": "1.25", "argos:semi-minor": "0.5", "argos:orientation": "83",
                   "argos:error-radius": "1.1"})
            for lc in ["3", "A", "B", "Z", "G", "unexpected"]]
    path = table(tmp_path, rows)
    frame = read_observations(path, spec(units={"ellipse_major": "km", "ellipse_minor": "km",
                                               "ellipse_orientation": "degrees", "error_radius": "km"},
                                         orientation_convention="clockwise_from_north"))
    assert frame.uncertain_observation_eligible.tolist() == [True, True, True, False, False, False]
    assert not frame.gps_observation_eligible.any()
    assert frame.ellipse_major_m.iloc[0] == 1250
    assert frame.ellipse_minor_m.iloc[0] == 500
    assert frame.ellipse_orientation_degrees.iloc[0] == 83
    assert frame.error_radius_reported.iloc[0] == "1.1"
    assert frame.error_radius_m.iloc[0] == 1100
    assert frame.uncertainty_model.eq("not_assumed").all()
    # Identical numbers without documented units must not become metre values.
    unknown_units = read_observations(path, spec())
    assert unknown_units.ellipse_major_m.isna().all()
    assert unknown_units.error_radius_m.isna().all()
    assert "sigma_m" not in frame
    assert unknown_units.argos_lc_reported.tolist() == ["3", "A", "B", "Z", "G", "unexpected"]


def test_no_sensor_or_timezone_inference_and_acoustic_receiver_role(tmp_path):
    path = table(tmp_path, [fix(**{"sensor-type": ""}), fix(**{"sensor-type": "acoustic", "receiver-id": "R09"})])
    frame = read_observations(path, SourceSpec("unknown", processing="observed"))
    assert frame.timestamp_utc.isna().all()
    assert frame.time_status.eq("timezone_unknown").all()
    assert frame.observation_method.iloc[0] == "unknown"
    assert frame.record_kind.iloc[1] == "receiver_detection"
    assert frame.coordinate_role.iloc[1] == "receiver"
    assert frame.receiver_id_reported.iloc[1] == "R09"
    assert not frame.gps_observation_eligible.any()
    assert not frame.uncertain_observation_eligible.any()


def test_actual_dryad_raw_argos_layout_with_date_time_columns(tmp_path):
    path = table(tmp_path, [{"ptt": "000177", "date": "2020-05-04", "time": "12:13:14", "lon": "-44.2",
                             "lat": "31.0", "lc": "B", "sex": "female", "FL": "193"}])
    source = SourceSpec("mako", format="declared_csv", method="argos_doppler", processing="observed",
                        timezone="UTC", columns={"individual": "ptt", "date": "date", "time": "time",
                        "latitude": "lat", "longitude": "lon", "argos_lc": "lc"},
                        evidence_urls=["https://datadryad.org/dataset/doi:10.5061/dryad.2rbnzs7vm"])
    frame = read_observations(path, source)
    assert frame.uncertain_observation_eligible.tolist() == [True]
    assert frame.individual_id_reported.iloc[0] == "000177"
    assert frame.timestamp_utc.iloc[0] == pd.Timestamp("2020-05-04T12:13:14Z")
    assert frame.error_radius_m.isna().all()
    assert json.loads(frame.reported_fields_json.iloc[0])["FL"] == "193"


def test_artifact_provenance_and_no_overwrite(tmp_path):
    path = table(tmp_path, [fix()])
    output = tmp_path / "intake"
    manifest = ingest(path, spec(license="CC0", evidence_urls=["https://example.org/readme"]), output)
    assert manifest["source_sha256"] == sha256(path)
    assert manifest["table_sha256"] == sha256(output / "observations.parquet")
    saved = pd.read_parquet(output / "observations.parquet")
    assert saved.reported_longitude.iloc[0] == "73.00000"
    assert manifest["source_spec"]["license"] == "CC0"
    assert manifest["quality_census"]["groups"][0]["gps_eligible_rows"] == 1
    with pytest.raises(ValueError, match="empty"):
        ingest(path, spec(), output)


def test_conflicts_and_ambiguous_or_malformed_source_schema_rejected(tmp_path):
    frame = read_observations(table(tmp_path, [fix()]), spec(method="argos_doppler"))
    assert not frame.gps_observation_eligible.any()
    assert "sensor_declaration_conflict" in frame.quality_issues_json.iloc[0]
    bad = tmp_path / "duplicate.csv"
    bad.write_text("timestamp,location-lat,location-long,location-long\n2020,1,2,3\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_observations(bad, spec())
    with pytest.raises(ValueError, match="column absent"):
        read_observations(table(tmp_path, [fix()], "second.csv"), spec(columns={"argos_lc": "not_here"}))


def test_utc_or_gps_time_ambiguity_is_not_silently_synchronized(tmp_path):
    frame = read_observations(table(tmp_path, [fix()]), spec(time_standard="UTC_or_GPS"))
    assert frame.timestamp_utc.isna().all()
    assert frame.time_status.iloc[0] == "unresolved_time_standard:UTC_or_GPS"
    assert not frame.gps_observation_eligible.any()
    assert frame.reported_timestamp.iloc[0] == "2021-02-03 12:30:00.000"


def test_streaming_census_deduplicates_content_and_keeps_processing_unknown(tmp_path):
    root = tmp_path / "raw"
    a, b = root / "release-A", root / "release-B"
    a.mkdir(parents=True)
    b.mkdir()
    original = table(a, [fix(timestamp="2020-01-01"), fix(timestamp="2021-01-02")])
    duplicate = b / "copy.csv"
    duplicate.write_bytes(original.read_bytes())
    table(a, [fix(**{"individual-taxon-canonical-name": "Mirounga angustirostris",
                    "sensor-type": "argos-doppler-shift", "modelled": "true", "argos:lc": "B"})], "estimated.csv")
    report = census(root, tmp_path / "census.json", chunksize=1)
    assert report["rows"] == 3
    assert sum("duplicate_of" in f for f in report["files"]) == 1
    groups = {g["species"]: g for g in report["groups"]}
    turtle = groups["Chelonia mydas"]
    assert turtle["processing"] == "unknown"
    assert turtle["source_individual_ids"] == 1
    assert turtle["source_individual_years"] == 2
    assert turtle["source_ids_with_span_at_least_days"]["365"] == 1
    seal = groups["Mirounga angustirostris"]
    assert seal["processing"] == "modelled" and seal["argos_classes_reported"] == {"B": 1}
    filtered = census(root, tmp_path / "turtles.json", taxa=["Chelonia mydas"])
    assert filtered["rows"] == 2 and len(filtered["groups"]) == 1
