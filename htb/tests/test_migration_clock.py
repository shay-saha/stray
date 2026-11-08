import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildcast.migration_clock import (adapt_frame, check_compatibility, export,
                                     fingerprint, make_policy, sha256,
                                     validate_policy, write_policy)


def observations(times=("2020-06-01T12:00:00Z",), source="release", explicit=False):
    nominal = pd.to_datetime(list(times), utc=True, format="ISO8601")
    return pd.DataFrame({"timestamp_nominal": nominal,
                         "timestamp_utc": nominal if explicit else pd.to_datetime([None]*len(times), utc=True),
                         "time_standard_status": "verified_UTC_or_explicit_offset" if explicit else "UTC_or_GPS",
                         "timestamp_reported": [t.removesuffix("Z") for t in times],
                         "source": source, "animal": source+":study:seal:0001", "species": "seal",
                         "study": source+":study", "original_animal": source+":study:seal:0001",
                         "event_id_reported": [str(i) for i in range(len(times))],
                         "lat": np.arange(len(times), dtype=float), "lon": 10.,
                         "processing": "observed", "source_record": np.arange(1, len(times)+1)})


def source_decision(tmp_path, *, mode="bounded_utc_or_gps", source="release"):
    kinds = {"bounded_utc_or_gps": ("UTC_or_GPS", "source_utc_or_gps"),
             "documented_gps": ("GPS", "source_declared_gps"),
             "declared_utc": ("UTC", "source_declared_utc"),
             "row_explicit_utc": ("UTC", "row_explicit_utc")}
    clock, kind = kinds[mode]
    doc = tmp_path / (source+"-README.txt")
    if not doc.exists():doc.write_text("Synthetic source clock documentation: "+clock)
    return {"mode": mode, "documented_clock": clock,
            "evidence": [{"path": str(doc), "kind": kind, "claim": "Synthetic documented clock "+clock}]}


def policy(tmp_path, *, mode="bounded_utc_or_gps", sources=None, manifests=(), **kwargs):
    return make_policy(manifests, sources or {"release": source_decision(tmp_path, mode=mode)}, **kwargs)


def clean_core(tmp_path, frame, name="qc"):
    processed = tmp_path / name / "processed";processed.mkdir(parents=True)
    path = processed / "observations.parquet";frame.to_parquet(path, index=False)
    manifest = processed / "manifest.json"
    manifest.write_text(json.dumps({"schema": "wildcast-breadth-clean-v1", "observations_sha256": sha256(path)}))
    return manifest


def test_historical_standard_bounds_preserve_nominal_columns_and_input(tmp_path):
    frame = observations(("2013-05-01T12:00:00Z", "2016-05-01T12:00:00Z", "2019-05-01T12:00:00Z"))
    before = frame.copy(deep=True)
    out, audit = adapt_frame(frame, policy(tmp_path))
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(out[frame.columns], frame)
    assert out.time_standard_ambiguity_seconds.tolist() == [16., 17., 18.]
    assert out.timestamp.equals(frame.timestamp_nominal)
    assert out.timestamp_utc.isna().all()
    assert out.clock_axis_status.str.contains("not_verified_UTC").all()
    assert audit == {"input_rows": 3, "output_rows": 3, "excluded": {}}
    assert not any("accuracy_seconds" in c or "spatial_error" in c for c in out)


def test_only_ambiguous_calendar_boundary_rows_are_excluded(tmp_path):
    frame = observations(("2020-12-31T23:59:59Z", "2021-01-01T00:00:00Z",
                          "2021-01-01T00:00:17Z", "2021-01-01T00:00:18Z",
                          "2022-01-01T00:00:01Z", "2022-01-01T00:00:18Z"))
    out, audit = adapt_frame(frame, policy(tmp_path))
    assert out.source_record.tolist() == [1, 4, 6]
    assert audit["excluded"] == {"UTC_interpretations_straddle_pinned_cutoff": 3}


def test_row_explicit_utc_resolves_a_nominal_boundary_without_upgrading_other_rows(tmp_path):
    frame = observations(("2021-01-01T00:00:00Z",), explicit=True)
    out, _ = adapt_frame(frame, policy(tmp_path))
    assert len(out) == 1 and out.time_standard_ambiguity_seconds.iloc[0] == 0
    assert out.clock_axis_status.iloc[0] == "row_declared_UTC_accuracy_unknown"
    out2, _ = adapt_frame(frame, policy(tmp_path, mode="row_explicit_utc"))
    assert out2.timestamp.iloc[0] == frame.timestamp_utc.iloc[0]
    with pytest.raises(ValueError, match="requires explicit UTC"):
        adapt_frame(observations(), policy(tmp_path, mode="row_explicit_utc"))


def test_source_declared_utc_keeps_declaration_distinct_from_device_verification(tmp_path):
    frame = observations(("2021-01-01T00:00:00Z",))
    out, _ = adapt_frame(frame, policy(tmp_path, mode="declared_utc"))
    assert len(out) == 1 and out.timestamp_utc.isna().all()
    assert out.time_standard_status.iloc[0] == "UTC_or_GPS"
    assert out.clock_axis_status.iloc[0] == "source_declared_UTC_accuracy_unknown"
    decision = source_decision(tmp_path, mode="declared_utc")
    decision["evidence"][0]["kind"] = "movebank_attribute_utc_or_gps"
    with pytest.raises(ValueError, match="does not establish"):
        policy(tmp_path, sources={"release": decision})


def test_documented_gps_conversion_and_unrepresentable_leap_second_hold(tmp_path):
    frame = observations(("2016-05-01T12:00:00Z", "2017-01-01T00:00:16Z",
                          "2017-01-01T00:00:17.500Z", "2017-01-01T00:00:18Z"))
    frame["time_standard_status"] = "GPS"
    out, audit = adapt_frame(frame, policy(tmp_path, mode="documented_gps"))
    assert out.source_record.tolist() == [1, 2, 4]
    assert out.timestamp.tolist() == pd.to_datetime(["2016-05-01T11:59:43Z", "2016-12-31T23:59:59Z", "2017-01-01T00:00:00Z"], utc=True).tolist()
    assert out.time_standard_ambiguity_seconds.eq(0).all()
    assert audit["excluded"] == {"outside_verified_standard_table_or_UTC_leap_second": 1}


def test_unknown_timezone_missing_source_and_inconsistent_clocks_are_rejected(tmp_path):
    p = policy(tmp_path)
    for status in ["local_unknown", "UTC+unknown"]:
        frame = observations();frame["time_standard_status"] = status
        with pytest.raises(ValueError, match="Unbounded"):
            adapt_frame(frame, p)
    with pytest.raises(ValueError, match="Unreviewed source"):
        adapt_frame(observations(source="other"), p)
    frame = observations(explicit=True);frame.loc[0, "timestamp_utc"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="Conflicting"):
        adapt_frame(frame, p)
    frame = observations();frame["time_standard_status"] = "verified_UTC_or_explicit_offset"
    with pytest.raises(ValueError, match="status disagree"):
        adapt_frame(frame, p)
    decision = source_decision(tmp_path);decision["documented_clock"] = "local_unknown"
    with pytest.raises(ValueError, match="contradicts"):
        policy(tmp_path, sources={"release": decision})


def test_standard_table_validity_is_bounded_and_nanosecond_units_explicit(tmp_path):
    frame = observations(("1979-05-01T12:00:00Z", "2026-11-09T00:00:00Z", "2020-05-01T12:00:00Z"))
    frame["timestamp_nominal"] = frame.timestamp_nominal.dt.as_unit("us")
    out, audit = adapt_frame(frame, policy(tmp_path))
    assert out.source_record.tolist() == [3]
    assert out.time_standard_ambiguity_seconds.iloc[0] == 18
    assert sum(audit["excluded"].values()) == 2


def test_future_rows_and_locations_cannot_change_prior_clock_decisions(tmp_path):
    p = policy(tmp_path)
    frame = observations(("2020-05-01T00:00:00Z", "2020-05-02T00:00:00Z"))
    first, _ = adapt_frame(frame, p)
    future = observations(("2024-09-01T00:00:00Z",));future["lat"] = 88.;future["lon"] = -170.
    extended, _ = adapt_frame(pd.concat([frame, future], ignore_index=True), p)
    pd.testing.assert_frame_equal(first, extended.iloc[:2])


def test_evidence_and_policy_hash_guards(tmp_path):
    p = policy(tmp_path);validate_policy(json.loads(json.dumps(p)))
    changed = copy.deepcopy(p);changed["train_end_year"] = 2019
    with pytest.raises(ValueError, match="content hash"):
        validate_policy(changed)
    changed = copy.deepcopy(p);changed["standards_sha256"] = "0"*64
    changed["policy_sha256"] = fingerprint({k:v for k,v in changed.items() if k != "policy_sha256"})
    with pytest.raises(ValueError, match="standard table"):
        validate_policy(changed)
    Path(p["sources"]["release"]["evidence"][0]["path"]).write_text("changed source evidence")
    with pytest.raises(ValueError, match="evidence hash"):
        validate_policy(p)


def test_synthetic_export_is_immutable_and_preserves_original_qc(tmp_path):
    frame = observations(("2021-01-01T00:00:00Z", "2021-01-01T00:00:18Z", "2021-01-02T00:00:00Z"))
    manifest = clean_core(tmp_path, frame)
    before = sha256(manifest.parent / "observations.parquet")
    p = policy(tmp_path, manifests=[manifest]);pp = tmp_path / "policy.json";write_policy(pp, p)
    out = tmp_path / "compat"
    result = export(pp, out, batch_size=1)  # First batch entirely held.
    assert result["counts"] == {"input_rows": 3, "output_rows": 2}
    assert sha256(manifest.parent / "observations.parquet") == before
    converted = pd.read_parquet(out / "processed/observations.parquet")
    pd.testing.assert_frame_equal(converted[frame.columns].reset_index(drop=True), frame.iloc[1:].reset_index(drop=True))
    check_compatibility(out, train_end_year=2020, validation_year=2021, horizons=[1, 3, 90])
    with pytest.raises(ValueError, match="cutoffs differ"):
        check_compatibility(out, train_end_year=2021, validation_year=2022)
    with pytest.raises(ValueError, match="horizons are shorter"):
        check_compatibility(out, train_end_year=2020, validation_year=2021, horizons=[1/24, 1])
    with pytest.raises(ValueError, match="immutable"):
        export(pp, out)
    assert not list(tmp_path.glob("compat.pending-*"))


def test_source_observation_mutation_prevents_publication(tmp_path):
    manifest = clean_core(tmp_path, observations())
    p = policy(tmp_path, manifests=[manifest]);pp = tmp_path / "policy.json";write_policy(pp, p)
    observations(("2020-05-02T00:00:00Z",)).to_parquet(manifest.parent / "observations.parquet", index=False)
    with pytest.raises(ValueError, match="content hash mismatch"):
        export(pp, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_streamed_union_preserves_source_specific_columns(tmp_path):
    first = observations();second = observations(source="second");second["extra_diagnostic"] = "retained"
    manifests = [clean_core(tmp_path, first), clean_core(tmp_path, second, "qc2")]
    sources = {s: source_decision(tmp_path, source=s) for s in ["release", "second"]}
    p = policy(tmp_path, manifests=manifests, sources=sources);pp = tmp_path / "policy.json";write_policy(pp, p)
    export(pp, tmp_path / "union", batch_size=1)
    out = pd.read_parquet(tmp_path / "union/processed/observations.parquet")
    assert out.extra_diagnostic.isna().tolist() == [True, False]
    assert out.extra_diagnostic.iloc[1] == "retained"


@pytest.mark.parametrize("bad", ["2020-06-01 12:00:0S", "2020-06-01 12:00:00 XYZ",
                                "2020-06-01 12:00:00+25:00", "2020-02-30 12:00:00",
                                "2020-06-01 12:00:01", "06/01/2020 12:00"])
def test_reported_clock_syntax_and_nominal_agreement_are_not_inferred(tmp_path, bad):
    frame = observations();frame["timestamp_reported"] = bad
    out, audit = adapt_frame(frame, policy(tmp_path, mode="declared_utc"))
    assert out.empty and audit["excluded"] == {"malformed_or_conflicting_reported_timestamp": 1}
