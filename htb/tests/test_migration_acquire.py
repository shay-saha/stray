"""Acquisition boundaries must hold even for malformed public archives."""
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("migration_acquire", Path(__file__).parents[1] / "scripts/migration_acquire.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("name", ["../events.csv", "/etc/events.csv", "x/../../events.csv", "C:\\evil.csv"])
def test_traversal_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        m.safe_name(name)


def test_contradictory_rights_quarantined():
    assert m.rights_status({"dc.rights.uri": ["https://creativecommons.org/publicdomain/zero/1.0/"],
                            "dc.rights": ["Non-commercial use only"]}).startswith("quarantined")
    assert m.file_role({"name": "storks-acc-1of8.csv.zip"}) == "non_gps"


HEADER = "timestamp,location-lat,location-long,individual-local-identifier,individual-taxon-canonical-name,sensor-type\n"


def test_chunking_preserves_sensor_labels_and_quoted_values(tmp_path):
    text = HEADER + '2020-01-01,1,2,"animal, one",Ciconia ciconia,gps\n2020-01-02,3,4,bird,Ciconia ciconia,argos\n'
    result = m.split_csv(io.StringIO(text), tmp_path, "events", m.Budget(10000), 180, ["GPS", "Argos Doppler Shift"], "events.csv")
    rows = []
    import csv
    for c in result["chunks"]:
        with (tmp_path / c["name"]).open() as f:
            rows.extend(csv.DictReader(f))
    assert [r["sensor-type"] for r in rows] == ["gps", "argos"]
    assert rows[0]["individual-local-identifier"] == "animal, one"
    assert len(rows) == 2


def test_mixed_sensors_without_row_labels_are_quarantined(tmp_path):
    text = HEADER.replace(",sensor-type", "") + "2020-01-01,1,2,bird,Ciconia ciconia\n"
    r = m.split_csv(io.StringIO(text), tmp_path, "events", m.Budget(10000), 1000,
                    ["GPS", "Argos Doppler Shift"], "events.csv")
    assert r["status"].startswith("quarantined")
    assert not list(tmp_path.glob("*.csv"))


def test_quota_failure_leaves_no_partial_cleanable_csv(tmp_path):
    budget = m.Budget(140)
    with pytest.raises(ValueError, match="budget"):
        m.split_csv(io.StringIO(HEADER + "2020-01-01,1,2,bird,Ciconia ciconia,gps\n"),
                    tmp_path, "events", budget, 1000, ["GPS"], "events.csv")
    assert budget.reserved == 0
    assert list(tmp_path.iterdir()) == []


def test_archive_bomb_and_traversal_rejected(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../events.csv", HEADER)
    with pytest.raises(ValueError, match="Unsafe"):
        m.extract_csvs(path, tmp_path, {"name": path.name}, ["GPS"], m.Budget(10000), 1000, 10000)
    with pytest.raises(ValueError, match="uncompressed quota"):
        m.extract_csvs(path, tmp_path, {"name": path.name}, ["GPS"], m.Budget(10000), 1000, 1)


def test_cp1252_is_recorded_and_utf8_chunks_preserve_text(tmp_path):
    path = tmp_path / "originals" / "events.csv"
    path.parent.mkdir()
    path.write_bytes((HEADER + '2020-01-01,1,2,"bird “one”",Ciconia ciconia,gps\n').encode("cp1252"))
    result = m.extract_csvs(path, tmp_path, {"name": path.name}, ["GPS"], m.Budget(10000), 1000, 10000)
    assert result[0]["source_encoding"] == "cp1252"
    assert "bird “one”" in (tmp_path / result[0]["chunks"][0]["name"]).read_text()


def test_reference_links_use_ring_corroboration_and_reject_hardware_reuse(tmp_path):
    import csv
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from wildcast.migration_identity import audit

    rows = []
    root = tmp_path / "raw"
    for source, local, ring, start in [("one", "birdA", "DEW 8N559", "2020-01-01"),
                                       ("two", "birdB", "DEW8N559", "2020-01-01"),
                                       ("three", "birdC", "DEW9N999", "2021-01-01")]:
        d = root / source
        d.mkdir(parents=True)
        ref = d / "reference.csv"
        row = {"animal-id": local, "animal-taxon": "Ciconia ciconia", "animal-ring-id": ring,
               "tag-manufacturer-name": "Manufacturer", "tag-serial-no": "12345678",
               "deploy-on-date": start, "study-site": "Same breeding colony", "animal-sex": "f"}
        with ref.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (d / "source.json").write_text(json.dumps({"files": [{"name": ref.name, "sha256": m.digest(ref)}]}))
        rows.append({"source": source, "species": "Ciconia ciconia", "animal": f"{source}:study:Ciconia ciconia:{local}"})
    obs = tmp_path / "observations.parquet"
    pq.write_table(pa.Table.from_pylist(rows), obs)
    result = audit([obs], [root], tmp_path / "audit.json")
    assert any({l["a"], l["b"]} == {rows[0]["animal"], rows[1]["animal"]} for l in result["links"])
    assert not any(rows[2]["animal"] in (l["a"], l["b"]) for l in result["links"])
    assert any(a["type"].startswith("possible_hardware_redeployment") for a in result["ambiguous"])


def test_reference_row_respects_event_study_namespace():
    from wildcast.migration_identity import scoped_animals
    names = ["doi:White Stork Adults 2017:Ciconia ciconia:007", "doi:White Stork Adults 2018:Ciconia ciconia:007"]
    assert scoped_animals(names, "doi", "Ciconia ciconia", "White Stork Adults 2017_part-reference-data.csv") == names[:1]
    assert scoped_animals(names, "doi", "Ciconia ciconia", "generic-reference-data.csv") == []


def test_printed_qualified_ring_links_without_tag_only_inference(tmp_path):
    import csv
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from wildcast.migration_identity import audit
    root = tmp_path / "raw"
    d = root / "one"
    d.mkdir(parents=True)
    ref = d / "reference.csv"
    row = {"animal-id": "Old", "animal-taxon": "Ciconia ciconia", "animal-ring-id": "DER AT881",
           "tag-manufacturer-name": "e-obs GmbH", "tag-id": "3946"}
    with ref.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader(); w.writerow(row)
    (d / "source.json").write_text(json.dumps({"files": [{"name": ref.name, "sha256": m.digest(ref)}]}))
    names = ["one:study:Ciconia ciconia:Old", "two:study:Ciconia ciconia:Aldo + / DER AT881 (eobs 3946)",
             "three:study:Ciconia ciconia:Other / DER AT882 (eobs 3946)"]
    obs = tmp_path / "observations.parquet"
    pq.write_table(pa.Table.from_pylist([{"animal": a, "source": a.split(":")[0], "species": "Ciconia ciconia"} for a in names]), obs)
    result = audit([obs], [root], tmp_path / "audit.json")
    assert any({l["a"], l["b"]} == set(names[:2]) for l in result["links"])
    assert not any(names[2] in (l["a"], l["b"]) for l in result["links"])


def test_resumed_reference_entries_remain_in_final_source_manifest(tmp_path, monkeypatch):
    import json
    out = tmp_path / "output"
    d = out / "raw" / "source"
    originals = d / "originals"
    originals.mkdir(parents=True)
    ref = originals / "birds-reference-data.csv"
    ref.write_text("animal-id,animal-taxon\nbird,Ciconia ciconia\n")
    previous = {"name": ref.name, "stored_as": "originals/" + ref.name,
                "status": "complete", "sha256": m.digest(ref)}
    (d / "source.json").write_text(json.dumps({"files": [previous]}))
    def fake_download(s, url, path, *args):
        path.write_text("event-id\n1\n")
        return path
    monkeypatch.setattr(m, "download", fake_download)
    monkeypatch.setattr(m, "extract_csvs", lambda *args: [{"chunks": []}])
    rec = {"id": "source", "metadata": {}}
    item = {"files": [{"name": "events.csv", "bytes": 11, "url": "https://example.invalid/events"},
                      {"name": ref.name, "bytes": ref.stat().st_size, "url": "https://example.invalid/ref"}],
            "sensors_metadata": ["GPS"], "reasons": []}
    m.acquire(rec, item, out, {}, m.Budget(10000), 1000, 1000, 1000)
    saved = json.loads((d / "source.json").read_text())
    assert [f["name"] for f in saved["files"]] == ["events.csv", ref.name]
    assert saved["acquisition_complete"] is True
