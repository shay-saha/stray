"""Development annotation joins must never consume reserved outcome fields."""
import csv
import json

import openpyxl
import pandas as pd
import pytest

from wildcast.migration_annotations import WYOMING, _auxiliary_annotation_coverage, _reference_coverage, load_development_timing, sha256


@pytest.fixture
def annotation_fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    origins = pd.DataFrame({"case_id": [0, 1, 2],
        "canonical_animal": [f"{WYOMING}:study:Odocoileus hemionus:{i}" for i in [1, 2, 3]],
        "year": [2019] * 3, "split": ["train", "val_individual", "test_individual"]})
    origins.to_parquet(data / "origins.parquet", index=False)
    (data / "manifest.json").write_text(json.dumps({"origins_sha256": sha256(data / "origins.parquet")}))
    workbook = tmp_path / "annotations.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["id_yr", "year", "DOY_Start", "DOY_End"])
    sheet.append(["1_2019", 2019, 60, 90])
    sheet.append(["2_2019", 2019, 70, 95])
    sheet.append(["3_2019", 2019, "RESERVED OUTCOME MUST NOT BE PARSED", "SEALED"])
    sheet.append(["4_2010", 2010, "UNMATCHED MUST NOT BE PARSED", "UNKNOWN"])
    book.save(workbook)
    (tmp_path / "source.json").write_text(json.dumps({"files": [{"path": str(workbook), "sha256": sha256(workbook)}]}))
    return data, workbook


def test_only_development_animal_year_outcomes_are_parsed(annotation_fixture):
    data, workbook = annotation_fixture
    result, audit = load_development_timing(data, workbook)
    assert list(result.split) == ["train", "val_individual"]
    assert list(result.departure_doy) == [60, 70]
    assert audit["counts"]["reserved_selectors_matched_outcomes_not_inspected"] == 1
    assert audit["counts"]["unmatched_animal_years"] == 1


def test_workbook_source_checksum_is_required(annotation_fixture):
    data, workbook = annotation_fixture
    with workbook.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_development_timing(data, workbook)


def test_trait_knowledge_is_anchored_to_deployment_and_excludes_test(tmp_path):
    path = tmp_path / "reference.csv"
    row = {"animal-id": "bird", "animal-sex": "f", "animal-life-stage": "adult",
           "animal-group-id": "group", "study-site": "field station", "deploy-on-date": "2020-01-01"}
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    reference = {"references_checked": [{"path": str(path), "sha256": sha256(path)}],
                 "reference_records": [{"record_type": "reference_table", "reference": str(path),
                                        "line": 2, "observation_animals": ["raw-bird"]}]}
    origins = pd.DataFrame({"canonical_animal": ["bird"] * 3, "case_id": [0, 1, 2],
                            "year": [2019, 2021, 2022], "origin_time": pd.to_datetime(["2019-01-01", "2021-01-01", "2022-01-01"], utc=True),
                            "split": ["train", "val_year", "test_year"]})
    result = _reference_coverage(origins, {"identity_mapping": {"raw-bird": "bird"}}, reference)
    sex = result["sex"]
    assert sex["matched_development_animals"] == {"train": 1, "val_year": 1}
    assert sex["deployment_date_anchored_development_cases"].get("train", 0) == 0
    assert sex["deployment_date_anchored_development_cases"]["val_year"] == 1
    assert "test_year" not in sex["matched_development_animals"]


def test_auxiliary_statistics_do_not_include_reserved_animal_years(annotation_fixture):
    data, workbook = annotation_fixture
    development, _ = load_development_timing(data, workbook)
    book = openpyxl.load_workbook(workbook)
    sheet = book.create_sheet("SFig1")
    sheet.append(["id_yr", "scaledIFBFat"])
    sheet.append(["1_2019", 1.])
    sheet.append(["2_2019", 2.])
    sheet.append(["3_2019", 99999.])
    book.save(workbook)
    result = _auxiliary_annotation_coverage(workbook, development)
    coverage = result["author_body_condition"]["development_coverage_by_split"]
    assert coverage == {"train": {"animal_years": 1, "animals": 1},
                        "val_individual": {"animal_years": 1, "animals": 1}}
