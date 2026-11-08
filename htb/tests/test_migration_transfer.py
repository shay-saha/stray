import hashlib
import json

import numpy as np
import pandas as pd

from wildcast.migration_transfer import prepare


def test_population_holdout_removes_alias_group_and_keeps_final_truth_sealed(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    n, h = 5, 2
    mask = np.ones((n, h), bool); mask[-1] = False
    origin = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 1577836800.)])
    a = {"history": np.zeros((n, 2, 10), np.float32), "history_mask": np.ones((n, 2), bool),
         "static": np.zeros((n, 14), np.float32), "species": np.ones(n, np.int64),
         "metric_species": np.zeros(n, np.int64), "y": np.zeros((n, h, 2), np.float32),
         "target_mask": mask, "split": np.array([0, 0, 1, 2, 3], np.int8),
         "individual": np.array([4, 5, 6, 4, 7]), "study": np.zeros(n, np.int64),
         "origin": origin, "target_latlon": np.zeros((n, h, 2)),
         "target_time": np.where(mask, origin[:, 2:3] + np.array([1, 3]) * 86400., 0),
         "horizons": np.array([1, 3], np.float32)}
    np.savez_compressed(base / "cases.npz", **a)
    origins = pd.DataFrame({"case_id": np.arange(n), "species": ["bird"] * n,
                            "canonical_animal": ["other:study:bird:alias", "keep", "validation", "other:study:bird:alias", "reserved"]})
    origins.to_parquet(base / "origins.parquet")
    file_hash = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = file_hash(base / "cases.npz")
    manifest = {"cases_sha256": before, "origins_sha256": file_hash(base / "origins.parquet"),
                "metric_species": {"bird": 0}, "reserved_test_targets_sealed": True,
                "identity_mapping": {"held:study:bird:original": "other:study:bird:alias", "other:study:bird:alias": "other:study:bird:alias"},
                "animals": {"other:study:bird:alias": 4, "keep": 5, "validation": 6, "reserved": 7}}
    (base / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "transfer"
    result = prepare(base, output, ["held"])
    with np.load(output / "cases.npz") as derived:
        assert derived["split"].tolist() == [1, 0, 1, 1, 3]
        assert not derived["target_mask"][2].any()  # unrelated old validation
        assert not derived["target_mask"][-1].any()
        assert derived["target_mask"][[0, 3]].all()
    assert result["scored_population_cases"] == 2
    assert not json.loads((output / "manifest.json").read_text())["eligible_for_final_assessment"]
    assert before == file_hash(base / "cases.npz")
