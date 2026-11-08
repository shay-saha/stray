"""Retraction closure and inherited protections of the reviewed-source tool."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from test_migration_physical_repair import proof
from wildcast import multiscale_data


spec = importlib.util.spec_from_file_location(
    "reviewed_source_derivative", Path(__file__).parents[1] / "scripts/migration_reviewed_source_derivative.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def inputs(tmp_path, proof, monkeypatch, *, alternate_path=False, parent_affected=False):
    parent, base, _ = proof
    base = copy.deepcopy(base)
    names = ["goose:D84", "goose:D85"]
    base["physical_mapping"].update({names[0]: names[0], names[1]: names[0]})
    base["exclusion_mapping"].update({names[0]: names[0], names[1]: names[0]})
    edge = {"a": names[0], "b": names[1], "evidence_source": "reviewed_public_reference"}
    base["retained_physical_primitives"] = [dict(edge), {**edge, "evidence_source": "duplicate_reference_record"}]
    if alternate_path:
        base["physical_mapping"]["goose:D86"] = names[0]
        base["exclusion_mapping"]["goose:D86"] = names[0]
        base["retained_physical_primitives"] += [
            {"a": names[0], "b": "goose:D86", "evidence_source": "other_reference"},
            {"a": "goose:D86", "b": names[1], "evidence_source": "other_reference"}]
    if parent_affected:
        base["parent_memberships"][0]["raw_ids"].append(names[1])
        base["parent_memberships"][0]["repaired_physical_components"].append(names[0])
    source_sha = "f" * 64
    base["raw_coordinate_ownership"] = [
        {"input_animal": name, "input_sha256": source_sha, "status": "unmixed_original_source_id",
         "original_raw_ids": [name], "repaired_physical_ids": [names[0]]} for name in names]
    for key in ["affected_parent_memberships", "parent_role_conflicts", "parent_freshness_conflicts"]:
        base[key] = []
    base["summary"] = {}
    combined = tmp_path / "combined.json"
    combined.write_text(json.dumps({"uncertainty_fold_groups": base["exclusion_mapping"]}))
    base["combined_identity_sha256"] = tool.digest(combined)
    base["inputs"] = {"combined": {"path": str(combined), "sha256": tool.digest(combined)}}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base))
    bird = tmp_path / "goose.json"
    bird.write_text(json.dumps({"complete": True, "repair": {"sha256": tool.digest(path)},
        "retracted_edges": [{"a": names[0], "b": names[1], "reason": "conflicting simultaneous original GPS"}]}))
    numeric = tmp_path / "numeric.json"
    numeric.write_text(json.dumps({"complete": True, "repair": {"sha256": tool.digest(path)},
        "summary": {"coordinate_values_modified": 0, "conflicting_nominal_timestamps": 8},
        "frozen_parent": {"affected_parent_animals": []},
        "record_preferences": [{"original_source_sha256": "e" * 64, "preferred_original_record": i * 2 + 1,
                                "suppressed_original_record": i * 2 + 2} for i in range(8)],
        "proposed_source_record_exclusions": [{"input_sha256": source_sha, "input_rows": 16,
            "row_positions": list(range(1, 16, 2)), "row_number_basis": "zero_based_full_input_order",
            "reason": "keep specifically reviewed native records unchanged"}]}))
    monkeypatch.setattr(multiscale_data, "_load_parent", lambda *args: parent)
    return path, bird, numeric, base


def test_derivative_removes_both_reference_primitives_but_preserves_risk_parent_and_bytes(tmp_path, proof, monkeypatch):
    path, bird, numeric, previous = inputs(tmp_path, proof, monkeypatch)
    original = path.read_bytes()
    output = tmp_path / "derived.json"
    result = tool.derive(path, bird, numeric, tmp_path, output)
    assert path.read_bytes() == original
    assert result["physical_mapping"]["goose:D84"] == "goose:D84"
    assert result["physical_mapping"]["goose:D85"] == "goose:D85"
    assert result["exclusion_mapping"] == previous["exclusion_mapping"]
    assert result["parent_memberships"] == previous["parent_memberships"]
    assert not result["retained_physical_primitives"]
    assert len(result["retract_edges"]) == len(previous["retract_edges"]) + 1
    selector = result["source_record_exclusions"][0]
    assert selector["row_positions"] == list(range(1, 16, 2))
    assert selector["evidence_sha256"] == tool.digest(numeric)
    assert result["raw_coordinate_ownership"][1]["repaired_physical_ids"] == ["goose:D85"]
    with pytest.raises(FileExistsError):
        tool.derive(path, bird, numeric, tmp_path, output)


@pytest.mark.parametrize("change", ["alternate_path", "parent_affected"])
def test_unreviewed_transitive_join_or_parent_ancestry_aborts_before_publication(tmp_path, proof, monkeypatch, change):
    path, bird, numeric, _ = inputs(tmp_path, proof, monkeypatch, **{change: True})
    output = tmp_path / "must-not-exist.json"
    message = "another physical identity" if change == "alternate_path" else "parent membership"
    with pytest.raises(ValueError, match=message):
        tool.derive(path, bird, numeric, tmp_path, output)
    assert not output.exists()
