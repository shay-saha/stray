"""Conservative, content-bound review of the two generic Raja Ampat ray releases.

Keep all physical identities and raw labels unchanged. Block a possible shared
animal/research-lineage alias from crossing folds while its ID remains unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wildcast.breadth_identity import Groups
from wildcast.marine_intake import sha256


SOURCES = ["f8c74489-c4ff-46fb-8771-5eebc5b4e8b6", "fc898722-d930-4f48-8c4b-8505cbb2b627"]
PAPER = "https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.815094/full"


def run(identity, observations, output, review_output):
    identity, observations, output, review_output = map(Path, [identity, observations, output, review_output])
    if output.exists() or review_output.exists():
        raise ValueError("Ray review outputs are immutable")
    report = json.loads(identity.read_text())
    checksum = sha256(observations)
    if not any(Path(v["path"]) == observations and v["sha256"] == checksum for v in report["observations"]):
        raise ValueError("Ray review observations differ from combined identity inputs")
    data = pq.read_table(observations, columns=["source", "animal", "species", "timestamp", "lat", "lon"],
                         filters=[("source", "in", SOURCES)]).to_pandas()
    old = data.loc[data.source.eq(SOURCES[0])]
    wayag = data.loc[data.source.eq(SOURCES[1])]
    if old.animal.nunique() != 1 or wayag.animal.nunique() != 5:
        raise ValueError("Actual ray release identity counts differ from the reviewed source scope")
    overlap = old.merge(wayag, on=["timestamp", "lat", "lon"], suffixes=("_old", "_wayag"))
    animals = sorted(data.animal.unique())
    if len(overlap):
        raise ValueError("Ray releases share exact GPS; review physical evidence rather than only adding risk links")
    selected = [v for v in report["new_animal_parent_bindings"] if v["animal"] in animals]
    if len(selected) != len(animals) or any(v["parent_members"] for v in selected):
        raise ValueError("Ray review intersects parent roles or lacks combined identity coverage")
    review = {"schema": "wildcast-ray-source-review-v1", "combined_identity": str(identity),
              "combined_identity_sha256": sha256(identity), "observations": str(observations), "observations_sha256": checksum,
              "source_ids": SOURCES, "animals": animals, "exact_shared_time_GPS_rows": len(overlap),
              "release_support": [{"source_id": source, "raw_labels": sorted(group.species.unique()), "rows": len(group),
                                   "source_animal_ids": group.animal.nunique(), "first": group.timestamp.min().isoformat(),
                                   "last": group.timestamp.max().isoformat()} for source, group in data.groupby("source")],
              "primary_evidence": [{"url": PAPER,
                                    "finding": "Wayag study identifies five juvenile Mobula alfredi carrying Fastloc GPS satellite tags; acoustic detections and six-hour SSM estimates are separate analyses. The paper reports1199 GPS records before15 analytical outlier removals; this is not a reproduction of that manual filter."},
                                   {"url": "https://doi.org/10.1002/eap.1704",
                                    "finding": "Earlier home-range example includes a generic reef-manta release from the same research lineage; no usable individual cross-release correspondence was established."}],
              "action": "One uncertain exclusion group, no physical identity assertion, no coordinate merge, no raw label modification",
              "species_resolution": {"source_id": SOURCES[1], "raw_label": "Mobula", "supported_label": "Mobula alfredi",
                                     "status": "documented source-specific candidate mapping, not applied to any frozen QC/clock table or named count"},
              "limitations": ["Zero exact GPS overlap does not exclude a disjoint deployment alias",
                              "The generic Manta label is not globally assigned to Mobula alfredi",
                              "No assumed metre precision, LC-to-noise conversion, SSM or receiver location enters GPS truth",
                              "Raw ray GPS retains reported observation error; manual publication filtering has not been reproduced"],
              "implementation_sha256": sha256(__file__)}
    review_output.parent.mkdir(parents=True, exist_ok=True)
    review_output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    risk = Groups(report["uncertainty_fold_groups"])
    for a, b in report["uncertainty_fold_groups"].items():
        risk.join(a, b)
    for a, b in zip(animals, animals[1:]):
        risk.join(a, b)
    component = sorted(name for name in risk.parent if risk.find(name) == risk.find(animals[0]))
    members = [v for v in report["new_animal_parent_bindings"] if v["animal"] in component]
    if any(v["parent_members"] for v in members):
        raise ValueError("Uncertain component would cross parent roles; no automatic override")
    report["uncertainty_fold_groups"] = risk.mapping()
    report["ambiguous"].append({"type": "possible_ray_release_lineage_alias_no_physical_merge",
                                "animals": component, "review": str(review_output), "review_sha256": sha256(review_output)})
    report["freshness_exclusion_animals"] = sorted(set(report["freshness_exclusion_animals"]) | set(component))
    for binding in report["new_animal_parent_bindings"]:
        binding["exclusion_group"] = risk.find(binding["animal"])
        if binding["animal"] in component:
            binding["status"] = "unresolved_alias_not_certified_fresh"
    report["summary"]["new_exclusion_groups"] = len({v["exclusion_group"] for v in report["new_animal_parent_bindings"]})
    report["summary"]["new_binding_status"] = dict(Counter(v["status"] for v in report["new_animal_parent_bindings"]))
    report["identity_review_derivation"] = {"base": str(identity), "base_sha256": sha256(identity),
                                            "review": str(review_output), "review_sha256": sha256(review_output),
                                            "implementation_sha256": sha256(__file__),
                                            "physical_groups_and_named_taxon_count_unchanged": True}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity", required=True); p.add_argument("--observations", required=True)
    p.add_argument("--output", required=True); p.add_argument("--review-output", required=True)
    a = p.parse_args(); run(a.identity, a.observations, a.output, a.review_output)
