"""Assemble an immutable physical-map repair proof while preserving exclusion roles."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class Groups:
    def __init__(self, names): self.parent = {n: n for n in names}
    def find(self, name):
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name
    def join(self, a, b):
        if a not in self.parent or b not in self.parent:
            raise ValueError("Audited primitive refers to an unknown raw ID")
        a, b = self.find(a), self.find(b)
        self.parent[max(a,b)] = min(a,b)
    def mapping(self): return {n: self.find(n) for n in sorted(self.parent)}


def assemble(parent, combined, review, lineage, legacy_review, recovery, quarantine, output):
    paths = {k: Path(v) for k,v in locals().copy().items() if k not in {"output"}}
    p, c, r, raw, old = [json.loads(paths[k].read_text()) for k in ["parent","combined","review","lineage","legacy_review"]]
    recovered = json.loads(paths["recovery"].read_text())
    nested=recovered.get("original_recovery")
    if isinstance(nested,dict) and "outputs" in nested:recovered=nested
    if recovered.get("complete") is not True or not recovered.get("outputs"):
        raise ValueError("A complete content-bound original recovery output is required")
    quarantine_rules=json.loads(paths["quarantine"].read_text())
    if (quarantine_rules["identity_audit"]["sha256"]!=sha(paths["combined"]) or
            quarantine_rules["source_QC_evidence"]["sha256"]!=sha(paths["legacy_review"])):
        raise ValueError("Global source-interval quarantine evidence bindings differ")
    if sha(paths["parent"]) != c["parent_manifest"]["sha256"] or p["cases_sha256"] != c["parent_cases_sha256"]:
        raise ValueError("Frozen parent bindings differ")
    if r["parent_manifest"]["sha256"] != sha(paths["parent"]) or raw["coordinate_review"]["sha256"] != sha(paths["review"]):
        raise ValueError("Coordinate review/lineage bindings differ")
    directory = paths["parent"].parent
    if sha(directory / "cases.npz") != p["cases_sha256"] or sha(directory / "origins.parquet") != p["origins_sha256"]:
        raise ValueError("Frozen sealed parent files differ")
    if p.get("reserved_test_targets_sealed") is not True:
        raise ValueError("Parent is not sealed")
    if any(s["QC_unmatched_rows"] for s in raw["sources"]):
        raise ValueError("Original coordinate/time lineage is incomplete")
    # Legacy audit must explicitly classify all older46 and new5 edges before a
    # final contract can be assembled. Never infer safe evidence from a star.
    retained_extra, retracted_extra = old["retained_edges"], old["retracted_edges"]
    old_quality = Path(old["quality_report"]["path"])
    if sha(old_quality)!=old["quality_report"]["sha256"]:
        raise ValueError("Older alias primitive quality-report checksum differs")
    paths["older_quality_report"]=old_quality
    older_edges = json.loads(old_quality.read_text())["alias_links"]
    edge_key = lambda e: tuple(sorted([e["a"],e["b"]]))
    expected = {edge_key(e) for e in [*older_edges,*c["exact_GPS_links"]]}
    classified = [edge_key(e) for e in [*retained_extra,*retracted_extra]]
    if len(classified) != len(set(classified)) or set(classified) != expected:
        raise ValueError("Legacy review must classify exactly46 older and5 new primitive edges")
    extra_names = {n for edge in older_edges for n in [edge["a"],edge["b"]]}
    names = set(c["fold_groups"]) | extra_names
    physical, risk = Groups(names), Groups(names)
    safe, retract = [], []
    for edge in p["identity_links"]:
        if edge["evidence"] != "exact_species_time_GPS_overlap":
            safe.append({**edge,"evidence_source":"parent_documented_reference_identity"})
    for edge in r["pair_reviews"]:
        item = {"a":edge["a"],"b":edge["b"],"original_evidence":edge["parent_link"],
                "pair_review_sha256":sha(paths["review"]),"raw_lineage_sha256":sha(paths["lineage"]),
                "simultaneous_fixes":edge["simultaneous_fix_pairs"],
                "exact_simultaneous_fixes":edge["exact_same_time_position_pairs"]}
        complete_duplicate = (edge["all_simultaneous_positions_identical"]
                              and edge["a_fixes"] == edge["b_fixes"] == edge["simultaneous_fix_pairs"]
                              and edge["a_start"] == edge["b_start"] and edge["a_end"] == edge["b_end"])
        if complete_duplicate:
            safe.append({**item,"type":"complete_observed_track_duplicate",
                         "evidence_source":"original_verified_coordinate_overlap",
                         "limitation":"Both entire retained tracks are identical. This supports coordinate-history deduplication; inconsistent biological metadata is retained and masked, not silently reconciled"})
        else:
            retract.append({**item,"type":"retract_coordinate_only_physical_edge",
                            "reason":("A duplicate overlapping deployment stream does not establish whole-animal identity or authorize transferring other-year personal history"
                                      if edge["all_simultaneous_positions_identical"] else
                                      "Partial exact coincidences do not establish physical identity in the presence of simultaneous contradictory coordinates")})
    public_path = Path(c["public_reference_audit"]["path"])
    if sha(public_path) != c["public_reference_audit"]["sha256"]:
        raise ValueError("Reference primitive evidence hash differs")
    for edge in json.loads(public_path.read_text())["links"]:
        safe.append({**edge,"evidence_source":"public_reference_documented_primitive",
                     "source_audit_sha256":sha(public_path)})
    for edge in c["metadata_physical_links"]:
        safe.append({**edge,"evidence_source":"combined_documented_metadata_primitive",
                     "source_audit_sha256":sha(paths["combined"])})
    safe.extend({**edge,"evidence_source":"legacy_and_manx_independent_review",
                 "source_audit_sha256":sha(paths["legacy_review"])} for edge in retained_extra)
    retract.extend({**edge,"source_audit_sha256":sha(paths["legacy_review"])} for edge in retracted_extra)
    for edge in safe: physical.join(edge["a"],edge["b"])
    for a,b in c["uncertainty_fold_groups"].items(): risk.join(a,b)
    for edge in older_edges: risk.join(edge["a"],edge["b"])
    for edge in safe: risk.join(edge["a"],edge["b"])
    risk_components = collections.defaultdict(list)
    for n,g in risk.mapping().items():risk_components[g].append(n)
    exclusion = {}
    for members in risk_components.values():
        anchors = {c["uncertainty_fold_groups"][n] for n in members if n in c["uncertainty_fold_groups"]}
        if len(anchors)>1:
            raise ValueError("Repair would merge previously separate frozen exclusion components; protocol review required")
        anchor = next(iter(anchors)) if anchors else min(members)
        exclusion.update({n:anchor for n in members})
    if any(exclusion[n] != g for n,g in c["uncertainty_fold_groups"].items()):
        raise ValueError("Existing conservative exclusion IDs changed")
    mapping = physical.mapping()
    for edge in retract:
        if mapping[edge["a"]] == mapping[edge["b"]]:
            raise ValueError("A retracted physical edge was reintroduced by an alternate primitive path")
    components = collections.defaultdict(list)
    for n,g in mapping.items():components[g].append(n)
    if any(len({exclusion[n] for n in members})>1 for members in components.values()):
        raise ValueError("A physical component crosses exclusion roles")
    origins = pd.read_parquet(directory / "origins.parquet",columns=["canonical_animal","fresh_identity"])
    freshness = {n:sorted(set(g.fresh_identity.tolist())) for n,g in origins.groupby("canonical_animal",sort=False)}
    parent_members = collections.defaultdict(list)
    for n,g in p["identity_mapping"].items():parent_members[g].append(n)
    collapsed = Groups(extra_names)
    for edge in older_edges:collapsed.join(edge["a"],edge["b"])
    old_components = collections.defaultdict(list)
    for n,g in collapsed.mapping().items():old_components[g].append(n)
    collapsed_members = {n:sorted(old_components[collapsed.find(n)]) for n in extra_names}
    memberships = []
    for group,members in sorted(parent_members.items()):
        recovered_aliases = sorted({a for n in members for a in collapsed_members.get(n,[])})
        all_members=sorted(set(members)|set(recovered_aliases))
        repaired = sorted({mapping[n] for n in all_members})
        memberships.append({"parent_canonical":group,"raw_ids":sorted(members),
                            "raw_ids_including_recovered":all_members,
                            "repaired_physical_components":repaired,"physical_partition_changed":len(repaired)>1,
                            "legacy_collapsed_raw_ids":recovered_aliases,
                            "requires_original_trajectory_recovery":bool(recovered_aliases),
                            "original_parent_exclusion_group":p.get("exclusion_groups",{}).get(group,group),
                            "child_exclusion_groups":sorted({exclusion[n] for n in all_members}),
                            "parent_role":p["assignments"][group],
                            "parent_origin_freshness_values":freshness.get(group,[])})
    roles = collections.defaultdict(set)
    for row in memberships:
        for group in row["child_exclusion_groups"]:roles[group].add(row["parent_role"])
    role_conflicts = {g:sorted(v) for g,v in roles.items() if len(v)>1}
    if role_conflicts:raise ValueError("Parent roles conflict after risk extension")
    legacy_parent_groups={p["identity_mapping"][n] for n in extra_names if n in p["identity_mapping"]}
    freshness_conflicts={g:freshness.get(g,[]) for g in legacy_parent_groups if True in freshness.get(g,[])}
    if freshness_conflicts:
        raise ValueError("Legacy freshness exclusion would change a frozen parent freshness declaration")
    # Original-source ownership is measured per input, never inferred from an
    # identifier in a compressed union graph. Older collapsed components are
    # explicitly unavailable as unmixed rows until reconstructed originals exist.
    ownership = []
    for item in c["observations"]:
        path = Path(item["path"])
        if sha(path) != item["sha256"]:raise ValueError("Observation input hash mismatch")
        counts = collections.Counter()
        for batch in pq.ParquetFile(path).iter_batches(columns=["animal"],batch_size=250_000,use_threads=False):
            counts.update(batch.column(0).to_pandas().value_counts().to_dict())
        for animal,count in sorted(counts.items()):
            is_collapsed = str(path)=="data/combined-landscape/processed/observations.parquet" and animal in collapsed_members
            ownership.append({"input_path":str(path),"input_sha256":item["sha256"],"input_animal":animal,"rows":count,
                              "status":"already_collapsed_needs_original_recovery" if is_collapsed else "unmixed_original_source_id",
                              "original_raw_ids":collapsed_members[animal] if is_collapsed else [animal],
                              "row_lineage_evidence":"pre-v3 quality-report alias graph" if is_collapsed else "source-preserving QC input and frozen observation hash",
                              "repaired_physical_ids":sorted({mapping[n] for n in collapsed_members[animal]}) if is_collapsed else [mapping[animal]]})
    required_replacements={(row["input_sha256"],row["input_animal"]) for row in ownership
                           if row["status"]!="unmixed_original_source_id"}
    replaced=set()
    recovered_ids=set()
    for record in recovered["outputs"]:
        path=Path(record["path"])
        if sha(path)!=record["sha256"] or not record.get("source_QC_provenance"):
            raise ValueError("Recovered output checksum/source-QC provenance differs")
        columns=pq.ParquetFile(path).schema_arrow.names
        row_id_column=record.get("row_id_column",recovered.get("row_id_column"))
        if not row_id_column or row_id_column not in columns:
            raise ValueError("Original row identity must survive recovered output")
        recovered_rows=pq.read_table(path,columns=["animal",row_id_column]).to_pandas()
        if recovered_rows[row_id_column].isna().any() or recovered_rows[row_id_column].duplicated().any():
            raise ValueError("Recovered original row IDs must be present and unique")
        counts=recovered_rows.animal.value_counts().to_dict()
        if set(counts)!=set(record["raw_ids"]) or set(counts)-mapping.keys():
            raise ValueError("Recovered raw-ID inventory differs from actual output or reviewed mapping")
        recovered_ids.update(counts)
        for previous in record.get("replaces",[]):
            key=previous["input_sha256"],previous["input_animal"]
            if key in replaced:raise ValueError("A collapsed input selector is replaced twice")
            replaced.add(key)
        for animal,count in sorted(counts.items()):
            ownership.append({"input_path":str(path),"input_sha256":record["sha256"],"input_animal":animal,"rows":count,
                              "status":"unmixed_original_source_id","original_raw_ids":[animal],
                              "repaired_physical_ids":[mapping[animal]],"recovered_from_publisher_original":True,
                              "row_id_column":row_id_column,"row_lineage_evidence":record["source_QC_provenance"]})
    if replaced!=required_replacements or not extra_names<=recovered_ids:
        raise ValueError("Recovery must replace exactly every collapsed selector and restore all72 source IDs")
    result = {"schema":"wildcast-parent-physical-retraction-v1","code_sha256":sha(__file__),
              "inputs":{k:{"path":str(v),"sha256":sha(v)} for k,v in paths.items()},
              "parent_manifest_sha256":sha(paths["parent"]),"parent_cases_sha256":p["cases_sha256"],
              "parent_origins_sha256":p["origins_sha256"],"combined_identity_sha256":sha(paths["combined"]),
              "physical_mapping":mapping,"exclusion_mapping":exclusion,
              "retained_physical_primitives":safe,"retract_edges":retract,
              "parent_memberships":memberships,"affected_parent_memberships":[r for r in memberships if r["physical_partition_changed"] or r["requires_original_trajectory_recovery"]],
              "raw_coordinate_ownership":ownership,"original_recovery":{**recovered,
                    "manifest":{"path":str(paths["recovery"]),"sha256":sha(paths["recovery"])}},
              "original_recovery_inventory":old["original_recovery"],
              "recovery_completeness":{"complete":True,"all_collapsed_input_selectors_replaced":True,
                    "all_legacy_raw_source_ids_restored":True,"original_row_identity_preserved":True},
              "source_interval_exclusions":[{**rule,"time_unit":"unix_seconds_on_nominal_axis"}
                        for rule in quarantine_rules["source_interval_exclusions"]],
              "global_raw_source_interval_rules":[{**rule,"time_unit":"unix_seconds_on_nominal_axis"}
                        for rule in quarantine_rules["global_raw_source_interval_rules"]],
              "source_interval_quarantine_manifest":{"path":str(paths["quarantine"]),"sha256":sha(paths["quarantine"])},
              "recovered_raw_ids_absent_frozen_mapping":sorted(extra_names-set(c["fold_groups"])),
              "freshness_exclusion_animals":sorted(set(c["freshness_exclusion_animals"]) | extra_names),
              "parent_role_conflicts":role_conflicts,
              "parent_freshness_conflicts":freshness_conflicts,
              "policy":{"generic_exact_position_physical_join":"disabled; coincidence remains exclusion-only",
                        "flat_star_physical_import":"disabled; rebuild only retained audited primitives",
                        "parent_role_freshness":"inherit unchanged through original conservative exclusion graph",
                        "collapsed_observation_reuse":"reject unless original per-source trajectories have been reconstructed and bound",
                        "parent_origin_reuse":"requires explicit retirement/reconstruction accounting; old mixed trajectories are not silently reinterpreted"},
              "summary":{"all_raw_ids":len(mapping),"physical_components":len(set(mapping.values())),
                         "exclusion_components":len(set(exclusion.values())),"retracted_physical_edges":len(retract),
                         "parent_components_split":sum(r["physical_partition_changed"] for r in memberships),
                         "recovered_raw_ids_absent_frozen_mapping":len(extra_names-set(c["fold_groups"])),
                         "input_source_ids_needing_original_recovery":sum(r["status"]!="unmixed_original_source_id" for r in ownership)},
              "scope":"Identity/row-ownership repair proof only; no case labels, predictions, errors or outcomes inspected; frozen inputs unchanged"}
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(output),"sha256":sha(output),"summary":result["summary"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["parent","combined","review","lineage","legacy-review","recovery","quarantine","output"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    assemble(args.parent,args.combined,args.review,args.lineage,args.legacy_review,args.recovery,args.quarantine,args.output)
