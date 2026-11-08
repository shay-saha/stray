import csv
import json

import pandas as pd
import pytest

from wildcast.breadth_identity import Groups, audit, exact_links, load_taxonomy, metadata_links
from wildcast.marine_intake import sha256


def record(source="a", species="Pekania pennanti", animal="animal-a", **kwargs):
    return {"source_id": source, "species": species, "local_identifier": animal,
            "observation_animals": [animal], "ring": "", "manufacturer": "vendor", "tag_serial": "",
            "tag_id": "", "deploy_on": "2019-01-01 00:00:00", "site": "", "sex": "f",
            "reference": source + "-reference.csv", "reference_sha256": "a" * 64, "line": 2, **kwargs}


def taxonomy(tmp_path):
    path=tmp_path/"taxonomy.json"
    path.write_text(json.dumps({"aliases":[{"from":"Martes pennanti","to":"Pekania pennanti","relationship":"genus_change","source_url":"https://authority.example/fisher"}],"unresolved":[]}))
    return path


def test_synonym_names_alone_never_link_individuals_but_corroborated_ring_can(tmp_path):
    canonical,_=load_taxonomy(taxonomy(tmp_path))
    rows=[record(),record("b","Martes pennanti","animal-b")]
    assert metadata_links(rows,canonical)[:2]==([],[])
    rows[0]["ring"]=rows[1]["ring"]="DERA12345"
    strong,uncertain,_=metadata_links(rows,canonical)
    assert len(strong)==1 and not uncertain
    assert strong[0]["evidence"]["same_taxon_key"] is True


def test_local_hardware_ids_are_uncertain_and_redeployments_with_distinct_rings_are_not_physical():
    canonical,_=load_taxonomy()
    rows=[record(tag_id="12345"),record("b",animal="animal-b",tag_id="12345")]
    strong,uncertain,_=metadata_links(rows,canonical)
    assert strong==[] and len(uncertain)==1
    rows[0].update(tag_serial="SERIAL12345",ring="DERA12345")
    rows[1].update(tag_serial="SERIAL12345",ring="DERB99999")
    strong,_,ruled=metadata_links(rows,canonical)
    assert not strong and ruled


def test_exact_gps_synonym_overlap_requires_five_distinct_moving_fixes(tmp_path):
    canonical,_=load_taxonomy(taxonomy(tmp_path))
    dates=pd.date_range("2019-01-01",periods=6,freq="h",tz="UTC")
    old=pd.DataFrame({"animal":"old","species":"Pekania pennanti","timestamp":dates,"lat":[1+i/10 for i in range(6)],"lon":2.})
    new=old.rename(columns={"timestamp":"timestamp_nominal"}).assign(animal="new",species="Martes pennanti")
    old.to_parquet(tmp_path/"old.parquet");new.to_parquet(tmp_path/"new.parquet")
    links=exact_links(tmp_path/"new.parquet",[tmp_path/"old.parquet"],canonical,Groups(["old","new"]))
    assert len(links)==1 and links[0]["shared_distinct_fixes_at_least"]==5
    repeated=pd.concat([new.iloc[[0,5]]]*5,ignore_index=True);repeated.to_parquet(tmp_path/"repeated.parquet")
    assert exact_links(tmp_path/"repeated.parquet",[tmp_path/"old.parquet"],canonical,Groups(["old","new"]))==[]


def source_reference(directory,source,local,taxon,tag="12345",ring=""):
    directory.mkdir(parents=True)
    path=directory/"Study-reference-data.csv"
    row={"animal-id":local,"animal-taxon":taxon,"tag-id":tag,"animal-ring-id":ring,
         "tag-manufacturer-name":"vendor","deploy-on-date":"2019-01-01 00:00:00","animal-sex":"f"}
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,list(row));writer.writeheader();writer.writerow(row)
    (directory/"source.json").write_text(json.dumps({"id":source,"files":[{"name":path.name,"sha256":sha256(path)}]}))


def frame(source,local,taxon,new=False):
    timecol="timestamp_nominal" if new else "timestamp"
    return pd.DataFrame({"animal":f"{source}:Study:{taxon}:{local}","source":source,"study":source+":Study","species":taxon,
                         timecol:pd.date_range("2019-01-01",periods=6,freq="h",tz="UTC"),"lat":10. if new else 1.,"lon":2.})


def parent_cache(tmp_path,old,roles):
    root=tmp_path/"parent";root.mkdir()
    origins=pd.DataFrame({"canonical_animal":list(roles),"fresh_identity":[True]*len(roles)})
    origins.to_parquet(root/"origins.parquet");(root/"cases.npz").write_bytes(b"sealed bytes; never loaded as labels")
    manifest={"origins_sha256":sha256(root/"origins.parquet"),"cases_sha256":sha256(root/"cases.npz"),
              "reserved_test_targets_sealed":True,"identity_mapping":{name:name for name in roles},
              "exclusion_groups":{name:name for name in roles},"assignments":roles}
    (root/"manifest.json").write_text(json.dumps(manifest))
    return root


def test_uncertain_bridge_preserves_physical_groups_and_reports_parent_role_conflict(tmp_path):
    raw=tmp_path/"raw"
    old_a=frame("old-a","a","Pekania pennanti");old_b=frame("old-b","b","Pekania pennanti")
    new=frame("new","c","Pekania pennanti",True)
    for source,local in [("old-a","a"),("old-b","b"),("new","c")]:source_reference(raw/source,source,local,"Pekania pennanti")
    old=pd.concat([old_a,old_b]);old_path=tmp_path/"old.parquet";old.to_parquet(old_path)
    new_path=tmp_path/"new.parquet";new.to_parquet(new_path)
    roles={old_a.animal.iloc[0]:"temporal",old_b.animal.iloc[0]:"test_individual"}
    parent=parent_cache(tmp_path,old,roles)
    result=audit(new_path,[old_path],[raw],parent,tmp_path/"identity.json")
    assert result["summary"]["parent_role_conflicts"]==1
    a,b,c=old_a.animal.iloc[0],old_b.animal.iloc[0],new.animal.iloc[0]
    assert len({result["fold_groups"][name] for name in [a,b,c]})==3
    assert len({result["uncertainty_fold_groups"][name] for name in [a,b,c]})==1
    assert result["new_animal_parent_bindings"][0]["status"]=="existing_sealed_individual_or_risk_group"
    assert json.loads((parent/"manifest.json").read_text())["assignments"]==roles
    assert result["ambiguous"] and c in result["freshness_exclusion_animals"]


def test_taxonomy_cycles_rejected(tmp_path):
    bad=tmp_path/"bad.json";bad.write_text(json.dumps({"aliases":[{"from":"A a","to":"B b","relationship":"genus_change","source_url":"x"},{"from":"B b","to":"A a","relationship":"genus_change","source_url":"x"}]}))
    with pytest.raises(ValueError,match="Cyclic"):
        load_taxonomy(bad)


def test_parent_case_bytes_are_hashed_without_loading_labels(tmp_path):
    old=frame("old","a","Pekania pennanti");new=frame("new","b","Pekania pennanti",True)
    old_path=tmp_path/"old.parquet";old.to_parquet(old_path)
    new_path=tmp_path/"new.parquet";new.to_parquet(new_path)
    parent=parent_cache(tmp_path,old,{old.animal.iloc[0]:"temporal"})
    (parent/"cases.npz").write_bytes(b"changed after sealing")
    with pytest.raises(ValueError,match="integrity/sealing"):
        audit(new_path,[old_path],[],parent,tmp_path/"identity.json")
    assert not (tmp_path/"identity-public-references.json").exists()


def test_combined_new_inputs_keep_all_source_namespaces(tmp_path):
    old=frame("old","a","Pekania pennanti")
    broad=frame("broad","b","Pekania pennanti",True)
    bird=frame("bird","c","Other species",True)
    paths=[]
    for name,table in [("old",old),("broad",broad),("bird",bird)]:
        path=tmp_path/(name+".parquet");table.to_parquet(path);paths.append(path)
    parent=parent_cache(tmp_path,old,{old.animal.iloc[0]:"temporal"})
    result=audit(paths[1:],paths[:1],[],parent,tmp_path/"combined.json")
    assert result["summary"]["new_source_animal_ids"]==2
    assert result["new_observation_inputs"]==[str(v) for v in paths[1:]]
    assert {v["animal"] for v in result["new_animal_parent_bindings"]}=={broad.animal.iloc[0],bird.animal.iloc[0]}
