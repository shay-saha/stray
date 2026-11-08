import pandas as pd

from wildcast.migration_evaluate import paired_animal_bootstrap, sha256
from wildcast import migration_uncertainty


def scores(cross_label):
    rows=[]
    for taxon,identity,delta in [(0,0,1.),(0,1,3.),(1,0 if cross_label else 2,-1.),(1,3,1.)]:
        for model,value in [("reference",10.),("candidate",10.+delta)]:
            rows.append({"model":model,"split":"val_individual","species":taxon,"individual":identity,
                         "horizon_days":1.,"case":len(rows)//2,"endpoint_energy_km":value})
    return pd.DataFrame(rows)


def test_cross_label_block_uses_coupled_bayesian_uncertainty_with_code_provenance():
    frame=scores(True)
    expected=migration_uncertainty.paired_global_cluster_bootstrap(frame,"reference",seed=13,replicates=20)
    actual=paired_animal_bootstrap(frame,"reference",seed=13,replicates=20)
    assert actual["splits"] == expected["splits"]
    assert actual["method"] == migration_uncertainty.METHOD
    assert "Bayesian" in actual["interval_type"]
    assert actual["code_sha256"] == sha256(migration_uncertainty.__file__)


def test_disjoint_species_keep_the_previous_bootstrap_path(monkeypatch):
    def unexpected(*args,**kwargs):
        raise AssertionError("No cross-label exclusion block is present")
    monkeypatch.setattr(migration_uncertainty,"paired_global_cluster_bootstrap",unexpected)
    result=paired_animal_bootstrap(scores(False),"reference",seed=13,replicates=20)
    assert "method" not in result
    assert result["replicates"] == 20
    assert result["splits"]["val_individual"]["candidate"]["delta_km"] == 1.
