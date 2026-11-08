import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('breadth_plan', Path(__file__).resolve().parents[1] / 'scripts/migration_breadth_plan.py')
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def test_scope_holds_explicit_domestics_and_humans_but_keeps_feral_horses():
    assert plan.scope_status(['Homo sapiens'], 'GPS tracking') == 'human_scope_hold'
    assert plan.scope_status(['Anas platyrhynchos','Cairina moschata'], 'Tracking domestic ducks') == 'domestic_scope_hold'
    assert plan.scope_status(['Columba livia'], 'Navigation by homing pigeons') == 'domestic_scope_hold'
    assert plan.scope_status(['Equus ferus'], 'Feral horse movement in Wyoming') == 'wildlife_candidate_unverified'
    assert plan.scope_status(['Calonectris diomedea'], 'Homing orientation in free-ranging shearwaters') == 'wildlife_candidate_unverified'


def test_temporal_hints_never_claim_verified_training_availability():
    for title in ['GPS tracking 2020-2022','GPS tracking 2022-2024','GPS tracking 2021','GPS tracking']:
        result = plan.temporal(title, [])
        assert result['counts_as_confirmed_pre2021_training'] is False
    assert not plan.temporal('GPS tracking 2020-2022',[])['excluded_from_potential_2020_training_taxa']
    assert plan.temporal('GPS tracking 2022-2024',[])['excluded_from_potential_2020_training_taxa']
    assert plan.temporal('GPS tracking 2021',[])['excluded_from_potential_2020_training_taxa']
    assert plan.temporal('GPS tracking',[])['status']=='tracking_years_unknown'


def test_sensor_media_and_documentation_roles_are_explicit():
    assert plan.role({'name':'bird-gps.csv','original_name':'bird-gps.csv','bytes':100})=='events'
    assert plan.role({'name':'README.csv','original_name':'README.csv','bytes':100})=='source_documentation'
    for name in ['bird-acc3of3.csv.zip','bird_part-tdr.csv.zip','cattle-video.zip']:
        assert plan.role({'name':name,'original_name':name,'bytes':100})=='non_location_sensor_or_media'


def test_inventory_preserves_distinct_readmes_deduplicates_exact_copies_and_pins_responses(tmp_path):
    def bitstream(uuid,name,checksum):
        return {'uuid':uuid,'name':name,'sizeBytes':10,'checkSum':{'checkSumAlgorithm':'MD5','value':checksum},
                '_links':{'content':{'href':'https://public.example/content/'+uuid}}}
    body={'_embedded':{'bitstreams':[bitstream('a','README.txt','1'),bitstream('b','README.txt','2'),
          bitstream('c','README.csv','3'),bitstream('d','README.csv','3'),bitstream('e','gps.csv','4')]}}
    response=tmp_path/'response.json';response.write_text(json.dumps(body))
    metadata={'http_responses':[{'source_id':'source','path':str(response),
                'decoded_response_sha256':hashlib.sha256(response.read_bytes()).hexdigest()}]}
    (tmp_path/'bitstreams.json').write_text(json.dumps(metadata))
    inventory,_=plan.inventory(tmp_path)
    files=inventory['source']
    assert len(files)==4 and len({f['name'] for f in files})==4
    assert {f['original_name'] for f in files}=={'README.txt','README.csv','gps.csv'}
    assert {f['name'] for f in files if f['original_name']=='README.txt'}=={'README__a.txt','README__b.txt'}
    response.write_text(response.read_text()+' ')
    with pytest.raises(ValueError,match='checksum'):plan.inventory(tmp_path)
