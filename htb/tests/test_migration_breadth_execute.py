import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

spec=importlib.util.spec_from_file_location('breadth_execute',Path(__file__).resolve().parents[1]/'scripts/migration_breadth_execute.py')
executor=importlib.util.module_from_spec(spec);spec.loader.exec_module(executor)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path/'intake';root.mkdir()
    source='00000000-0000-0000-0000-000000000001'
    bit='11111111-1111-1111-1111-111111111111'
    data=b'animal,timestamp,lat,lon,modelled\r\na,2020-01-01,1,2,true\r\n'
    cat=tmp_path/'catalogue.json';cat.write_text(json.dumps({'records':[{'id':source,'title':'Public GPS','metadata':{'sensor.name':['GPS']}}]}))
    file={'bitstream_uuid':bit,'name':'gps.csv','original_name':'gps.csv','bytes':len(data),
          'url':'https://datarepository.movebank.org/server/api/core/bitstreams/'+bit+'/content',
          'published_checksum':{'checkSumAlgorithm':'MD5','value':hashlib.md5(data).hexdigest()},
          'role':'events','included_for_intake':True,'action':'public_checksum_verified_download_after_approval',
          'cached_matches':[],'extracted_hard_cap_bytes':1024}
    plan={'format':'wildcast-breadth-metadata-plan-v1','catalogue_sha256':executor.sha(cat),
        'quotas':{'original_plus_extracted_hard_cap_bytes':1000000,'maximum_original_file_bytes':100000,
                  'minimum_server_free_bytes':50000000000},'selected_sources':[{'id':source,'files':[file]}],
        'execution_contract':{'raw_root':str(root/'raw'),'approved_to_download':False}}
    plan_path=tmp_path/'plan.json';approval=tmp_path/'approval.json'
    def freeze():
        plan_path.write_text(json.dumps(plan));approval.write_text(json.dumps({'approved':True,'plan_sha256':executor.sha(plan_path)}))
    freeze()
    calls=[];payload={'value':data}
    class Response:
        def __init__(self):self.headers={'Content-Length':str(len(payload['value']))}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):
            for offset in range(0,len(payload['value']),17):yield payload['value'][offset:offset+17]
    class Session:
        def get(self,url,**kwargs):calls.append(url);return Response()
    monkeypatch.setattr(executor.a,'session',Session)
    monkeypatch.setattr(executor.shutil,'disk_usage',lambda *args:SimpleNamespace(free=150000000000))
    return SimpleNamespace(root=root,source=source,file=file,plan=plan,plan_path=plan_path,approval=approval,
                           cat=cat,freeze=freeze,calls=calls,payload=payload,data=data)


def run(s):return executor.execute(s.plan_path,s.root,s.approval,s.cat,workers=2)


def test_exact_original_csv_streams_without_cleaning_or_redundant_chunks_and_resumes(setup):
    s=setup;first=run(s)
    assert first['complete_sources']==1 and first['original_plus_extracted_bytes']==len(s.data)
    source=s.root/'raw'/s.source
    assert (source/'gps.csv').is_symlink() and (source/'gps.csv').read_bytes()==s.data
    assert b'modelled' in (source/'gps.csv').read_bytes() and b'true' in (source/'gps.csv').read_bytes()
    launch=list((s.root/'executions').glob('*/launch.json'))[0];before=launch.read_bytes()
    second=run(s)
    assert len(s.calls)==1 and second['complete_sources']==1
    assert launch.read_bytes()==before
    assert json.loads(s.plan_path.read_text())['execution_contract']['approved_to_download'] is False


def test_storage_and_free_floor_are_hard_before_writing(tmp_path,monkeypatch):
    monkeypatch.setattr(executor.shutil,'disk_usage',lambda *args:SimpleNamespace(free=1000))
    budget=executor.Storage(tmp_path,3,0)
    with (tmp_path/'x').open('wb') as stream:
        with pytest.raises(ValueError,match='quota'):budget.write(stream,b'1234')
    assert budget.used==0
    budget=executor.Storage(tmp_path,100,999)
    with (tmp_path/'y').open('wb') as stream:
        with pytest.raises(ValueError,match='free-floor'):budget.write(stream,b'12')
    assert budget.used==0


def test_wrong_checksum_retains_no_unverified_original_or_partial_bytes(setup):
    s=setup;s.payload['value']=s.data.replace(b'true',b'fake')
    report=run(s)
    assert report['complete_sources']==0 and report['original_plus_extracted_bytes']==0
    assert not list((s.root/'cache').rglob('*.part'))
    assert 'checksum' in report['results'][0]['files'][0]['error']


def zip_payload(s,files):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:
        for name,value in files.items():z.writestr(name,value)
    s.payload['value']=data.getvalue();s.file.update(name='gps.zip',original_name='gps.zip',role='archive',
        bytes=len(data.getvalue()),published_checksum={'checkSumAlgorithm':'MD5','value':hashlib.md5(data.getvalue()).hexdigest()})
    s.freeze()


@pytest.mark.parametrize('failure',['traversal','expanded-quota'])
def test_archive_guards_keep_verified_original_but_write_no_unsafe_events(setup,failure):
    s=setup
    zip_payload(s,{'../../secret.csv':s.data} if failure=='traversal' else {'events.csv':s.data*100})
    report=run(s)
    assert report['complete_sources']==0
    entry=report['results'][0]['files'][0]
    assert entry['sha256'] and (s.root/'raw'/s.source/entry['stored_as']).is_file()
    assert not (s.root.parent/'secret.csv').exists()
    assert not list((s.root/'raw').rglob('*.part'))


def test_corrupt_derived_archive_member_is_reconstructed_from_verified_original(setup):
    s=setup;zip_payload(s,{'events.csv':s.data})
    first=run(s);entry=first['results'][0]['files'][0]
    path=s.root/'raw'/s.source/entry['derived'][0]['chunks'][0]['name']
    path.write_bytes(b'x'*len(s.data))
    second=run(s)
    assert second['complete_sources']==1 and path.read_bytes()==s.data and len(s.calls)==1


def test_approval_parent_and_public_url_guards_prevent_unapproved_or_private_requests(setup):
    s=setup;s.approval.write_text(json.dumps({'approved':True,'plan_sha256':'0'*64}))
    with pytest.raises(ValueError,match='approval'):run(s)
    s.freeze();s.file['url']='https://localhost/private/key';s.freeze()
    report=run(s)
    assert report['complete_sources']==0 and not s.calls
    assert 'public repository' in report['results'][0]['files'][0]['error']


def test_missing_approved_local_cache_never_silently_redownloads(setup):
    s=setup;s.file['action']='transfer_reverify_local_original';s.freeze()
    report=run(s)
    assert report['complete_sources']==0 and not s.calls
