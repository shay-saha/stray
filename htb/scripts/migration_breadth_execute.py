"""Execute one explicitly approved, checksum-pinned breadth intake plan.

Original bytes and archive members are preserved. No event filtering, coordinate
parsing, source quality cleaning, or training occurs in this finite job.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import threading
from urllib.parse import urlparse
import zipfile

PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('migration_acquire',PROJECT/'scripts/migration_acquire.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):return a.digest(path)


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


class Storage:
    """Shared byte accounting; serialized writes prevent free-floor races."""
    def __init__(self,root,maximum,floor):
        self.root=Path(root);self.maximum=int(maximum);self.floor=int(floor);self.lock=threading.Lock()
        areas=[self.root/'cache/originals',self.root/'cache/pending',*list((self.root/'raw').glob('*/extracted'))]
        self.used=sum(p.stat().st_size for area in areas for p in area.rglob('*') if p.is_file() and not p.is_symlink())
        if self.used>self.maximum:raise ValueError('Existing intake storage exceeds the approved total')
    def write(self,stream,block):
        with self.lock:
            if self.used+len(block)>self.maximum:raise ValueError('Shared ORIGINAL-plus-extracted quota exhausted')
            if shutil.disk_usage(self.root).free-len(block)<self.floor:raise ValueError('Server free-floor would be breached')
            stream.write(block);stream.flush();self.used+=len(block)
    def remove(self,path):
        path=Path(path)
        with self.lock:
            if path.exists() and not path.is_symlink():
                size=path.stat().st_size;path.unlink();self.used-=size


def safe_url(url):
    u=urlparse(url)
    if (u.scheme!='https' or u.hostname not in {'datarepository.movebank.org','www.datarepository.movebank.org'}
        or u.username or u.password or u.query or u.fragment
        or not re.fullmatch(r'/server/api/core/bitstreams/[0-9a-fA-F-]+/content',u.path)):
        raise ValueError('Only pinned public repository bitstream content URLs are allowed')
    return url


def verify(path,record,expected_sha=None):
    path=Path(path)
    if not path.is_file() or path.stat().st_size!=record['bytes']:raise ValueError('ORIGINAL size mismatch')
    ck=record['published_checksum'];algorithm=ck['checkSumAlgorithm'].casefold()
    if algorithm not in {'md5','sha256','sha1','sha512'}:raise ValueError('Unsupported published checksum')
    if a.digest(path,algorithm)!=ck['value'].casefold():raise ValueError('Published ORIGINAL checksum mismatch')
    actual=sha(path)
    if expected_sha is not None and actual!=expected_sha:raise ValueError('Cached ORIGINAL SHA256 mismatch')
    return actual


def link(source,destination):
    source,destination=Path(source),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve()!=source.resolve():raise ValueError('Existing intake link conflicts with pinned source')
        return
    destination.symlink_to(os.path.relpath(source,destination.parent))


class Intake:
    def __init__(self,plan,root,storage):
        self.plan=plan;self.root=Path(root);self.storage=storage;self.locks={};self.guard=threading.Lock()
    def original(self,record):
        if record['bytes']>self.plan['quotas']['maximum_original_file_bytes']:raise ValueError('ORIGINAL exceeds per-file quota')
        safe_url(record['url']);a.safe_name(record['name'])
        ck=record['published_checksum'];key=(ck['checkSumAlgorithm'].casefold(),ck['value'].casefold())
        with self.guard:lock=self.locks.setdefault(key,threading.Lock())
        with lock:
            # Prefer the exact approved SHA-addressed cached bytes.
            for known in record.get('cached_matches',[]):
                if not re.fullmatch(r'[a-f0-9]{64}',known['sha256']):raise ValueError('Unsafe cached SHA256 path')
                path=self.root/'cache/originals'/known['sha256']
                if path.exists():return path,verify(path,record,known['sha256'])
            pointer=self.root/'cache/checksums'/hashlib.sha256(repr(key).encode()).hexdigest()
            if pointer.exists():
                actual=pointer.read_text().strip()
                if not re.fullmatch(r'[a-f0-9]{64}',actual):raise ValueError('Unsafe cached checksum pointer')
                path=self.root/'cache/originals'/actual
                return path,verify(path,record,actual)
            if record['action']!='public_checksum_verified_download_after_approval':
                raise ValueError('Approved reusable ORIGINAL is missing; transfer/reverify it instead of re-downloading')
            pending=self.root/'cache/pending'/(record['bitstream_uuid']+'.part');pending.parent.mkdir(parents=True,exist_ok=True)
            self.storage.remove(pending)
            try:
                with a.session().get(record['url'],stream=True,timeout=(20,120),headers={'Accept-Encoding':'identity'}) as response:
                    response.raise_for_status()
                    length=response.headers.get('Content-Length')
                    if length and response.headers.get('Content-Encoding','identity')=='identity' and int(length)!=record['bytes']:raise ValueError('Pinned ORIGINAL response size changed')
                    size=0
                    with pending.open('wb') as stream:
                        for block in response.iter_content(1024*1024):
                            if not block:continue
                            size+=len(block)
                            if size>record['bytes']:raise ValueError('Pinned ORIGINAL bytes exceeded')
                            self.storage.write(stream,block)
                actual=verify(pending,record)
                final=self.root/'cache/originals'/actual;final.parent.mkdir(parents=True,exist_ok=True)
                if final.exists():verify(final,record,actual);self.storage.remove(pending)
                else:pending.rename(final)
                pointer.parent.mkdir(parents=True,exist_ok=True);pointer.write_text(actual+'\n')
                return final,actual
            finally:self.storage.remove(pending)
    def extract(self,original,record,directory):
        role=record['role'];derived=[]
        if role=='events':
            dest=directory/record['name'];link(original,dest)
            return [{'status':'byte_exact_original_streaming_no_quality_filter','chunks':[{'name':dest.name,'bytes':record['bytes'],'sha256':sha(original),'rows':None}]}]
        if role!='archive':return []
        cap=record['extracted_hard_cap_bytes'];max_file=self.plan['quotas']['maximum_original_file_bytes']
        destination=directory/'extracted'/record['bitstream_uuid'];destination.mkdir(parents=True,exist_ok=True)
        def write_member(stream,name,expected=None):
            safe=a.safe_name(name)
            suffix=hashlib.sha256(name.encode()).hexdigest()[:12]
            target=destination/(Path(safe).stem+'__'+suffix+Path(safe).suffix)
            # extract() is skipped only after the previous complete file set was
            # verified. Reconstruct an invalid/incomplete set from ORIGINAL.
            if target.is_symlink():raise ValueError('Archive destination must not be a symlink')
            self.storage.remove(target)
            pending=target.with_name(target.name+'.part');self.storage.remove(pending);n=0
            already=sum(p.stat().st_size for p in destination.iterdir() if p.is_file() and not p.name.endswith('.part'))
            try:
                with pending.open('wb') as sink:
                    while True:
                        block=stream.read(1024*1024)
                        if not block:break
                        n+=len(block)
                        if n>max_file or already+n>cap:raise ValueError('Pinned archive extraction quota exceeded')
                        self.storage.write(sink,block)
                if expected is not None and n!=expected:raise ValueError('Archive member size mismatch')
                pending.rename(target)
            finally:self.storage.remove(pending)
            return {'name':str(target.relative_to(directory)),'bytes':n,'sha256':sha(target),'rows':None}
        if original.name.casefold().endswith('.gz') or record['name'].casefold().endswith('.csv.gz'):
            with gzip.open(original,'rb') as stream:chunk=write_member(stream,record['name'][:-3])
            derived.append({'status':'byte_exact_decompression_no_quality_filter','chunks':[chunk]})
        else:
            with zipfile.ZipFile(original) as archive:
                members=archive.infolist()
                if len(members)>10000:raise ValueError('Archive member-count quota exceeded')
                if len({m.filename for m in members})!=len(members):raise ValueError('Duplicate archive member names require source review')
                selected=[]
                for member in members:
                    a.safe_name(member.filename)
                    if stat.S_ISLNK(member.external_attr>>16):raise ValueError('Archive symlinks are forbidden')
                    if member.is_dir() or '__MACOSX' in Path(member.filename).parts:continue
                    if Path(member.filename).suffix.casefold() not in {'.csv','.txt','.md','.html','.htm'}:continue
                    selected.append(member)
                if sum(m.file_size for m in selected)>cap or any(m.file_size>max_file for m in selected):raise ValueError('Pinned archive extraction quota exceeded')
                for member in sorted(selected,key=lambda m:m.filename):
                    with archive.open(member) as stream:chunk=write_member(stream,member.filename,member.file_size)
                    derived.append({'archive_member':member.filename,'status':'byte_exact_decompression_no_quality_filter','chunks':[chunk]})
        return derived


def execute(plan_path,root,approval_path,catalogue,workers=4):
    plan_path,root,approval_path,catalogue=map(Path,(plan_path,root,approval_path,catalogue))
    plan_hash=sha(plan_path);plan=json.loads(plan_path.read_text());approval=json.loads(approval_path.read_text())
    if approval.get('approved') is not True or approval.get('plan_sha256')!=plan_hash:raise ValueError('Explicit approval must bind this exact plan SHA256')
    if plan.get('format')!='wildcast-breadth-metadata-plan-v1' or not 1<=workers<=4:raise ValueError('Unsupported plan or worker count')
    if sha(catalogue)!=plan['catalogue_sha256']:raise ValueError('Parent catalogue SHA256 mismatch')
    quota=plan['quotas']
    if quota['original_plus_extracted_hard_cap_bytes']>30_000_000_000 or quota['maximum_original_file_bytes']>2_000_000_000 or quota['minimum_server_free_bytes']<50_000_000_000:raise ValueError('Plan weakens the approved hard quotas')
    root.mkdir(parents=True,exist_ok=True)
    if root.resolve()!=Path(plan['execution_contract']['raw_root']).parent.resolve():raise ValueError('Output root differs from the approved raw layout')
    records={r['id']:r for r in json.loads(catalogue.read_text())['records']}
    run=root/'executions'/plan_hash;run.mkdir(parents=True,exist_ok=True)
    launch=run/'launch.json'
    metadata={str(p.relative_to(root)):sha(p) for p in sorted((root/'metadata').rglob('*')) if p.is_file()}
    descriptor={'plan_sha256':plan_hash,'approval_sha256':sha(approval_path),'catalogue_sha256':sha(catalogue),
        'executor_sha256':sha(__file__),'acquisition_helpers_sha256':sha(PROJECT/'scripts/migration_acquire.py'),
        'public_sources_helpers_sha256':sha(PROJECT/'wildcast/sources.py'),
        'frozen_v3_manifest_sha256':sha(PROJECT/'data/migration-expanded-dev-v3/manifest.json') if (PROJECT/'data/migration-expanded-dev-v3/manifest.json').exists() else None,
        'metadata_files_sha256':metadata,'workers':workers,'original_plus_extracted_cap_bytes':quota['original_plus_extracted_hard_cap_bytes'],
        'maximum_file_bytes':quota['maximum_original_file_bytes'],'minimum_free_bytes':quota['minimum_server_free_bytes'],'scope':'Exact approved public intake; no reranking, quality cleaning, coordinate parsing, or training','started_utc':now()}
    if launch.exists():
        previous=json.loads(launch.read_text())
        for key in ('plan_sha256','approval_sha256','catalogue_sha256','executor_sha256','acquisition_helpers_sha256','public_sources_helpers_sha256','frozen_v3_manifest_sha256','metadata_files_sha256'):
            if previous[key]!=descriptor[key]:raise ValueError('Resume provenance changed: '+key)
    else:atomic(launch,descriptor)
    storage=Storage(root,quota['original_plus_extracted_hard_cap_bytes'],quota['minimum_server_free_bytes']);intake=Intake(plan,root,storage)
    results={};progress_lock=threading.Lock()
    def worker(item):
        source=records[item['id']];directory=root/'raw'/item['id'];directory.mkdir(parents=True,exist_ok=True)
        existing=directory/'source.json';previous=json.loads(existing.read_text()) if existing.exists() else {}
        if previous and previous.get('plan_sha256')!=plan_hash:raise ValueError('Source belongs to another plan')
        previous_files={f['bitstream_uuid']:f for f in previous.get('files',[])}
        completed=[]
        for record in item['files']:
            if not record['included_for_intake']:continue
            value={**record,'status':'pending'}
            try:
                original,digest=intake.original(record)
                target=directory/'originals'/record['name'];link(original,target)
                old=previous_files.get(record['bitstream_uuid'])
                valid=old and old.get('status')=='complete' and old.get('sha256')==digest
                if valid:
                    for d in old.get('derived',[]):
                        for chunk in d.get('chunks',[]):
                            relative=Path(chunk['name'])
                            if relative.is_absolute() or '..' in relative.parts or '\\' in chunk['name']:raise ValueError('Unsafe resumed extracted path')
                            path=directory/relative
                            if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Resumed extracted path escapes intake')
                            if not path.is_file() or sha(path)!=chunk['sha256']:valid=False
                if valid:value=old
                else:
                    value.update(sha256=digest,stored_as=str(target.relative_to(directory)),status='original_verified')
                    value.update(derived=intake.extract(original,record,directory),status='complete',completed_utc=now())
                    if record['role']=='identity_reference':link(original,directory/record['name'])
            except Exception as e:value.update(status='error',error=type(e).__name__+': '+str(e))
            completed.append(value)
            atomic(existing,{**source,'source_id':item['id'],'plan_sha256':plan_hash,'metadata_source_qa':'Unreviewed; no cleaner or actual-GPS assertion performed by acquisition','files':completed,'acquisition_complete':False,'retrieved_utc':now()})
        result={'id':item['id'],'status':'complete' if all(f['status']=='complete' for f in completed) else 'complete_with_gaps','files':completed,'source_manifest':str(existing)}
        atomic(existing,{**source,'source_id':item['id'],'plan_sha256':plan_hash,'metadata_source_qa':'Unreviewed; no cleaner or actual-GPS assertion performed by acquisition','files':completed,'acquisition_complete':True,'retrieved_utc':now()})
        with progress_lock:
            results[item['id']]=result
            atomic(run/'progress.json',{'plan_sha256':plan_hash,'launch_sha256':sha(launch),'completed_sources':len(results),'planned_sources':len(plan['selected_sources']),'original_plus_extracted_bytes':storage.used,'results':[results[k] for k in sorted(results)],'updated_utc':now()})
        print(item['id'],result['status'],len(completed),'files',flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,item):item['id'] for item in plan['selected_sources']}
        for future in as_completed(futures):
            try:future.result()
            except Exception as e:
                with progress_lock:results[futures[future]]={'id':futures[future],'status':'source_error','error':type(e).__name__+': '+str(e)}
    final={'plan_sha256':plan_hash,'launch_sha256':sha(launch),'sources':len(results),'complete_sources':sum(r['status']=='complete' for r in results.values()),'original_plus_extracted_bytes':storage.used,'free_bytes':shutil.disk_usage(root).free,'results':[results[k] for k in sorted(results)],'finished_utc':now(),'quality_cleaning_or_training':False}
    finished=run/('finished.json' if not (run/'finished.json').exists() else 'resumed-finished-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
    atomic(finished,final);print(json.dumps({k:v for k,v in final.items() if k!='results'},indent=2));return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan_path','root','approval_path','catalogue'):p.add_argument('--'+key.replace('_','-'),required=True)
    p.add_argument('--workers',type=int,default=4)
    execute(**vars(p.parse_args()))

if __name__=='__main__':main()
