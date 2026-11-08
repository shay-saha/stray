#!/usr/bin/env python3
"""Loopback-only WildCast inference bridge. No synthetic histories or model weights."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'wildlife-atlas-wildcast-v1'


def make_handler(bundle: Path, device: str):
    # Import the immutable bundle, rather than the actively changing research checkout.
    sys.path.insert(0, str(bundle))
    spec = importlib.util.spec_from_file_location('atlas_verify', bundle / 'verify_package.py')
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    verified = verify.verify(bundle)
    import torch
    from cohort_forecast import CohortEngine, Observations, utc
    from scenario import Scenarios
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment. Install CUDA PyTorch, or explicitly select --device cpu.')
    selected = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if selected == 'cuda':
        torch.cuda.init()
    # The packaged checkpoints contain relative paths. Preserve the verified bundle's layout.
    os.chdir(bundle)
    gate = threading.Lock()
    engine = CohortEngine(bundle, selected)
    scenarios = Scenarios(engine)
    default_observations = ROOT / 'assets/observation-sources.json'
    jobs, jobs_lock = {}, threading.Lock()
    info = {'service': SERVICE, 'device': selected, 'gpu': torch.cuda.get_device_name(0) if selected == 'cuda' else None,
            'bundle': str(bundle), 'model_kind': verified['model_kind'], 'verified_files': verified['verified_files'],
            'trained_taxa':len(engine.manifest['species']),'horizons_days':engine.horizons.tolist(),
            'checkpoint_sha256':engine.checkpoint_hash,'model_protocol':engine.manifest.get('protocol'),
            'api_version': 2, 'scenario_api_version': 1,
            'scope': 'Research migration checkpoint. One forecast per observed animal; study/species cohorts are not verified colonies.'}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {**info, 'busy': gate.locked()})
            elif self.path == '/demo':
                # Explicit recorded history, never the atlas's synthetic dots.
                meta = json.loads((ROOT.parent / 'htb/artifacts/migration-demo-history.json').read_text())
                self.reply(200, {**meta, 'observations': str(ROOT.parent / 'htb/artifacts/migration-demo-history.csv')})
            elif self.path == '/observations':
                try:
                    value=engine.cached_default(default_observations)
                    if value is None:value=engine.dataset(default_observations).preview()
                    value['runtime']=info
                    self.reply(200,value)
                except (OSError,ValueError,KeyError) as error:
                    self.reply(400,{'error':str(error)})
            elif self.path.startswith('/jobs/'):
                job_id=self.path.removeprefix('/jobs/')
                with jobs_lock:
                    job=dict(jobs.get(job_id,{}))
                self.reply(200,job) if job else self.reply(404,{'error':'Forecast job not found'})
            else:
                self.reply(404, {'error': 'Unknown endpoint'})

        def do_POST(self):
            if self.path not in {'/forecast','/observations','/forecast-cohorts','/scenario','/scenario/advance'}:
                self.reply(404, {'error': 'Unknown endpoint'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32768:
                    raise ValueError('Expected a JSON request of at most 32 KiB')
                request = json.loads(self.rfile.read(length))
                if self.path in {'/observations','/forecast-cohorts','/scenario','/scenario/advance'}:
                    path=Path(request.get('observations') or default_observations).expanduser().resolve(strict=True)
                    arguments={key:request[key] for key in ('issue_time','max_age_hours','species') if request.get(key) is not None}
                    if self.path=='/observations':
                        value=engine.dataset(path).preview(**arguments)
                        value['runtime']=info
                        self.reply(200,value)
                        return
                    samples=int(request.get('samples',16))
                    if not 2<=samples<=128:
                        raise ValueError('Choose 2 to 128 uncertainty paths per tracked animal')
                    if not gate.acquire(blocking=False):
                        self.reply(409,{'error':'A forecast is already running'})
                        return
                    job_id=uuid.uuid4().hex
                    with jobs_lock:
                        if len(jobs)>=4:
                            oldest=next((key for key,value in jobs.items() if value['state']!='running'),None)
                            if oldest is not None:jobs.pop(oldest)
                        jobs[job_id]={'job_id':job_id,'state':'running','phase':'Reading observations','completed':0,'total':0}
                    def run_job():
                        try:
                            def progress(phase,completed,total):
                                with jobs_lock:
                                    jobs[job_id].update(phase=phase,completed=completed,total=total)
                            if self.path=='/scenario':
                                result=scenarios.start(engine.dataset(path),issue_time=request.get('issue_time') or None,
                                      samples=samples,seed=int(request.get('seed',2026)),progress=progress)
                            elif self.path=='/scenario/advance':
                                result=scenarios.advance(str(request['scenario_id']),int(request['generation']),progress=progress)
                            else:
                                result=engine.forecast(engine.dataset(path),samples=samples,seed=int(request.get('seed',2026)),progress=progress,**arguments)
                            result['runtime'].update(info)
                            with jobs_lock:
                                jobs[job_id].update(state='complete',result=result)
                        except Exception as error:
                            self.log_error('Cohort forecast failed: %s',error)
                            with jobs_lock:
                                jobs[job_id].update(state='failed',error=str(error))
                        finally:
                            gate.release()
                    threading.Thread(target=run_job,name='wildcast-cohort-forecast',daemon=True).start()
                    self.reply(202,{'job_id':job_id,'state':'running'})
                    return
                path = Path(request['observations']).expanduser().resolve(strict=True)
                if path.suffix.lower() not in {'.csv', '.parquet', '.pq'}:
                    raise ValueError('Observations must be CSV or parquet')
                species = str(request['species']).strip()
                samples = int(request.get('samples', 32))
                if not 2 <= samples <= 128:
                    raise ValueError('Choose 2 to 128 trajectory samples')
                if not gate.acquire(blocking=False):
                    self.reply(409, {'error': 'A forecast is already running'})
                    return
                try:
                    observed=Observations(path,species_override=species)
                    selected_animal=request.get('animal')
                    if selected_animal:
                        observed.frame=observed.frame.loc[observed.frame.animal==str(selected_animal)].copy()
                    if request.get('issue_time'):
                        observed.frame=observed.frame.loc[observed.frame.timestamp<=utc(request['issue_time'])].copy()
                    if observed.frame.empty:raise ValueError('No valid observed GPS history at or before issue time')
                    if observed.frame.animal.nunique()!=1:raise ValueError('Select one observed animal with the animal argument')
                    one=engine.forecast(observed,issue_time=observed.frame.timestamp.max().isoformat(),max_age_hours=168,
                                        samples=samples,seed=int(request.get('seed',2026)),species=[species])
                    result=one['tracks'][0]
                    result.update(model_kind=engine.kind,selected_animal=result['id'],requested_issue_time=request.get('issue_time'),
                                  retrospective_replay=one['retrospective_replay'],checkpoint_sha256=engine.checkpoint_hash,
                                  runtime={**one['runtime'],**info})
                    self.reply(200, result)
                finally:
                    gate.release()
            except (ValueError, KeyError, OSError, TypeError) as error:
                self.reply(400, {'error': str(error)})
            except Exception as error:
                self.log_error('Inference failed: %s', error)
                self.reply(500, {'error': str(error)})

    return Handler, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT / 'models/server-repaired-v2')
    p.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    handler, info = make_handler(args.bundle.resolve(), args.device)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    server.daemon_threads = True
    print(json.dumps({**info, 'url': f'http://127.0.0.1:{args.port}'}), flush=True)
    server.serve_forever()

if __name__ == '__main__':
    main()
