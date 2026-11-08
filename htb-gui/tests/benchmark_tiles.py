"""Controlled tile-loader comparison; serves existing raster tiles only to hidden test windows."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]

def run(workers):
    photo=next((ROOT/'assets/earth-detail').glob('*.jpg')).read_bytes()
    elevation=next((ROOT/'assets/earth-detail').glob('*.png')).read_bytes()
    lock=threading.Lock();stats={'active':0,'maximum':0,'started':None,'finished':None,'requests':0}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            with lock:
                stats['active']+=1;stats['maximum']=max(stats['maximum'],stats['active'])
                if stats['started'] is None:stats['started']=time.monotonic()
            time.sleep(.12)
            body=photo if self.path.startswith('/imagery/') else elevation
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
            with lock:
                stats['active']-=1;stats['requests']+=1;stats['finished']=time.monotonic()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-tile-benchmark-') as directory:
            env={**os.environ,'__NV_PRIME_RENDER_OFFLOAD':'1','__GLX_VENDOR_LIBRARY_NAME':'nvidia',
                 'ATLAS_TILE_BASE_URL':f'http://127.0.0.1:{server.server_port}',
                 'ATLAS_TILE_CACHE_DIR':directory,'ATLAS_TILE_WORKERS':str(workers),
                 'ATLAS_INFERENCE_URL':'http://127.0.0.1:9','ATLAS_STREAM_WARMUP':'20','ATLAS_STREAM_TILES':'1',
                 'ATLAS_TEST_LOCATION':'40,-105','ATLAS_TEST_ZOOM':'160'}
            env.pop('ATLAS_FORECAST',None);env.pop('ATLAS_FOCUS_SPECIES',None)
            result=subprocess.run([str(ROOT/'build/wildlife_atlas'),'--smoke-test'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=45)
            if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        with lock:
            return {'workers':workers,'maximum_concurrent_requests':stats['maximum'],'requests':stats['requests'],
                    'load_seconds':stats['finished']-stats['started'] if stats['started'] else None}
    finally:server.shutdown();server.server_close()

if __name__=='__main__':
    single=run(1);parallel=run(4)
    assert single['maximum_concurrent_requests']==1,single
    assert 2<=parallel['maximum_concurrent_requests']<=4,parallel
    assert single['requests']==parallel['requests'],(single,parallel)
    assert parallel['load_seconds']<single['load_seconds']*.8,(single,parallel)
    print(json.dumps({'single':single,'parallel':parallel,'controlled_speedup':single['load_seconds']/parallel['load_seconds']},indent=2))
