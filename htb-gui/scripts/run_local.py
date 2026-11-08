#!/usr/bin/env python3
"""Build the native atlas and run its owned local inference service."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def health(url):
    try:
        with urllib.request.urlopen(url + '/health', timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--no-build', action='store_true')
    p.add_argument('--python', default=str(ROOT.parent / 'htb/.venv/bin/python'))
    p.add_argument('--bundle',default=str(ROOT/'models/server-repaired-v2'),help='Verified local model deployment')
    p.add_argument('--focus', help='Open a recorded species on the particle map, e.g. Ciconia ciconia')
    p.add_argument('--zoom', type=float, default=80, help='Initial map zoom for --focus (maximum 160)')
    p.add_argument('--renderer', choices=['dots','legacy'], default='dots', help='Sand-like particle map; legacy is a compatibility alias')
    args, gui_args = p.parse_known_args()
    if not Path(args.python).is_file() and not shutil.which(args.python):
        p.error('Stray model Python was not found. Pass --python /path/to/your/cuda/python.')
    binary=ROOT / 'build/wildlife_atlas'
    if not args.no_build:
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release'], check=True)
        subprocess.run(['cmake', '--build', str(ROOT / 'build'), '--parallel', '6'], check=True)
    if not binary.is_file():
        p.error(f'{binary} is not built. Run ./run-local without --no-build once.')
    base = f'http://127.0.0.1:{args.port}'
    server = None
    running = health(base)
    if running and running.get('service') != 'wildlife-atlas-wildcast-v1':
        p.error('Port is occupied by a different service; select another --port.')
    if running and (running.get('api_version',0)<2 or running.get('scenario_api_version',0)<1 or Path(running.get('bundle','')).resolve()!=Path(args.bundle).resolve()):
        p.error('An older/different model service is running; close it or choose another --port.')
    if running and args.device != 'auto' and running.get('device') != args.device:
        p.error('Existing service uses a different device; choose another --port.')
    try:
        if not running:
            server = subprocess.Popen([args.python, str(ROOT / 'scripts/inference_server.py'), '--device', args.device, '--port', str(args.port),'--bundle',args.bundle], cwd=ROOT)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Inference service exited; see its error above.')
                running = health(base)
                if running:
                    break
                time.sleep(.2)
            else:
                raise RuntimeError('Inference service did not become ready in 60 seconds.')
        print(f"Stray: {running['device']} | {running.get('gpu') or 'CPU'}", flush=True)
        print(f"Renderer: {args.renderer} | {binary}", flush=True)
        env = {**os.environ, 'ATLAS_INFERENCE_URL': base}
        env['ATLAS_RENDERER']=args.renderer
        if args.focus:
            env.update(ATLAS_FOCUS_SPECIES=args.focus, ATLAS_TEST_ZOOM=str(args.zoom))
        if args.device != 'cpu' and shutil.which('nvidia-smi'):
            env.update(__NV_PRIME_RENDER_OFFLOAD='1', __GLX_VENDOR_LIBRARY_NAME='nvidia')
        return subprocess.call([str(binary), *gui_args], cwd=ROOT, env=env)
    finally:
        if server:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
