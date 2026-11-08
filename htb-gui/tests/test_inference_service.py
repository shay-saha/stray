"""Integration checks against the real local model, not a fake inference response.
Run with the service listening on 127.0.0.1:8765.
"""
import json
import os
import time
import tempfile
import subprocess
from pathlib import Path
import unittest
import urllib.error
import urllib.request

BASE = os.environ.get('ATLAS_INFERENCE_URL','http://127.0.0.1:8765')

def request(path, data=None):
    payload = None if data is None else json.dumps(data).encode()
    with urllib.request.urlopen(urllib.request.Request(BASE+path, data=payload,
            headers={'Content-Type':'application/json'}), timeout=180) as response:
        return json.load(response)

def job_result(path, payload):
    job=request(path,payload)
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        status=request('/jobs/'+job['job_id'])
        if status['state']=='failed':raise AssertionError(status['error'])
        if status['state']=='complete':return status['result']
        time.sleep(.05)
    raise AssertionError('GPU job did not finish')

def real_pair(path):
    source=json.loads((Path(__file__).resolve().parents[1]/'runs/latest-cohort-forecast.json').read_text())
    tracks=source['tracks'][:2]
    import pandas as pd
    rows=[dict(row,animal=track['id'],species=track['species'],study=track['study'])
          for track in tracks for row in track['observed_history']]
    pd.DataFrame(rows).to_csv(path,index=False)
    return tracks

class InferenceServiceTests(unittest.TestCase):
    def test_scenario_rolls_into_next_year_without_new_animals_or_observations(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.csv'
            real=real_pair(path)
            original=path.read_bytes()
            result=job_result('/scenario',dict(observations=str(path),issue_time='2024-12-15T00:00:00Z',samples=2))
            expected_ids={t['id'] for t in real}
            for generation in range(14):
                self.assertEqual(result['scenario']['generation'],generation)
                self.assertEqual({t['id'] for t in result['tracks']},expected_ids)
                self.assertFalse(result['retrospective_replay'])
                self.assertTrue(all(not t['observed_history'] and t['scenario_history'] and t['seed_observation'] for t in result['tracks']))
                if generation==13:break
                previous={t['id']:t['sampled_latlon_paths'][t['representative_sample']][t['horizons_days'].index(30.)] for t in result['tracks']}
                result=job_result('/scenario/advance',dict(scenario_id=result['scenario']['id'],generation=generation))
                for track in result['tracks']:
                    np.testing.assert_allclose([track['origin']['lat'],track['origin']['lon']],previous[track['id']],rtol=0,atol=1e-7)
            self.assertTrue(result['issue_time'].startswith('2026-'))
            self.assertEqual(path.read_bytes(),original,'Predicted positions must never become recorded observations')

    def test_native_viewer_continues_a_gpu_scenario_at_month_boundary(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.csv';real_pair(path)
            result=job_result('/scenario',dict(observations=str(path),issue_time='2024-12-15T00:00:00Z',samples=2))
            saved=Path(directory)/'scenario.json';saved.write_text(json.dumps(result))
            env={**os.environ,'ATLAS_INFERENCE_URL':BASE,'ATLAS_FORECAST':str(saved),
                '__NV_PRIME_RENDER_OFFLOAD':'1','__GLX_VENDOR_LIBRARY_NAME':'nvidia',
                'ATLAS_TEST_FORECAST_DAY':'29.9999','ATLAS_TEST_AUTOPLAY':'1',
                'ATLAS_TEST_SCENARIO_GENERATION':'1','ATLAS_STREAM_WARMUP':'3',
                'ATLAS_SCREENSHOT':'/tmp/atlas-scenario-continuation.ppm'}
            checked=subprocess.run([str(root/'build/wildlife_atlas'),'--smoke-test','forecast'],cwd=root,env=env,
                                   capture_output=True,text=True,timeout=60)
            self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
            self.assertIn('Observed animals: 2',checked.stdout)

    def test_health_identifies_verified_model_and_device(self):
        h = request('/health')
        self.assertEqual(h['service'], 'wildlife-atlas-wildcast-v1')
        self.assertEqual(h['model_kind'], 'wildcast_migration_joint')
        self.assertGreater(h['verified_files'], 0)
        self.assertIn(h['device'], ['cuda', 'cpu'])
        if h['device'] == 'cuda':
            self.assertTrue(h['gpu'])

    def test_real_observations_produce_finite_joint_forecast(self):
        demo = request('/demo')
        f = request('/forecast', {k:demo[k] for k in ['observations','species','animal']} | {'samples':2})
        self.assertEqual(len(f['sampled_latlon_paths']), 2)
        self.assertEqual(f['species'], demo['species'])
        self.assertEqual(f['horizons_days'],request('/health')['horizons_days'])
        self.assertGreaterEqual(request('/health')['trained_taxa'],100)
        self.assertTrue(f['retrospective_replay'])
        self.assertEqual(f['runtime']['device'], request('/health')['device'])
        for path in f['sampled_latlon_paths']:
            self.assertEqual(len(path), len(f['horizons_days']))
            for lat,lon in path:
                self.assertTrue(-90 <= lat <= 90 and -180 <= lon <= 180)

    def test_bad_request_returns_actionable_error(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            request('/forecast', {'observations':'/no/such/history.csv', 'species':'Ciconia ciconia'})
        self.assertEqual(caught.exception.code, 400)
        self.assertTrue(json.load(caught.exception)['error'])

    def test_sample_bound_is_enforced(self):
        demo = request('/demo')
        with self.assertRaises(urllib.error.HTTPError) as caught:
            request('/forecast', {k:demo[k] for k in ['observations','species','animal']} | {'samples':100000})
        self.assertEqual(caught.exception.code,400)
        self.assertIn('128',json.load(caught.exception)['error'])

    def test_two_real_observed_animals_produce_two_forecasts(self):
        source=json.loads((Path(__file__).resolve().parents[1]/'runs/latest-cohort-forecast.json').read_text())
        tracks=source['tracks'][:2]
        import pandas as pd
        rows=[dict(row,animal=track['id'],species=track['species'],study=track['study'])
              for track in tracks for row in track['observed_history']]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.csv';pd.DataFrame(rows).to_csv(path,index=False)
            job=request('/forecast-cohorts',{'observations':str(path),'issue_time':source['issue_time'],'samples':2,'max_age_hours':24})
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                status=request('/jobs/'+job['job_id'])
                if status['state']=='failed':self.fail(status['error'])
                if status['state']=='complete':break
                time.sleep(.1)
            else:self.fail('Model job did not finish')
            result=status['result']
            self.assertEqual({t['id'] for t in result['tracks']},{t['id'] for t in tracks})
            self.assertEqual(len(result['tracks']),2)
            self.assertTrue(all(len(t['sampled_latlon_paths'])==2 for t in result['tracks']))
            self.assertEqual(result['checkpoint_sha256'],request('/health')['checkpoint_sha256'])

if __name__ == '__main__':
    unittest.main()
