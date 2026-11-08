"""Native import checks using existing real model output as a small fixture."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class ArchiveImportTests(unittest.TestCase):
    def fixture(self,directory):
        source=json.loads((ROOT/'runs/latest-scenario.json').read_text())
        tracks=source['tracks'][:2]
        metadata=[{k:t[k] for k in ('id','species','name','cohort_id','cohort_label')} for t in tracks]
        coords=np.asarray([[[t['origin']['lat'],t['origin']['lon']],
                          *t['sampled_latlon_paths'][t['representative_sample']]] for t in tracks],dtype='<f4').transpose(1,0,2)
        binary=directory/'positions.f32';binary.write_bytes(coords.tobytes())
        days=[0,*tracks[0]['horizons_days']]
        manifest=dict(schema='wildlife-atlas-trajectory-archive-v1',kind='forecast',issue_time=source['issue_time'],
            forecast_end_days=days[-1],years=0,animals=metadata,times_days=days,
            positions=dict(file=binary.name,dtype='float32-le',shape=list(coords.shape)),
            dataset=dict(species_catalog=sorted({t['species'] for t in tracks}),recorded_animals=2,recorded_species=1))
        path=directory/'forecast.json';path.write_text(json.dumps(manifest))
        return path,binary,manifest

    def run_viewer(self,path):
        env={**os.environ,'ATLAS_FORECAST':str(path),'ATLAS_INFERENCE_URL':'http://127.0.0.1:9',
             '__NV_PRIME_RENDER_OFFLOAD':'1','__GLX_VENDOR_LIBRARY_NAME':'nvidia'}
        return subprocess.run([str(ROOT/'build/wildlife_atlas'),'--smoke-test','forecast'],cwd=ROOT,
                              env=env,capture_output=True,text=True,timeout=30)

    def test_imported_predictions_play_without_a_live_service(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_,_=self.fixture(Path(directory))
            result=self.run_viewer(path)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('Observed animals: 2',result.stdout)
            self.assertIn('Scenario generation: -1',result.stdout)

    def test_truncated_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,binary,_=self.fixture(Path(directory))
            binary.write_bytes(binary.read_bytes()[:-1])
            self.assertNotEqual(self.run_viewer(path).returncode,0)

    def test_duplicate_recorded_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_,manifest=self.fixture(Path(directory))
            manifest['animals'][1]['id']=manifest['animals'][0]['id']
            path.write_text(json.dumps(manifest))
            self.assertNotEqual(self.run_viewer(path).returncode,0)


if __name__=='__main__':unittest.main()
