"""Compare optimized features with the frozen encoder on real saved histories."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'models/server-repaired-v2'))
from trajectory_math import encode,advance
from wildcast.migration_protocol import schema_for


class TrajectoryMathTests(unittest.TestCase):
    def test_real_histories_and_projected_states_match_the_exact_encoder(self):
        manifest=json.loads((ROOT/'models/server-repaired-v2/data/reference/manifest.json').read_text())
        schema=schema_for(manifest,legacy_default=True)
        forecast=json.loads((ROOT/'runs/latest-scenario.json').read_text())
        tracks=forecast['tracks'][:32]
        rows=[]
        for track in tracks:
            history=track['scenario_history']
            rows.append((pd.to_datetime([f['timestamp'] for f in history],utc=True).as_unit('ns').asi8/1e9,
                         np.asarray([[f['lat'],f['lon']] for f in history])))
        followed=np.asarray([t['sampled_latlon_paths'][t['representative_sample']] for t in tracks])
        horizons=np.asarray(tracks[0]['horizons_days'])
        for step in (0,30,90):
            current=rows if not step else advance(rows,followed,horizons,rows[0][0][-1],schema.LAGS,step)
            h,mask,static,_=encode(current,schema)
            for i,(times,coords) in enumerate(current):
                exact=schema.query_features(coords[:,0],coords[:,1],times,pd.to_datetime(times,unit='s',utc=True),len(times)-1)
                np.testing.assert_array_equal(mask[i],exact[1])
                np.testing.assert_allclose(h[i],exact[0],rtol=2e-6,atol=2e-6)
                np.testing.assert_allclose(static[i],exact[2],rtol=2e-6,atol=2e-6)
            if step:
                boundary=followed[:,list(horizons).index(float(step))]
                np.testing.assert_allclose(np.asarray([c[-1] for _,c in current]),boundary,rtol=0,atol=1e-7)


if __name__=='__main__':unittest.main()
