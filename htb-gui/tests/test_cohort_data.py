"""Test-only fixtures verify causal filtering; the app never creates these animals."""
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cohort_forecast import Observations

class ObservationTests(unittest.TestCase):
    def dataset(self,rows):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        path=Path(directory.name)/'test.csv'
        pd.DataFrame(rows,columns=['timestamp','animal','species','lat','lon','study']).to_csv(path,index=False)
        return Observations(path)

    def test_future_rows_never_create_present_animals_or_change_past_species(self):
        o=self.dataset([('2020-01-01T00:00:00Z','a','Ciconia ciconia',45,7,'study'),
                        ('2020-01-01T01:00:00Z','a','Ciconia ciconia',45.1,7.1,'study'),
                        ('2021-01-01T00:00:00Z','a','Bad future label','invalid','invalid','study'),
                        ('2021-01-01T00:00:00Z','future-only','Ciconia ciconia',40,4,'study')])
        p=o.preview('2020-01-01T02:00:00Z')
        self.assertEqual([t['id'] for t in p['tracks']],['a'])
        self.assertEqual(p['tracks'][0]['species'],'Ciconia ciconia')
        self.assertEqual(p['tracks'][0]['origin']['lat'],45.1)
        self.assertNotIn('sampled_latlon_paths',p['tracks'][0])

    def test_stale_histories_are_explicitly_excluded(self):
        o=self.dataset([('2020-01-01T00:00:00Z','a','Ciconia ciconia',45,7,'study')])
        p=o.preview('2020-01-05T00:00:00Z',max_age_hours=24)
        self.assertEqual(p['tracks'],[])
        self.assertEqual(p['excluded']['stale'],1)

    def test_conflicting_identical_times_do_not_invent_a_location(self):
        o=self.dataset([('2020-01-01T00:00:00Z','a','Ciconia ciconia',45,7,'study'),
                        ('2020-01-01T00:00:00Z','a','Ciconia ciconia',46,8,'study')])
        p=o.preview('2020-01-01T01:00:00Z')
        self.assertEqual(p['tracks'],[])
        self.assertEqual(p['excluded']['conflicting_identity'],1)

    def test_more_than_512_observed_identities_are_retained(self):
        o=self.dataset([('2020-01-01T00:00:00Z',f'test-tag-{i}','Ciconia ciconia',45,7,'study') for i in range(600)])
        p=o.preview('2020-01-01T01:00:00Z')
        self.assertEqual(len(p['tracks']),600)
        self.assertEqual(len({t['id'] for t in p['tracks']}),600)
        self.assertEqual(len({t['cohort_id'] for t in p['tracks']}),1)
