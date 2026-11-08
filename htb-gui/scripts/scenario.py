"""Explicit model-only rolling scenarios, initialized from real tracked identities.

Predicted/interpolated states are never written into the observation archive.
The scenario clock reanchors seed histories; source timestamps remain provenance.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from cohort_forecast import track_metadata, utc


def positions_at(track, days):
    """Interpolate the followed model route on the sphere, including its origin."""
    origin = track["origin"]
    coordinates = np.asarray([[origin["lat"], origin["lon"]], *track["sampled_latlon_paths"][track["representative_sample"]]])
    knots = np.r_[0., track["horizons_days"]]
    days = np.asarray(days, dtype=float)
    ix = np.clip(np.searchsorted(knots, days, side="right"), 1, len(knots)-1)
    t = np.clip((days-knots[ix-1])/(knots[ix]-knots[ix-1]), 0, 1)
    lat, lon = np.deg2rad(coordinates).T
    vectors = np.column_stack([np.cos(lat)*np.cos(lon), np.sin(lat), np.cos(lat)*np.sin(lon)])
    value = vectors[ix-1]*(1-t[:,None])+vectors[ix]*t[:,None]
    length = np.linalg.norm(value, axis=1)
    value[length < 1e-10] = vectors[ix-1][length < 1e-10]
    value /= np.maximum(np.linalg.norm(value, axis=1)[:,None], 1e-12)
    return np.rad2deg(np.column_stack([np.arctan2(value[:,1], np.hypot(value[:,0],value[:,2])), np.arctan2(value[:,2],value[:,0])]))


class Scenarios:
    step_days = 30

    def __init__(self, engine):
        self.engine = engine
        self.states = OrderedDict()

    def start(self, dataset, issue_time=None, samples=16, seed=2026, progress=lambda *args: None):
        issue = utc(issue_time) if issue_time else pd.Timestamp.now(tz="UTC").floor("s")
        selected, sources = [], {}
        frame = dataset.frame.copy()
        frame["lat"] = pd.to_numeric(frame.lat, errors="coerce")
        frame["lon"] = pd.to_numeric(frame.lon, errors="coerce")
        frame = frame.loc[frame.lat.between(-90,90) & frame.lon.between(-180,180)]
        groups = frame.groupby("animal", sort=True)
        excluded = 0
        for i, (animal, rows) in enumerate(groups):
            if rows.species.nunique()!=1 or (rows.groupby("timestamp")[["lat","lon"]].nunique()>1).any().any():
                excluded += 1
                continue
            rows = rows.sort_values("timestamp", kind="stable").drop_duplicates("timestamp").reset_index(drop=True)
            sources[str(animal)] = track_metadata(animal, rows)["origin"]
            # Preserve the exact original lag features and recent real positions,
            # without retaining millions of irrelevant rows in each live session.
            times = pd.DatetimeIndex(rows.timestamp).as_unit("ns").asi8/1e9
            indices, mask = self.engine.schema.past_indices(times, len(rows)-1)
            recent = np.flatnonzero(times>=times[-1]-96*3600)
            recent = recent[np.linspace(0,len(recent)-1,min(64,len(recent)),dtype=int)]
            keep = np.unique(np.r_[0, indices[mask], recent, len(rows)-1])
            rows = rows.iloc[keep].copy().reset_index(drop=True)
            rows["timestamp"] += issue - rows.timestamp.iloc[-1]
            selected.append((animal, rows))
            progress("Seeding scenario from real tracked animals", i+1, len(groups))
        if not selected:
            raise ValueError("No unambiguous real animal histories can seed this scenario")
        state = dict(id=uuid.uuid4().hex, dataset=dataset, issue=issue, start=issue, generation=0,
                     selected=selected, sources=sources, samples=int(samples), seed=int(seed), excluded=excluded)
        result = self._predict(state, progress)
        self.states[state["id"]] = state
        while len(self.states)>2:
            self.states.popitem(last=False)
        return result

    def advance(self, scenario_id, generation, progress=lambda *args: None):
        if scenario_id not in self.states:
            raise ValueError("Scenario session expired. Start a new scenario in this service.")
        state = self.states[scenario_id]
        if int(generation)!=state["generation"]:
            raise ValueError("Scenario generation changed; reload its latest result before advancing")
        issue = state["issue"]+pd.Timedelta(days=self.step_days)
        from trajectory_math import advance as advance_arrays
        rows=state.get('array_rows')
        if rows is None:
            rows=[(pd.DatetimeIndex(frame.timestamp).as_unit('ns').asi8/1e9,frame[['lat','lon']].to_numpy(dtype=float))
                  for _,frame in state['selected']]
        templates=state.get('templates')
        if templates is None:
            omitted={'sampled_latlon_paths','observed_history','scenario_history','horizons_days','forecast_timestamps','origin'}
            templates=[{k:v for k,v in t.items() if k not in omitted} for t in state['result']['tracks']]
        followed=np.asarray([t['sampled_latlon_paths'][t['representative_sample']] for t in state['result']['tracks']])
        rows=advance_arrays(rows,followed,self.engine.horizons,state['issue'].timestamp(),self.engine.schema.LAGS,self.step_days)
        next_state={**state,'issue':issue,'generation':state['generation']+1,'array_rows':rows,'templates':templates,'selected':[]}
        result=self._predict(next_state,progress)
        self.states[scenario_id]=next_state
        return result

    def _predict(self, state, progress):
        prepared=(state['issue'],state['selected'],{'conflicting_identity':state['excluded']})
        if 'array_rows' in state:
            prepared=dict(cutoff=state['issue'],arrays=state['array_rows'],templates=state['templates'],
                          excluded={'conflicting_identity':state['excluded']})
        result = self.engine.forecast(state["dataset"], samples=state["samples"],
                    seed=state["seed"]+state["generation"], progress=progress,
                    prepared=prepared)
        for track in result["tracks"]:
            track["scenario_history"] = track.pop("observed_history")
            track["observed_history"] = []
            track["seed_observation"] = state["sources"][track["id"]]
            track["history_kind"] = "time-reanchored seed history" if state["generation"]==0 else "projected scenario state"
        result["retrospective_replay"] = False
        result["scenario_adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["scenario"] = dict(id=state["id"], generation=state["generation"], start_time=state["start"].isoformat(),
             step_days=self.step_days, elapsed_days=state["generation"]*self.step_days,
             seed_policy="Latest actual history per recorded animal, time-reanchored to the scenario start; not live locations",
             history_kind="Predicted positions become inputs to subsequent model forecasts; interpolation supplies intermediate states",
             accuracy="Recursive scenario beyond 90 days is not validated by this checkpoint")
        result["scope"] = "Continuously projected model scenario. Real source identities; no generated animals or current-location claim."
        state["result"] = result
        return result
