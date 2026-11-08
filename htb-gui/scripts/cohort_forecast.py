"""Observed-animal cohorts and batched inference through the verified WildCast package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

SCHEMA = "wildlife-atlas-cohorts-v2"


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Choose a valid forecast issue time")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class Observations:
    @staticmethod
    def paths(path):
        path = Path(path).expanduser().resolve(strict=True)
        if path.suffix.lower() != ".json":
            return [path]
        config = json.loads(path.read_text())
        return [(path.parent / source).resolve(strict=True) for source in config["sources"]]

    @classmethod
    def signature(cls, path):
        return tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in [Path(path).resolve(),*cls.paths(path)])

    def __init__(self, path, species_override=None):
        self.path = Path(path).expanduser().resolve(strict=True)
        before = self.signature(self.path)
        frames, self.sources = [], []
        for source in self.paths(self.path):
            if source.suffix.lower() in {".parquet", ".pq"}:
                import pyarrow.parquet as parquet
                columns=parquet.ParquetFile(source).schema.names
                part = pd.read_parquet(source,columns=[c for c in ("timestamp","lat","lon","animal","species","study","colony_id","cohort_id") if c in columns])
            elif source.suffix.lower() == ".csv":
                part = pd.read_csv(source,dtype="string")
            else:
                raise ValueError("Choose CSV/parquet observations or a JSON source collection")
            frames.append(part)
            self.sources.append({"path":str(source),"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"rows":len(part)})
        self.sha256 = hashlib.sha256(json.dumps(self.sources,sort_keys=True).encode()).hexdigest()
        frame = pd.concat(frames,ignore_index=True)
        if species_override is not None and "species" not in frame:
            frame["species"] = species_override
        if species_override is not None and "animal" not in frame:
            frame["animal"] = "observed-animal"
        required = {"timestamp", "lat", "lon", "animal", "species"}
        if not required.issubset(frame):
            raise ValueError("Cohort observations require timestamp, lat, lon, animal and scientific species columns")
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce").dt.as_unit("ns")
        frame = frame.loc[frame.timestamp.notna() & frame.animal.notna() & frame.species.notna()].copy()
        frame["animal"] = frame.animal.astype(str)
        frame["species"] = frame.species.astype(str)
        if "study" not in frame:
            frame["study"] = self.path.stem
        frame["study"] = frame.study.fillna(self.path.stem).astype(str)
        if frame.empty:
            raise ValueError("No dated animal observations in this table")
        if before != self.signature(self.path):
            raise ValueError("Observation file changed while reading; use a frozen snapshot")
        self.frame = frame
        counts = frame.assign(day=frame.timestamp.dt.floor("D")).drop_duplicates(["animal", "day"])
        coverage=counts.groupby("day").agg(taxa=("species","nunique"),animals=("animal","nunique"))
        busiest = coverage.sort_values(["taxa","animals"],ascending=False,kind="stable").index[0]
        self.default_issue = (busiest + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat()

    def prepare(self, issue_time=None, max_age_hours=24, species=None):
        cutoff = utc(issue_time or self.default_issue)
        age = float(max_age_hours)
        if not np.isfinite(age) or not 1 <= age <= 168:
            raise ValueError("Observation freshness must be 1 to 168 hours")
        # Exclude future rows before coordinate coercion, animal selection or features.
        frame = self.frame.loc[self.frame.timestamp <= cutoff].copy()
        if species:
            frame = frame.loc[frame.species.isin(species)]
        frame["lat"] = pd.to_numeric(frame.lat, errors="coerce")
        frame["lon"] = pd.to_numeric(frame.lon, errors="coerce")
        frame = frame.loc[frame.lat.between(-90, 90) & frame.lon.between(-180, 180)]
        selected, excluded = [], {"stale": 0, "conflicting_identity": 0}
        for animal, rows in frame.groupby("animal", sort=True):
            if rows.timestamp.max() < cutoff - pd.Timedelta(hours=age):
                excluded["stale"] += 1
                continue
            if rows.species.nunique() != 1 or (rows.groupby("timestamp")[["lat", "lon"]].nunique() > 1).any().any():
                excluded["conflicting_identity"] += 1
                continue
            rows = rows.sort_values("timestamp", kind="stable").drop_duplicates("timestamp").reset_index(drop=True)
            selected.append((animal, rows))
        return cutoff, selected, excluded

    def info(self):
        if hasattr(self,"_info"):
            return self._info
        self._info={"observations": str(self.path), "source_sha256": self.sha256,
                "sources": self.sources,
                "recorded_animals": int(self.frame.animal.nunique()),
                "recorded_species": int(self.frame.species.nunique()),
                "species_catalog": sorted(self.frame.species.unique().tolist()),
                "first_observation": self.frame.timestamp.min().isoformat(),
                "last_observation": self.frame.timestamp.max().isoformat(),
                "default_issue_time": self.default_issue}
        return self._info

    def preview(self, issue_time=None, max_age_hours=24, species=None):
        cutoff, selected, excluded = self.prepare(issue_time, max_age_hours, species)
        tracks = [track_metadata(animal, rows) for animal, rows in selected]
        return {"schema": SCHEMA, "kind": "observations", "issue_time": cutoff.isoformat(),
                "dataset": self.info(), "tracks": tracks, "excluded": excluded,
                "max_age_hours": float(max_age_hours), "forecast_end_days": 0,
                "cohort_definition": "Study and scientific species; not verified biological colonies"}


def track_metadata(animal, frame):
    row = frame.iloc[-1]
    # Source identities are used verbatim. Never infer aliases from proximity.
    taxon, study = str(row.species), str(row.study)
    cohort = str(row.get("colony_id", row.get("cohort_id", study)))
    if not cohort or cohort == "nan":
        cohort = study
    recent=frame.loc[frame.timestamp>=row.timestamp-pd.Timedelta(hours=96)]
    recent=recent.iloc[np.linspace(0,len(recent)-1,min(64,len(recent)),dtype=int)]
    history = [{"timestamp": r.timestamp.isoformat(), "lat": float(r.lat), "lon": float(r.lon)}
               for r in recent.itertuples()]
    return {"id": str(animal), "species": taxon, "name": taxon,
            "cohort_id": taxon + "|" + cohort, "cohort_label": cohort,
            "study": study, "origin": history[-1], "observed_history": history,
            "observations": len(frame)}


class CohortEngine:
    def __init__(self, bundle, device):
        import torch
        from wildcast.migration_model import load_model
        from wildcast.migration_protocol import schema_for
        self.bundle, self.device = Path(bundle), device
        manifest_path=self.bundle/"data/reference/manifest.json"
        self.manifest=json.loads(manifest_path.read_text())
        self.hashes={"manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"cases_sha256":self.manifest["cases_sha256"]}
        self.model, metadata=load_model(self.bundle/"model/selected.pt",device)
        self.metadata={key:metadata.get(key) for key in ("model_kind","manifest_sha256","cases_sha256","selection_version")}
        self.kind=metadata["model_kind"]
        del metadata
        self.checkpoint_hash=hashlib.sha256((self.bundle/"model/selected.pt").read_bytes()).hexdigest()
        if any(self.metadata.get(key)!=expected for key,expected in self.hashes.items()):
            raise ValueError("Checkpoint/reference hashes do not match")
        if self.kind != "wildcast_migration_joint":
            raise ValueError("Cohort batching requires the verified joint migration checkpoint")
        self.model.eval()
        self.schema=schema_for(self.manifest,legacy_default=True)
        self.builder=getattr(self.schema,"query_features",self.schema.features)
        self.horizons = np.asarray(self.manifest["horizons_days"], dtype=float)
        self.training_end = pd.Timestamp(int(self.manifest.get("train_end_year",2020))+1,1,1,tz="UTC").timestamp()
        self.code_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('trajectory_math.py').read_bytes()).hexdigest()
        self.cache_dir=Path(__file__).resolve().parents[1]/"runs/cohort-cache"
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.lock = threading.Lock()
        self.datasets = {}

    def dataset(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        key = Observations.signature(path)
        with self.lock:
            if key not in self.datasets:
                value = Observations(path)
                self.datasets.clear()
                self.datasets[key] = value
            return self.datasets[key]

    def forecast(self, dataset, issue_time=None, max_age_hours=24, species=None, samples=16, seed=2026,
                 progress=lambda *args: None, prepared=None):
        import torch
        from wildcast.data import unproject
        count = int(samples)
        if not 2 <= count <= 128:
            raise ValueError("Choose 2 to 128 uncertainty paths per tracked animal")
        arrays=isinstance(prepared,dict)
        if arrays:
            cutoff,selected,excluded=prepared['cutoff'],[],prepared['excluded']
        else:
            cutoff, selected, excluded = prepared if prepared is not None else dataset.prepare(issue_time, max_age_hours, species)
        if not selected and not arrays:
            raise ValueError("No observed animals meet this issue time and freshness window")
        started, tracks = time.monotonic(), []
        histories, masks, statics, taxa = [], [], [], []
        if arrays:
            from trajectory_math import encode
            histories,masks,statics,_=encode(prepared['arrays'],self.schema)
            stamps={}
            for i,((times,coordinates),base) in enumerate(zip(prepared['arrays'],prepared['templates'])):
                track=dict(base)
                keep=np.flatnonzero(times>=times[-1]-96*3600)
                keep=keep[np.linspace(0,len(keep)-1,min(64,len(keep)),dtype=int)]
                history=[]
                for index in keep:
                    when=float(times[index])
                    if when not in stamps:stamps[when]=pd.Timestamp(when,unit='s',tz='UTC').isoformat()
                    history.append(dict(timestamp=stamps[when],lat=float(coordinates[index,0]),lon=float(coordinates[index,1])))
                track.update(origin=history[-1],observed_history=history,history_available_lags=int(masks[i].sum()))
                tracks.append(track);taxa.append(int(self.manifest['species'].get(track['species'],0)))
            progress('Encoded scenario state',len(tracks),len(tracks))
        for i, (animal, frame) in enumerate(selected):
            dates = pd.DatetimeIndex(frame.timestamp).as_unit("ns")
            times = dates.asi8 / 1e9
            history, mask, static, _, _ = self.builder(frame.lat.to_numpy(), frame.lon.to_numpy(), times, dates, len(frame)-1)
            if history.shape != (self.model.config["history_length"],self.model.config["history_features"]) or static.shape != (self.model.config["static_features"],):
                raise ValueError("Observed feature schema does not match this checkpoint")
            histories.append(history); masks.append(mask); statics.append(static)
            taxon = str(frame.species.iloc[-1])
            taxa.append(int(self.manifest["species"].get(taxon, 0)))
            track = track_metadata(animal, frame)
            track["unknown_species_fallback"] = taxa[-1] == 0
            track["history_support_matches_training"] = bool(mask.sum() >= self.manifest.get("minimum_history_lags",5)
                and len(frame)-1 >= self.manifest.get("minimum_history_prior_fixes",8)
                and (dates[-1]-dates[0]).total_seconds() >= self.manifest.get("minimum_history_span_days",7)*86400)
            track["history_available_lags"] = int(mask.sum())
            tracks.append(track)
            progress("Preparing observed histories", i+1, len(selected))
        encoded=time.monotonic()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        batch_size=256 if self.device=='cuda' else 64
        timestamp_cache={}
        for first in range(0, len(tracks), batch_size):
            last = min(first+batch_size, len(tracks))
            with torch.inference_mode():
                inputs = [torch.as_tensor(np.asarray(x[first:last]), device=self.device)
                          for x in (histories, masks, statics, taxa)]
                paths = self.model.sample_paths(*inputs, samples=count, generator=generator).detach().cpu().numpy()
            if paths.shape != (last-first, count, len(self.horizons), 2) or not np.isfinite(paths).all():
                raise ValueError("Model returned invalid joint route samples")
            origins=np.asarray([[t['origin']['lat'],t['origin']['lon']] for t in tracks[first:last]])
            coordinates=unproject(paths*1_000_000,origins[:,0,None,None],origins[:,1,None,None])
            if not np.isfinite(coordinates).all():raise ValueError('Model coordinates are non-finite')
            lat,lon=np.deg2rad(coordinates[...,0]),np.deg2rad(coordinates[...,1])
            vectors=np.stack([np.cos(lat)*np.cos(lon),np.sin(lat),np.cos(lat)*np.sin(lon)],axis=-1)
            center=vectors.mean(axis=1);center/=np.maximum(np.linalg.norm(center,axis=-1,keepdims=True),1e-12)
            representatives=np.argmin((1-(vectors*center[:,None]).sum(axis=-1)).sum(axis=-1),axis=1)
            for offset, track in enumerate(tracks[first:last]):
                origin = track["origin"]
                if origin['timestamp'] not in timestamp_cache:
                    stamp=utc(origin['timestamp'])
                    timestamp_cache[origin['timestamp']]=[(stamp+pd.Timedelta(days=float(h))).isoformat() for h in self.horizons]
                track.update(horizons_days=self.horizons.tolist(), sampled_latlon_paths=coordinates[offset].tolist(),
                    representative_sample=int(representatives[offset]),forecast_timestamps=timestamp_cache[origin['timestamp']])
            progress("Predicting tracked animals on " + self.device, last, len(tracks))
        end = min((utc(t["forecast_timestamps"][-1])-cutoff).total_seconds()/86400 for t in tracks)
        result={"schema": SCHEMA, "kind": "forecast", "issue_time": cutoff.isoformat(), "tracks": tracks,
                "dataset": dataset.info(), "excluded": excluded, "max_age_hours": float(max_age_hours),
                "forecast_end_days": end, "samples_per_animal": count, "seed": int(seed),
                "model_kind": self.kind, "checkpoint_sha256": self.checkpoint_hash, **self.hashes,
                "feature_code_sha256": hashlib.sha256(Path(self.schema.__file__).read_bytes()).hexdigest(),
                "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "retrospective_replay": bool(cutoff.timestamp() < max(self.training_end, time.time())),
                "cohort_definition": "Study and scientific species; not verified biological colonies",
                "scope": "One forecast per observed animal; uncertainty samples are not extra animals",
                "runtime": {"device": self.device, "inference_seconds": time.monotonic()-started,
                            "animals": len(tracks), "batch_size": batch_size,"cache_hit":False,
                            "feature_seconds":encoded-started,"sampling_seconds":time.monotonic()-encoded,
                            "array_history":arrays}}
        if (prepared is None and not species and count==16 and int(seed)==2026 and float(max_age_hours)==24
                and cutoff==utc(dataset.default_issue)):
            entry={"signature":self.source_signature(dataset.path),"checkpoint":self.checkpoint_hash,"code":self.code_sha256,"result":result}
            temp=self.cache_dir/"default.partial.json"
            temp.write_text(json.dumps(entry,allow_nan=False,separators=(',',':')))
            temp.replace(self.cache_dir/"default.json")
        return result

    def source_signature(self,path):
        return hashlib.sha256(json.dumps(Observations.signature(path)).encode()).hexdigest()

    def cached_default(self,path):
        try:
            entry=json.loads((self.cache_dir/"default.json").read_text())
            if (entry['signature']==self.source_signature(path) and entry['checkpoint']==self.checkpoint_hash and entry['code']==self.code_sha256):
                result=entry['result'];result['runtime']['cache_hit']=True
                return result
        except (OSError,ValueError,KeyError):
            pass
        return None
