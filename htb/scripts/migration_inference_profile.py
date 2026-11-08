"""Time frozen checkpoint inference on the current GPU using past inputs only."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--repeats", type=int, default=8)
    args = p.parse_args()
    root = Path(args.bundle).resolve()
    output = Path(args.output).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from wildcast.migration_model import load_model as load_single
    from wildcast.migration_ensemble import load_model as load_ensemble
    from wildcast.migration_evaluate import forecast_latlon
    catalogue = json.loads((root / "package-manifest.json").read_text())
    single = next(root / r["path"] for r in catalogue["checkpoints"]
                  if r["source_path"].endswith("scaled-history_direct/best.pt"))
    with np.load(root / "data/reference/cases.npz", allow_pickle=False) as f:
        species, split = f["species"], f["split"]
        taxa = np.unique(species[split == 0])
        chosen = np.asarray([np.flatnonzero((species == s) & (split == 0))[0] for s in taxa])
        a = {k: f[k][chosen] for k in ["history", "history_mask", "static", "species", "origin"]}
        a["horizons"] = f["horizons"]
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    records = []
    for kind, path, loader in [("standalone17M", single, load_single),
                               ("four_member_ensemble", root / "model/selected.pt", load_ensemble)]:
        start = time.perf_counter()
        model, ck = loader(path, device=args.device)
        del ck
        gc.collect()
        torch.cuda.synchronize()
        loaded = time.perf_counter() - start
        for count in [1, 108, 1024]:
            index = np.arange(count) % len(chosen)
            b = {k: v if k == "horizons" else v[index] for k, v in a.items()}
            ids = np.arange(count)
            if kind == "standalone17M":
                inputs = [torch.as_tensor(b[k], device=args.device) for k in
                          ["history", "history_mask", "static", "species"]]
                def run():
                    with torch.inference_mode():
                        return model.sample_paths(*inputs, 64).cpu().numpy()
            else:
                def run():
                    return model.predict_cases(b, ids, 64, np.random.default_rng(2026))
            result = run()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times, geographic = [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = run()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                start = time.perf_counter()
                ll = forecast_latlon(result, b["origin"])
                geographic.append(time.perf_counter() - start)
            assert result.shape == (count, 64, 9, 2) and np.isfinite(ll).all()
            median = float(np.median(times))
            records.append({"model": kind, "animals": count, "samples": 64, "horizons": 9,
                "median_model_seconds": median, "median_coordinate_conversion_seconds": float(np.median(geographic)),
                "median_model_plus_coordinates_seconds": float(np.median(np.asarray(times) + geographic)),
                "animal_forecasts_per_second": count / median,
                "peak_torch_allocated_MiB": torch.cuda.max_memory_allocated() / 1024**2,
                "load_seconds": loaded})
        del model, result, run, b
        if kind == "standalone17M":
            del inputs
        gc.collect()
        torch.cuda.empty_cache()
    report = {"scope": "Inference cost only; one prepared past example per108 training labels; no target arrays read",
              "hardware": torch.cuda.get_device_name(), "torch": torch.__version__,
              "matmul_precision": "high", "samples_per_track": 64, "horizons": 9,
              "timing_limits": "Warm inputs/model; excludes checkpoint loading and raw-track parsing/feature construction",
              "package_manifest": catalogue["checkpoint"]["sha256"], "records": records}
    out = output
    if out.exists():
        raise FileExistsError("Use a new immutable inference profile")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
