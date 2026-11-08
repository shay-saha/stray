from __future__ import annotations

import argparse
import json


def main():
    p = argparse.ArgumentParser(description="WildCast data and model pipeline")
    p.add_argument("--data", default="data", help="Data root (place before subcommand)")
    sub = p.add_subparsers(dest="command", required=True)
    mp = sub.add_parser("migration-prepare")
    mp.add_argument("--roots", nargs="+", required=True)
    mp.add_argument("--output", required=True)
    mp.add_argument("--identity-audit", default="reports/identity-audit.json")
    mp.add_argument("--existing-manifest", default="data/combined-landscape/processed/windows.json")
    mp.add_argument("--stride-days", type=float, default=7.)
    mp.add_argument("--train-end-year", type=int, default=2020)
    mp.add_argument("--validation-year", type=int, default=2021)
    mp.add_argument("--seed", type=int, default=1701)
    mp.add_argument("--reference-audit")
    mg = sub.add_parser("migration-train")
    mg.add_argument("--output", required=True)
    mg.add_argument("--variant", choices=["seasonal_static", "history_direct", "goal_route", "residence_route"], default="goal_route")
    for name, value in [("epochs", 40), ("batch-size", 128), ("width", 96), ("layers", 2),
                        ("components", 6), ("seed", 42), ("patience", 8), ("energy-repeats", 2),
                        ("validation-repeats", 8), ("max-train", 0)]:
        mg.add_argument("--" + name, type=int, default=value)
    for name, value in [("lr", .0003), ("nll-weight", .1), ("energy-weight", 1.),
                        ("weight-decay", .01), ("dropout", .1), ("species-dropout", .1)]:
        mg.add_argument("--" + name, type=float, default=value)
    mg.add_argument("--device", default="auto")
    mg.add_argument("--validation-samples", type=int, default=None)
    mg.add_argument("--sampling", choices=["animal", "species_animal", "species_sqrt"], default="animal")
    mg.add_argument("--init-checkpoint")
    mg.add_argument("--case-filter")
    mg.add_argument("--min-scale", type=float, default=.0005,
                    help="Numerical marginal-scale floor in 1000 km units; not measured sensor error")
    mg.add_argument("--covariance-jitter", type=float, default=None,
                    help="Optional variance floor in (1000 km)^2; historical default is 1e-8")
    mx = sub.add_parser("migration-forecast")
    mx.add_argument("--checkpoint", required=True)
    mx.add_argument("--observations", required=True)
    mx.add_argument("--output", required=True)
    mx.add_argument("--species", required=True)
    mx.add_argument("--animal")
    mx.add_argument("--issue-time")
    mx.add_argument("--samples", type=int, default=64)
    mx.add_argument("--seed", type=int, default=2026)
    mx.add_argument("--device", default="cpu")
    mi = sub.add_parser("migration-infer")
    mi.add_argument("--checkpoint", required=True)
    mi.add_argument("--observations", required=True)
    mi.add_argument("--output", required=True)
    mi.add_argument("--species", required=True)
    mi.add_argument("--animal")
    mi.add_argument("--canonical-animal")
    mi.add_argument("--issue-time")
    mi.add_argument("--samples", type=int, default=64)
    mi.add_argument("--seed", type=int, default=2026)
    mi.add_argument("--device", default="cpu")
    mv = sub.add_parser("migration-evaluate")
    mv.add_argument("--output", required=True)
    mv.add_argument("--checkpoints", nargs="*", default=[])
    mv.add_argument("--baselines", nargs="*", default=["stationary", "movement", "seasonal"])
    mv.add_argument("--splits", nargs="+", choices=["val_individual", "val_year", "test_individual", "test_year"], default=["val_individual", "val_year"])
    mv.add_argument("--assessment", help="Verified final-truth sidecar; requires a previously frozen model selection")
    mv.add_argument("--selection")
    mv.add_argument("--samples", type=int, default=64)
    mv.add_argument("--seed", type=int, default=2026)
    mv.add_argument("--batch-size", type=int, default=64)
    mv.add_argument("--bootstrap-replicates", type=int, default=300)
    mv.add_argument("--device", default="cpu")
    fs = sub.add_parser("full-moe-assemble")
    fs.add_argument("--global-checkpoint", required=True)
    fs.add_argument("--specialists", required=True, help="JSON mapping family names to independently trained checkpoints")
    fs.add_argument("--output", required=True)
    fs.add_argument("--identity-audit", required=True)
    fs.add_argument("--taxonomy-root", default="data/combined-v2/environment/taxonomy")
    fs.add_argument("--shared-checkpoint")
    fs.add_argument("--shared-data")
    fe = sub.add_parser("full-moe-evaluate")
    fe.add_argument("--experiments", required=True)
    fe.add_argument("--output", required=True)
    fe.add_argument("--identity-audit", required=True)
    fe.add_argument("--samples", type=int, default=64)
    fe.add_argument("--batch-size", type=int, default=256)
    fe.add_argument("--seed", type=int, default=2026)
    fe.add_argument("--device", default="auto")
    fe.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    fg = sub.add_parser("full-moe-calibrate")
    fg.add_argument("--checkpoint", required=True)
    fg.add_argument("--output", required=True)
    fg.add_argument("--identity-audit", required=True)
    fg.add_argument("--samples", type=int, default=32)
    fg.add_argument("--batch-size", type=int, default=256)
    fg.add_argument("--steps", type=int, default=300)
    fg.add_argument("--lr", type=float, default=.01)
    fg.add_argument("--seed", type=int, default=2026)
    fg.add_argument("--device", default="auto")
    fg.add_argument("--table-only", action="store_true")
    ef = sub.add_parser("ecology-fetch")
    ef.add_argument("--output", default="data/ecology-sources")
    ef.add_argument("--max-dem-tiles", type=int, default=96)
    ep = sub.add_parser("ecology-spatial")
    ep.add_argument("--sources", default="data/ecology-sources")
    ep.add_argument("--output", default="data/ecology-spatial")
    ep.add_argument("--workers", type=int, default=6)
    mf = sub.add_parser("marine-fetch")
    mf.add_argument("--output", default="data/ecology-sources")
    mf.add_argument("--maximum-mb", type=int, default=192)
    ee = sub.add_parser("ecology-expand")
    ee.add_argument("--source-data", required=True)
    ee.add_argument("--sources", default="data/ecology-sources")
    ee.add_argument("--spatial-cache", default="data/ecology-spatial")
    mc = sub.add_parser("moe-prepare")
    mc.add_argument("--checkpoints", nargs="+", required=True)
    mc.add_argument("--output", required=True)
    mc.add_argument("--identity-audit", required=True)
    mc.add_argument("--taxonomy-root", default="data/combined-v2/environment/taxonomy")
    mc.add_argument("--batch-size", type=int, default=512)
    mc.add_argument("--device", default="auto")
    mt_exp = sub.add_parser("moe-train")
    mt_exp.add_argument("--cache", required=True)
    mt_exp.add_argument("--output", required=True)
    mt_exp.add_argument("--mode", choices=["shared", "family", "learned"], default="family")
    mt_exp.add_argument("--hidden", type=int, default=96)
    mt_exp.add_argument("--epochs", type=int, default=25)
    mt_exp.add_argument("--patience", type=int, default=5)
    mt_exp.add_argument("--batch-size", type=int, default=512)
    mt_exp.add_argument("--lr", type=float, default=.0001)
    mt_exp.add_argument("--seed", type=int, default=42)
    mt_exp.add_argument("--nll-weight", type=float, default=.01)
    mt_exp.add_argument("--regularization", type=float, default=.001)
    mt_exp.add_argument("--max-train", type=int, default=0)
    mt_exp.add_argument("--device", default="auto")
    me = sub.add_parser("moe-evaluate")
    me.add_argument("--cache", required=True)
    me.add_argument("--checkpoints", nargs="+", required=True)
    me.add_argument("--output", required=True)
    me.add_argument("--samples", type=int, default=64)
    me.add_argument("--batch-size", type=int, default=512)
    me.add_argument("--seed", type=int, default=2026)
    me.add_argument("--device", default="auto")
    sub.add_parser("fetch-jaguar")
    sub.add_parser("catalogue")
    weather = sub.add_parser("weather")
    weather.add_argument("--max-cells", type=int, default=64)
    weather.add_argument("--grid-degrees", type=float, default=1.)
    weather.add_argument("--budget-mb", type=int, default=128)
    sub.add_parser("taxonomy")
    enrich = sub.add_parser("enrich")
    enrich.add_argument("--source-data", default="data")
    enrich.add_argument("--no-weather", action="store_true")
    enrich.add_argument("--weather-lag-days", type=int, default=2)
    enrich.add_argument("--traits", action="store_true")
    mem = sub.add_parser("add-memory")
    mem.add_argument("--source-data", required=True)
    veg = sub.add_parser("vegetation")
    veg.add_argument("--cache-root", required=True)
    veg.add_argument("--max-groups", type=int, default=128)
    veg.add_argument("--workers", type=int, default=4)
    av = sub.add_parser("add-vegetation")
    av.add_argument("--source-data", required=True)
    av.add_argument("--cache-root", required=True)
    sat = sub.add_parser("satellite")
    sat.add_argument("--cache-root", required=True)
    sat.add_argument("--max-files", type=int, default=32)
    sat.add_argument("--workers", type=int, default=3)
    ads = sub.add_parser("add-satellite")
    ads.add_argument("--cache-root", required=True)
    ads.add_argument("--source-data", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--query", default="", help="Comma-separated OR terms in archive metadata")
    h.add_argument("--max-studies", type=int, default=20)
    h.add_argument("--budget-mb", type=int, default=512)
    h.add_argument("--max-file-mb", type=int, default=50)
    c = sub.add_parser("clean")
    c.add_argument("--min-points", type=int, default=200)
    c.add_argument("--max-gap-hours", type=float, default=48)
    c.add_argument("--max-speed-mps", type=float, default=100)
    w = sub.add_parser("prepare")
    w.add_argument("--context", type=int, default=32)
    w.add_argument("--stride", type=int, default=4)
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24, 168])
    t = sub.add_parser("train")
    t.add_argument("--output", default="runs/local")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--width", type=int, default=128)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--components", type=int, default=8)
    t.add_argument("--lr", type=float, default=.0003)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--max-train", type=int, default=0)
    t.add_argument("--device", default="auto")
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--energy-weight", type=float, default=.05)
    t.add_argument("--student-df", type=int, default=4, help="Heavy-tailed components; 0 selects Gaussian")
    t.add_argument("--driver-groups", default="all", help="Comma-separated all,none,memory,clock,weather,traits,vegetation,satellite")
    t.add_argument("--driver-base-features", type=int, default=0, help="Preserve this existing driver prefix and learn a gated residual from new inputs")
    t.add_argument("--anchors", action="store_true")
    t.add_argument("--init-checkpoint")
    t.add_argument("--nll-weight", type=float, default=1.)
    t.add_argument("--stratified-energy", action="store_true")
    t.add_argument("--energy-repeats", type=int, default=2)
    t.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    t.add_argument("--preload", action="store_true")
    t.add_argument("--long-memory", action="store_true")
    t.add_argument("--memory-fusion", choices=["concat", "residual"], default="concat")
    t.add_argument("--horizon-adapter", action="store_true")
    t.add_argument("--regime-weight", type=float, default=0.)
    t.add_argument("--freeze-backbone", action="store_true")
    t.add_argument("--identity-audit")
    t.add_argument("--scope-family")
    t.add_argument("--taxonomy-root",default="data/combined-v2/environment/taxonomy")
    t.add_argument("--weight-decay",type=float,default=.01)
    t.add_argument("--environment-lr",type=float)
    t.add_argument("--environment-gate-lr",type=float)
    mt = sub.add_parser("moveformer-train")
    mt.add_argument("--output", required=True)
    mt.add_argument("--epochs", type=int, default=100)
    mt.add_argument("--patience", type=int, default=12)
    mt.add_argument("--batch-size", type=int, default=512)
    mt.add_argument("--width", type=int, default=128)
    mt.add_argument("--layers", type=int, default=6)
    mt.add_argument("--heads", type=int, default=8)
    mt.add_argument("--lr", type=float, default=.00005)
    mt.add_argument("--seed", type=int, default=42)
    mt.add_argument("--candidates", type=int, default=100)
    mt.add_argument("--validation-candidates", type=int, default=256)
    mt.add_argument("--device", default="auto")
    mt.add_argument("--max-train", type=int, default=0)
    mt.add_argument("--proposal-kind", choices=["crw", "empirical"], default="empirical")
    mt.add_argument("--variable-context", action="store_true")
    mt.add_argument("--warmup-steps", type=int, default=1000)
    rb=sub.add_parser("router-bank")
    rb.add_argument("--base-data",required=True)
    rb.add_argument("--checkpoints",nargs="+",required=True)
    rb.add_argument("--output",required=True)
    rb.add_argument("--samples",type=int,default=32)
    rb.add_argument("--batch-size",type=int,default=256)
    rb.add_argument("--device",default="auto")
    rf=sub.add_parser("router-fit")
    rf.add_argument("--bank",required=True)
    rf.add_argument("--output",required=True)
    rf.add_argument("--conditional",action="store_true")
    rf.add_argument("--steps",type=int,default=250)
    rf.add_argument("--lr",type=float,default=.02)
    rf.add_argument("--seed",type=int,default=42)
    rf.add_argument("--device",default="auto")
    ra=sub.add_parser("router-audit")
    ra.add_argument("--bank",required=True)
    ra.add_argument("--output",required=True)
    ra.add_argument("--identity-audit",required=True)
    hp=sub.add_parser("habitat-cache")
    hp.add_argument("--base-data",required=True)
    hp.add_argument("--checkpoints",nargs="+",required=True)
    hp.add_argument("--output",required=True)
    hp.add_argument("--count",type=int,default=256)
    hp.add_argument("--batch-size",type=int,default=256)
    hp.add_argument("--device",default="auto")
    ht=sub.add_parser("habitat-train")
    ht.add_argument("--cache",required=True)
    ht.add_argument("--fields",required=True)
    ht.add_argument("--output",required=True)
    ht.add_argument("--epochs",type=int,default=50)
    ht.add_argument("--patience",type=int,default=10)
    ht.add_argument("--batch-size",type=int,default=512)
    ht.add_argument("--lr",type=float,default=.0003)
    ht.add_argument("--seed",type=int,default=42)
    ht.add_argument("--device",default="auto")
    ht.add_argument("--identity-audit")
    tp = sub.add_parser("trajectory-prepare")
    tp.add_argument("--checkpoint", required=True)
    tp.add_argument("--output", required=True)
    tp.add_argument("--identity-audit", required=True)
    tp.add_argument("--fields")
    tp.add_argument("--taxonomy-root")
    tp.add_argument("--max-steps", type=int, default=20)
    tp.add_argument("--batch-size", type=int, default=256)
    tp.add_argument("--device", default="auto")
    tt = sub.add_parser("trajectory-train")
    tt.add_argument("--cache", required=True)
    tt.add_argument("--output", required=True)
    tt.add_argument("--epochs", type=int, default=30)
    tt.add_argument("--batch-size", type=int, default=192)
    tt.add_argument("--width", type=int, default=128)
    tt.add_argument("--layers", type=int, default=3)
    tt.add_argument("--lr", type=float, default=.0003)
    tt.add_argument("--seed", type=int, default=42)
    tt.add_argument("--patience", type=int, default=7)
    tt.add_argument("--device", default="auto")
    tt.add_argument("--validation-origins", type=int, default=2048)
    tt.add_argument("--validation-samples", type=int, default=8)
    tt.add_argument("--ode-steps", type=int, default=6)
    tt.add_argument("--max-train", type=int, default=0)
    tt.add_argument("--state-weight", type=float, default=.05)
    tt.add_argument("--prior-kind", choices=["bridge", "copula"], default="copula")
    tt.add_argument("--objective", choices=["flow", "energy"], default="flow")
    tt.add_argument("--training-samples", type=int, default=4)
    tt.add_argument("--reconstruction-weight", type=float, default=.05)
    te = sub.add_parser("trajectory-evaluate")
    te.add_argument("--checkpoint", required=True)
    te.add_argument("--output", required=True)
    te.add_argument("--identity-audit", required=True)
    te.add_argument("--samples", type=int, default=64)
    te.add_argument("--batch-size", type=int, default=32)
    te.add_argument("--seed", type=int, default=2026)
    te.add_argument("--max-origins", type=int, default=0)
    te.add_argument("--device", default="auto")
    te.add_argument("--export-origins", type=int, default=20)
    te.add_argument("--baseline-checkpoints", nargs="+")
    tj = sub.add_parser("trajectory-prior-ensemble")
    tj.add_argument("--cache", required=True)
    tj.add_argument("--checkpoints", nargs="+", required=True)
    tj.add_argument("--output", required=True)
    tf = sub.add_parser("trajectory-forecast")
    tf.add_argument("--checkpoint", required=True)
    tf.add_argument("--output", required=True)
    tf.add_argument("--window-ids", type=int, nargs="+", required=True)
    tf.add_argument("--times", type=float, nargs="+")
    tf.add_argument("--samples", type=int, default=64)
    tf.add_argument("--seed", type=int, default=2026)
    tf.add_argument("--device", default="auto")
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoints", nargs="+", required=True)
    e.add_argument("--output", default="runs/evaluation")
    e.add_argument("--samples", type=int, default=64)
    e.add_argument("--batch-size", type=int, default=256)
    e.add_argument("--seed", type=int, default=2026)
    e.add_argument("--max-per-split", type=int, default=0)
    e.add_argument("--device", default="auto")
    e.add_argument("--splits", nargs="+", choices=["val_temporal", "val_individual", "test_temporal", "test_individual"])
    e.add_argument("--identity-audit")
    e.add_argument("--fresh-only",action="store_true")
    x = sub.add_parser("export")
    group = x.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--checkpoints", nargs="+")
    x.add_argument("--output", default="runs/forecasts.json")
    x.add_argument("--count", type=int, default=100)
    x.add_argument("--samples", type=int, default=64)
    x.add_argument("--split", choices=["val_temporal", "val_individual", "test_temporal", "test_individual"], default="test_individual")
    x.add_argument("--seed", type=int, default=2026)
    x.add_argument("--forecaster", choices=["transformer", "analogue", "crw", "hybrid"], default="transformer")
    args = vars(p.parse_args()); command = args.pop("command"); root = args.pop("data")
    if command == "migration-prepare":
        from .migration_data import prepare
        print(json.dumps(prepare(**args), indent=2))
    elif command == "migration-train":
        from .migration_train import train
        print(json.dumps(train(root, **args), indent=2))
    elif command == "migration-evaluate":
        from .migration_evaluate import evaluate
        print(json.dumps(evaluate(root, allow_test=bool(args.get("assessment")), **args), indent=2))
    elif command == "migration-forecast":
        from .migration_forecast import forecast
        print(json.dumps(forecast(**args), indent=2))
    elif command == "migration-infer":
        from .migration_inference import forecast
        print(json.dumps(forecast(root, **args), indent=2))
    elif command == "full-moe-evaluate":
        from .full_moe_evaluate import evaluate
        print(json.dumps(evaluate(root, **args), indent=2))
    elif command == "full-moe-assemble":
        from .full_moe import assemble
        print(json.dumps(assemble(root, **args), indent=2))
    elif command == "full-moe-calibrate":
        from .full_moe_train import calibrate
        print(json.dumps(calibrate(root, **args), indent=2))
    elif command == "marine-fetch":
        from .marine import fetch
        print(json.dumps(fetch(**args), indent=2))
    elif command == "ecology-fetch":
        from .ecology_sources import fetch
        print(json.dumps(fetch(root, **args), indent=2))
    elif command == "ecology-spatial":
        from .landscape_context import prepare
        print(json.dumps(prepare(root, **args), indent=2))
    elif command == "ecology-expand":
        from .ecology import expand
        print(json.dumps(expand(args.pop("source_data"), root, **args), indent=2))
    elif command == "moe-evaluate":
        from .moe_evaluate import evaluate
        print(json.dumps(evaluate(root, **args), indent=2))
    elif command == "moe-prepare":
        from .moe import prepare
        print(json.dumps(prepare(root, **args), indent=2))
    elif command == "moe-train":
        from .moe_train import train
        print(json.dumps(train(root, **args), indent=2))
    elif command == "trajectory-forecast":
        from .trajectory_evaluate import forecast
        print(json.dumps(forecast(root, **args), indent=2))
    elif command == "trajectory-prior-ensemble":
        from .trajectory import build_prior_ensemble
        print(json.dumps(build_prior_ensemble(root, **args), indent=2))
    elif command == "trajectory-prepare":
        from .trajectory_data import prepare
        print(json.dumps(prepare(root, **args), indent=2))
    elif command == "trajectory-train":
        from .trajectory_train import train
        print(json.dumps(train(root, **args), indent=2))
    elif command == "trajectory-evaluate":
        from .trajectory_evaluate import evaluate
        print(json.dumps(evaluate(root, **args), indent=2))
    elif command=="router-audit":
        from .identity import audit_router_bank
        print(json.dumps(audit_router_bank(args["bank"],args["output"],args["identity_audit"]),indent=2))
    elif command in ("habitat-cache","habitat-train"):
        if command=="habitat-cache":
            from .habitat import cache_proposals
            print(json.dumps(cache_proposals(root,args.pop("base_data"),**args),indent=2))
        else:
            from .habitat_train import train
            print(json.dumps(train(root,**args),indent=2))
    elif command in ("router-bank","router-fit"):
        from .router import build_bank,fit_router
        if command=="router-bank":
            print(json.dumps(build_bank(root,args.pop("base_data"),**args),indent=2))
        else:print(json.dumps(fit_router(**args),indent=2))
    elif command == "moveformer-train":
        from .moveformer_train import train
        print(json.dumps(train(root, **args), indent=2))
    elif command in ("satellite", "add-satellite"):
        from .satellite import cache_satellite, add_satellite
        if command == "satellite":
            result = cache_satellite(root, **args)
        else:
            result = add_satellite(args["source_data"], root, args["cache_root"])
        print(json.dumps(result, indent=2))
    elif command in ("vegetation", "add-vegetation"):
        from .vegetation import cache_vegetation, add_vegetation
        if command == "vegetation":
            result = cache_vegetation(root, **args)
        else:
            result = add_vegetation(args["source_data"], root, args["cache_root"])
        print(json.dumps(result, indent=2))
    elif command == "add-memory":
        from .memory import add_memory
        print(json.dumps(add_memory(args["source_data"], root), indent=2))
    elif command == "taxonomy":
        from .traits import cache_taxonomy
        print(json.dumps(cache_taxonomy(root), indent=2))
    elif command in ("weather", "enrich"):
        from .drivers import cache_weather, enrich
        if command == "weather":
            result = cache_weather(root, **args)
            print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
        else:
            source = args.pop("source_data")
            args["weather"] = not args.pop("no_weather")
            print(json.dumps(enrich(source, root, **args), indent=2))
    elif command in ("fetch-jaguar", "catalogue", "harvest"):
        from . import sources
        fn = {"fetch-jaguar": sources.jaguar, "catalogue": sources.catalogue, "harvest": sources.harvest}[command]
        result = fn(root, **args)
        print(json.dumps({"command": command, "result": str(result) if command == "fetch-jaguar" else f"{len(result)} records"}))
    else:
        if command in ("clean", "prepare"):
            from . import data
            result = getattr(data, command)(root, **args)
        elif command == "train":
            from .train import train
            result = train(root, **args)
        else:
            from . import evaluate
            result = getattr(evaluate, command)(root, **args)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
