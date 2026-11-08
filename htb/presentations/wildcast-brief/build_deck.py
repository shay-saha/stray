"""Build the short WildCast presentation from the frozen local evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT / "artifacts/migration-current-ensemble-v1"
PUBLIC = ROOT / "reports/migration-public-panel-v1"
FULL = ROOT / "reports/migration-repaired-dev-combined-v1"
COLORS = dict(ink="123047", muted="52697A", teal="008E86", mint="D9F0EB", blue="3B6F92",
              pale="F5F8FA", white="FFFFFF", line="D9E2E8", amber="95621C", sand="F7EAD7")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb(value):
    return RGBColor.from_string(COLORS.get(value, value))


def box(slide, x, y, w, h, fill="white", line=None, rounded=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                  Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line); shape.line.width = Pt(.7)
    else:
        shape.line.fill.background()
    if rounded:
        shape.adjustments[0] = .12
    return shape


def text(slide, value, x, y, w, h, size=20, color="ink", bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.alignment = align; p.space_after = Pt(6)
        p.font.name = "Arial"; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = rgb(color)
    return shape


def pill(slide, value, x, y, w, fill="mint", color="teal", size=14):
    box(slide, x, y, w, .39, fill, rounded=True)
    text(slide, value, x + .12, y + .065, w - .24, .25, size, color, True)


def footer(slide, number, source):
    box(slide, .6, 6.94, 12.13, .012, "line")
    text(slide, "WILDCAST  /  DEVELOPMENT EVIDENCE  /  8 NOV 2026", .6, 7.08, 7.2, .18, 9, "muted")
    text(slide, source, .6, 6.58, 11.9, .29, 9, "muted")
    text(slide, str(number), 12.13, 7.05, .6, .22, 11, "muted", align=PP_ALIGN.RIGHT)


def slide(prs, title, kicker, source):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb("pale")
    text(s, kicker.upper(), .62, .35, 12.1, .25, 12, "teal", True)
    text(s, title, .6, .84, 12.1, .85, 32, "ink", True)
    footer(s, len(prs.slides), source)
    return s


def note(slide, talk, sources, backup=""):
    value = talk + ("\n\nTechnical detail\n" + backup if backup else "")
    slide.notes_slide.notes_text_frame.text = value + "\n\nEvidence\n" + "\n".join(sources)


TALK = [
    ("0:00–0:20", "Forecasting where animals move next",
     "WildCast predicts an individual animal's possible future locations, from one hour to ninety days. The goal is a frontier migration model. What we have today is a probabilistic movement model with an encouraging local benchmark result, broad training coverage, and a clear path to stronger evidence."),
    ("0:20–1:00", "The dataset and the inputs",
     "We reviewed 34.8 million position observations from public tracking studies. The corrected training set contains about 800,000 forecast origins across 108 taxon labels: 107 species names and one tortoise family label. Training data represents 131 study labels, including terrestrial animals, birds and some marine species. Our current leader takes observed GPS history, species and time. Season, recent velocity and daylight are derived from that history. Weather, currents, habitat, age and sex are not yet inputs to the leading model. Missing observations remain explicitly missing, and future positions are never interpolated into training labels."),
    ("1:00–1:45", "How the architecture works",
     "We encode the animal's observed history at 29 lag slots, covering minutes through a year, together with masks, species and 22 derived features. Four predictors use causal Transformer history encoders: two forecast locations directly, and two parameterize a route through a possible destination. Each predicts an eight-component joint distribution across nine horizons. A learned ensemble selects a predictor for each complete sampled route. This lets us retain different plausible destinations and dependence between future times, rather than forcing one average path. The complete ensemble contains about 35 million parameters."),
    ("1:45–2:30", "What the comparison shows",
     "For a controlled local comparison, every method forecast the same 4,096 origins at one hour, twelve hours and one day, with 64 samples. The ensemble's endpoint energy score is 10.18 kilometres, versus 11.08 for a correlated random walk and 12.26 for the deepSSF adaptation. Lower scores mean better predicted distributions. That is about eight percent and seventeen percent lower, respectively. These are development results from local method adaptations. Ensemble weights used development data, and this is not an official MoveBench ranking or an independent final test."),
    ("2:30–3:10", "Why the design may help",
     "The design combines recent movement and seasonal context, represents several possible futures, and mixes direct and destination-based predictors. The evidence for combining predictors is specific: on this same panel, the ensemble improves the score by 2.9 percent over our strongest single model. Most of that improvement is in the unseen-animal split; the held-out-year gain is small and uncertain. The architectural mechanisms are plausible explanations, but we still need controlled ablations to establish which features cause the gain. We cannot claim that every component independently outperforms every alternative."),
    ("3:10–4:00", "Constraints and the next improvements",
     "The main constraints are incomplete environmental and demographic inputs, uneven species and regional coverage, and limited independent evaluation. Movement observations also do not always identify active migration. Human built-up maps are being tested with controls; completed adaptation attempts have not improved the leader. Next, we need a frozen independent test and stronger public comparisons, then issue-available weather, ocean conditions, habitat and infrastructure, with reliable age and sex metadata. We should expand independent marine populations and check uncertainty under geographic and temporal shift. The claim today is a useful, fast model with promising local gains, while frontier superiority remains to be demonstrated.")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = BUNDLE / "data/reference/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    public = json.loads((PUBLIC / "metrics.json").read_text())
    full = json.loads((FULL / "metrics.json").read_text())
    preflight = json.loads((ROOT / "reports/migration-repaired-source-preflight-v1.json").read_text())
    score = pd.DataFrame(public["scores"]).groupby("model").endpoint_energy_km.mean()
    names = ["migration-repaired-ensemble-v1", "scaled-history_direct", "migration-public-crw-v1", "deepSSF", "stationary"]
    labels = ["WildCast ensemble", "WildCast single", "Correlated random walk", "deepSSF adapter", "Stationary"]
    values = score[names].to_numpy()
    relative_crw = 100 * (1-values[0]/values[2]); relative_deep = 100*(1-values[0]/values[3])
    relative_single = 100*(1-values[0]/values[1])
    with np.load(BUNDLE / "data/reference/cases.npz", allow_pickle=False) as a:
        split, study, species = a["split"], a["study"], a["species"]
        train = split == 0
        inventory = []
        for name, token in manifest["studies"].items():
            counts = {key:int(np.sum((study == token) & (split == code))) for code, key in
                      enumerate(("train", "val_individual", "val_year", "test_individual", "test_year"))}
            inventory.append({"study_title": name.split(":",1)[-1], "study_key": name, "study_token": token, **counts})
        pd.DataFrame(inventory).sort_values("study_title").to_csv(OUT / "study-inventory.csv", index=False)
        taxon = [{"training_taxon": name, "token": token, "training_origins":int(np.sum(train & (species == token))),
                  "taxonomic_level": "family" if name == "Testudinidae" else "species_name"} for name, token in manifest["species"].items()]
        pd.DataFrame(taxon).to_csv(OUT / "taxon-inventory.csv", index=False)
        train_studies = len(np.unique(study[train])); all_studies = len(np.unique(study))
    ensemble = torch.load(BUNDLE / "model/selected.pt", map_location="cpu", weights_only=True)
    members = []
    for record, weight in zip(ensemble["members"], ensemble["weights"].tolist()):
        ck = torch.load(BUNDLE / record["path"], map_location="cpu", weights_only=True)
        from_model = ck["config"]
        members.append({"variant":from_model["variant"], "width":from_model["width"], "layers":from_model["layers"],
                        "components":from_model["components"], "weight":weight, "checkpoint_sha256":record["sha256"]})
    counts = manifest["case_counts"]
    evidence = {"created_utc":datetime.now(timezone.utc).isoformat(), "public_report":str(PUBLIC.relative_to(ROOT)),
        "manifest_sha256":sha(manifest_path), "cases_sha256":manifest["cases_sha256"],
        "public_metrics_sha256":sha(PUBLIC/"metrics.json"), "full_metrics_sha256":sha(FULL/"metrics.json"),
        "preflight_sha256":sha(ROOT/"reports/migration-repaired-source-preflight-v1.json"),
        "training_taxon_labels":len(manifest["species"]), "species_named_labels":107, "family_labels":["Testudinidae"],
        "reviewed_observations":preflight["rows"], "training_studies":train_studies, "case_studies":all_studies,
        "case_counts":counts, "endpoint_comparison":dict(zip(labels,map(float,values))), "public_panel_origins":public["cases"],
        "public_panel_observed_targets":public["observed_targets"], "public_panel_horizons_days":public["horizons_days"],
        "samples":public["samples"], "members":members,
        "relative_energy_improvement_percent":{"versus_CRW":relative_crw,"versus_deepSSF":relative_deep,"versus_single":relative_single},
        "interpretation":"Local development comparisons; ensemble weights fitted on development; no official benchmark, independent final-test or global-frontier-superiority claim",
        "primary_method_sources":["https://github.com/cifkao/moveformer","https://github.com/swforrest/deepSSF_package"]}
    (OUT/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":13})
    fig, ax = plt.subplots(figsize=(8.8, 3.9), facecolor="#F5F8FA")
    ax.set_facecolor("#F5F8FA")
    bars = ax.barh(np.arange(5), values, color=["#008E86","#6BBEB3","#748B9C","#748B9C","#B8C7D2"],height=.57)
    ax.set_yticks(np.arange(5),labels,fontsize=13.5,color="#123047"); ax.invert_yaxis()
    ax.set_xlim(0,16); ax.set_xticks([0,5,10,15]); ax.tick_params(axis="both",length=0,labelcolor="#52697A")
    ax.grid(axis="x",color="#D9E2E8",linewidth=.7); ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
    for b,v in zip(bars,values):ax.text(v+.18,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=15,fontweight="bold",color="#123047")
    ax.set_xlabel("Endpoint energy score (km) — lower is better",fontsize=12.5,labelpad=13,color="#52697A")
    fig.subplots_adjust(left=.34,right=.98,top=.98,bottom=.2)
    for ext in ("png","svg","pdf"):fig.savefig(OUT/f"benchmark-chart.{ext}",dpi=220,facecolor=fig.get_facecolor())
    plt.close(fig)
    prs = Presentation(); prs.slide_width=Inches(13.333333); prs.slide_height=Inches(7.5)
    prs.core_properties.title="WildCast Animal Movement Forecasting"
    prs.core_properties.subject="3–5 minute research presentation with six main slides and technical backup"
    prs.core_properties.author="WildCast"

    # 1: the scope and the useful claim.
    s=slide(prs,"Forecasting where animals move next","WildCast","Evidence: immutable corrected cache; current packaged ensemble; matched local development panel.")
    text(s,"One animal’s history.\nMany plausible futures.",.65,2.0,8.6,1.6,43,"ink",True)
    text(s,"Probabilistic movement forecasts from 1 hour to 90 days",.68,3.93,10.8,.5,23,"muted")
    pill(s,"108 training taxon labels",.68,5.0,3.1)
    pill(s,"Four predictors, one ensemble",4.05,5.0,3.75)
    pill(s,"Measured local gains",8.07,5.0,3.0)
    text(s,"Research objective: frontier migration forecasting",.68,5.7,11.7,.4,20,"muted")
    note(s,TALK[0][2],["evidence.json",str(PUBLIC.relative_to(ROOT)/"metrics.json")])

    # 2: the data and exactly what enters the leading model.
    s=slide(prs,"The dataset and the inputs","Data and scope","Sources: corrected manifest and source preflight; migration_repaired_features.py. Forecast origins are not independent animals.")
    for x,value,label in [(0.65,"34.8M","reviewed position observations"),(4.85,"799,681","training forecast origins"),(9.05,"108","training taxon labels")]:
        text(s,value,x,1.85,3.65,.8,43,"teal",True);text(s,label,x,2.72,3.62,.65,17,"muted")
    box(s,.65,3.65,5.87,2.12,"white",rounded=True);box(s,6.8,3.65,5.88,2.12,"white",rounded=True)
    text(s,"Public animal tracking studies",.9,3.93,5.3,.4,21,"ink",True)
    text(s,f"{train_studies} study labels in TRAIN\nTerrestrial animals, birds and marine examples",.9,4.48,5.3,.95,19,"muted")
    text(s,"Current leading inputs",7.05,3.93,5.3,.4,21,"ink",True)
    text(s,"Observed GPS + timestamps + species\nDerived season, velocity and daylight",7.05,4.48,5.3,.95,19,"muted")
    text(s,"108 labels = 107 species names + Testudinidae (family). Observed future positions are never interpolated.",.68,6.0,11.9,.4,15,"muted")
    note(s,TALK[1][2],["evidence.json","study-inventory.csv","taxon-inventory.csv",str(manifest_path.relative_to(ROOT))],
         f"{all_studies} study labels have cases across all splits; {train_studies} have TRAIN cases. There are 4,300 conservative training exclusion groups, not a certified count of biological animals. The leader excludes weather, currents, habitat, infrastructure, age and sex. Missing history is masked.")

    # 3: a simple, editable architecture diagram.
    s=slide(prs,"From observed history to plausible futures","Architecture","Sources: migration_model.py; migration_ensemble.py; packaged member configurations and weights in evidence.json.")
    cards=[(.65,3.43,"Observed history","29 lag slots + masks\nSpecies + 22 derived features\nMinutes through one year"),
           (4.72,3.43,"Four joint predictors","Causal Transformer encoders\nDirect location × 2\nGoal route × 2"),
           (8.8,3.88,"Joint future distribution","8 components per predictor\nWeighted ensemble\n64 sampled paths × 9 horizons")]
    for x,w,title,body in cards:
        box(s,x,2.0,w,2.9,"white",rounded=True)
        text(s,title,x+.22,2.31,w-.44,.7,23,"ink",True)
        text(s,body,x+.22,3.25,w-.44,1.4,18,"muted")
    for x in (4.25,8.35):
        a=s.shapes.add_shape(MSO_SHAPE.CHEVRON,Inches(x),Inches(3.15),Inches(.32),Inches(.5));a.fill.solid();a.fill.fore_color.rgb=rgb("teal");a.line.fill.background()
    text(s,"≈35M total parameters",.72,5.45,4.1,.4,23,"teal",True)
    text(s,"A predictor is selected once per complete sampled route",5.0,5.42,7.6,.73,21,"ink")
    note(s,TALK[2][2],["wildcast/migration_model.py","wildcast/migration_ensemble.py","evidence.json"],
         "Compact experts have width128/layers2; scaled experts width512/layers6, with4 attention heads in each encoder. Each component has9×2 mean coordinates and an18×18 structured covariance. The goal-route parameterization is a destination/timing/curve bias, not an externally supplied destination. Weights are global, not an animal-specific gating network.")

    # 4: matched numerical evidence.
    s=slide(prs,"Better scores on the matched local panel","Measured comparison","Source: migration-public-panel-v1/metrics.json. Local adaptations; ensemble probabilities fitted on development; no official MoveBench rank.")
    text(s,"4,096 identical origins  •  1h / 12h / 1d  •  64 samples",.68,1.72,11.9,.4,20,"muted")
    s.shapes.add_picture(str(OUT/"benchmark-chart.png"),Inches(.55),Inches(2.12),width=Inches(9.15))
    box(s,10.13,2.55,2.5,1.45,"mint",rounded=True)
    text(s,f"{relative_crw:.1f}%",10.31,2.74,2.1,.62,39,"teal",True)
    text(s,"lower than CRW",10.31,3.42,2.1,.35,16,"teal")
    box(s,10.13,4.24,2.5,1.45,"white",rounded=True)
    text(s,f"{relative_deep:.1f}%",10.31,4.43,2.1,.62,39,"ink",True)
    text(s,"lower than deepSSF",10.31,5.11,2.1,.35,16,"muted")
    text(s,"Energy evaluates the full predicted distribution. Development results are averaged across two validation splits.",.68,6.22,11.9,.30,13,"muted")
    note(s,TALK[3][2],[str(PUBLIC.relative_to(ROOT)/"metrics.json"),str(PUBLIC.relative_to(ROOT)/"comparison-summary.json")],
         "8,831 observed targets. Equal conservative identity weight within species, then horizons and validation splits. deepSSF uses its selected trained epoch3 checkpoint. More recent MoveFormer and iSSF runs are excluded from this fixed table until a common64-draw comparison is complete. Energy is not metabolic energy or a percentage accuracy score.")

    # 5: observed ensemble effect and bounded explanations.
    s=slide(prs,"Why combining predictors can help","Evidence and design rationale","Sources: matched public-panel scores and paired cluster bootstrap; model source. Mechanisms below are hypotheses, not isolated causal findings.")
    for x,title,body in [(0.65,"History and season","Connect recent movement\nwith annual context"),
                         (4.78,"Several possible futures","Represent different destinations\nand uncertainty over time"),
                         (8.91,"Complementary predictors","Combine direct forecasts\nwith a destination-based bias")]:
        box(s,x,2.03,3.77,2.4,"white",rounded=True)
        text(s,title,x+.2,2.32,3.37,.75,22,"ink",True);text(s,body,x+.2,3.23,3.37,1.12,18,"muted")
    text(s,f"{relative_single:.1f}%",.75,4.82,2.15,.9,48,"teal",True)
    text(s,"lower energy than our strongest single predictor",3.0,4.98,9.2,.8,26,"ink",True)
    text(s,"The gain is concentrated in unseen animals. The held-out-year difference is small and uncertain.",.75,6.0,11.9,.4,17,"muted")
    note(s,TALK[4][2],[str(PUBLIC.relative_to(ROOT)/"metrics.json"),str(PUBLIC.relative_to(ROOT)/"comparison-summary.json")],
         "Ensemble vs single: val_individual7.866 vs8.454km; val_year12.488 vs12.515km. The year-split paired95% Bayesian bootstrap interval includes zero. These intervals are conditional on observed exclusion blocks and do not adjust for model selection. Controlled feature/architecture ablations and independent test evidence are still needed.")

    # 6: the useful research conclusion and priorities.
    s=slide(prs,"Constraints and the next improvements","What comes next","Sources: current input code, metadata proposal, context trials and evaluation scope. Frontier superiority remains unestablished.")
    box(s,.65,1.95,5.82,3.8,"white",rounded=True);box(s,6.76,1.95,5.92,3.8,"white",rounded=True)
    text(s,"Current constraints",.92,2.25,5.28,.5,24,"ink",True)
    text(s,"Environment and demographics missing\nUneven species and regional coverage\nMovement labels mix migration and residence\nIndependent final test still pending",.92,3.02,5.23,2.37,20,"muted")
    text(s,"Priorities",7.03,2.25,5.34,.5,24,"teal",True)
    text(s,"1  Freeze independent public comparisons\n2  Add dated environmental and trait data\n3  Expand independent marine populations\n4  Validate calibration and transfer",7.03,3.02,5.25,2.37,20,"ink")
    text(s,"Today: promising local gains. Next: demonstrate reliable superiority on independent data.",.73,6.0,11.95,.4,20,"teal",True)
    note(s,TALK[5][2],["wildcast/migration_context_joint.py","reports/migration-causal-deployment-feature-proposal-v1.json","reports/migration-delivery-v1-status.json"],
         "Human-pressure trials use GHSL built-up surface with prior nominal epochs and retrospective-vintage flags. Completed fine-tuning/frozen-projection attempts did not beat the GPS parent; training from original initialization remains under evaluation. GHSL is not a complete road/fence/habitat layer. Weather/ocean additions must respect forecast issue availability. Current architecture forecasts individuals, not social interactions or whole populations.")
    prs.save(OUT/"wildcast-brief.pptx")

    # Optional backup slides; not part of the four-minute talk.
    s=slide(prs,"Inputs and outputs in detail","Technical backup 1","Sources: migration_repaired_features.py; packaged configs; inference benchmark; data manifest.")
    text(s,"Model inputs",.72,1.92,5.7,.5,25,"ink",True)
    text(s,"29 × 10 history features + 29 masks\n22 derived static features\n108 taxon tokens + unknown token\nAt least 8 prior fixes spanning ≥6h",.72,2.62,5.7,2.3,21,"muted")
    text(s,"Model outputs",7.05,1.92,5.6,.5,25,"ink",True)
    text(s,"9 horizons: 1h, 12h, 1, 3, 7, 14, 30, 60, 90d\nSamples from a joint route distribution\nLocation and uncertainty at each horizon\nNo learned animal-ID embedding",7.05,2.62,5.58,2.3,20,"muted")
    text(s,"Local RTX 4060 Laptop: 108 tracks × 64 futures × 9 horizons in ≈41 ms",.75,5.7,11.9,.5,22,"teal",True)
    text(s,"Warm model and prepared inputs; coordinate conversion included; raw-track preparation and loading excluded.",.75,6.22,11.9,.3,13,"muted")
    note(s,"Use only if asked about implementation or deployment.",["reports/migration-inference-local-v1.json","wildcast/migration_repaired_features.py"])

    s=slide(prs,"Dataset split and coverage","Technical backup 2","Sources: corrected cache manifest, preflight, study-inventory.csv and taxon-inventory.csv.")
    rows=[("TRAIN",counts["train"],"Cutoff through 2020"),("Validation: unseen animals",counts["val_individual"],"Conservative identity exclusions"),
          ("Validation: held-out year",counts["val_year"],"2021"),("Reserved assessment",counts["test_individual"]+counts["test_year"],"Targets remain sealed")]
    for i,(label,n,description) in enumerate(rows):
        y=1.95+i*.84;box(s,.7,y,11.95,.66,"white",rounded=True)
        text(s,label,.9,y+.14,4.9,.4,20,"ink",True);text(s,f"{n:,}",6.1,y+.12,2.0,.4,22,"teal",True);text(s,description,8.3,y+.16,4.0,.36,15,"muted")
    text(s,"Marine examples: green and hawksbill turtles, grey seals, sperm whales and seabirds.",.75,5.7,11.8,.4,19,"ink")
    text(s,"144 study labels have cases; 131 have TRAIN cases. Coverage is uneven and is not a sample of all animal migration.",.75,6.18,11.8,.3,13,"muted")
    note(s,"Explain that a forecast origin is a prediction cutoff and can share an animal's history with other origins. Independent groups, regions and years matter more than a raw example count.",["evidence.json","study-inventory.csv","taxon-inventory.csv"])

    s=slide(prs,"Architecture and evaluation details","Technical backup 3","Sources: packaged checkpoint configurations; migration_train.py; migration_evaluate.py; migration_ensemble.py.")
    text(s,"Predictor",.8,1.87,5.0,.4,20,"muted",True);text(s,"Width / layers",6.2,1.87,2.6,.4,20,"muted",True);text(s,"Ensemble share",9.5,1.87,2.8,.4,20,"muted",True)
    for i,item in enumerate(members):
        y=2.42+i*.62;label=("Direct" if item["variant"]=="history_direct" else "Goal route")+(" compact" if item["width"]==128 else " scaled")
        text(s,label,.8,y,5.0,.4,21,"ink",True);text(s,f"{item['width']} / {item['layers']}",6.2,y,2.6,.4,21,"muted");text(s,f"{item['weight']*100:.1f}%",9.5,y,2.8,.4,21,"teal",True)
    text(s,"TRAIN-only normalization; square-root taxon sampling; NLL + energy optimization.",.8,5.3,11.7,.46,18,"ink")
    text(s,"Full development: 245,459 origins, 1,734,789 observed targets and 64 draws at nine horizons.",.8,5.94,11.7,.46,18,"muted")
    note(s,"The full development benchmark covers the four single predictors and basic references. The latest full64-draw ensemble/context comparison is pending; do not present a16-draw ensemble screen as the same experiment. The sampler is square-root taxon balancing within conservative identity groups.",[str(FULL.relative_to(ROOT)/"metrics.json"),"evidence.json"])

    s=slide(prs,"What the score means and what it proves","Technical backup 4","Evidence: exact local report hashes in evidence.json. Method code: github.com/cifkao/moveformer; github.com/swforrest/deepSSF_package.")
    text(s,"Energy score",.77,1.91,5.7,.45,25,"ink",True)
    text(s,"Rewards samples near the observed future,\nwith a correction for forecast dispersion.\nLower is better; units are kilometres.",.77,2.62,5.8,1.6,21,"muted")
    text(s,"Mean point error",7.03,1.91,5.6,.45,25,"ink",True)
    text(s,"Measures the miss of the forecast mean.\nIt answers a different question from\nthe full distribution score.",7.03,2.62,5.6,1.6,21,"muted")
    text(s,"What we can claim",.77,4.7,5.7,.5,22,"teal",True)
    text(s,"Better energy on the tested local panel",.77,5.32,5.7,.7,20,"ink")
    text(s,"What still needs evidence",7.03,4.7,5.6,.5,22,"amber",True)
    text(s,"Independent transfer, official benchmark\nranking and global frontier superiority",7.03,5.32,5.6,.85,20,"muted")
    note(s,"Energy is a proper scoring rule for a probability distribution. It is not metabolic expenditure. All published-paper comparisons here are local adapters, not official leaderboard reproductions. Released pretrained checkpoint overlap with the evaluation animals has not been established, so no comparison against those weights is claimed.",[str(PUBLIC.relative_to(ROOT)/"metrics.json"),"https://github.com/cifkao/moveformer","https://github.com/swforrest/deepSSF_package"])
    prs.save(OUT/"wildcast-with-technical-backup.pptx")
    lines=["WILDCAST  |  3–5 MINUTE PRESENTATION", "Six main slides; target delivery about four minutes.",
           "Evidence is the frozen local development snapshot in evidence.json.", ""]
    for i,(timing,title,spoken) in enumerate(TALK,1):
        lines.extend([f"SLIDE {i}  |  {timing}  |  {title}",spoken,""])
    lines.extend(["PRESENTATION GUARDRAILS", "Say '108 training taxon labels':107 species names plus one family label.",
        "Say 'lower endpoint energy on our matched local development panel', rather than 'best in the world'.",
        "The8.1% and17.0% figures are relative reductions in energy, not accuracy percentages.",
        "Do not compare the short-panel10.18km score with an all-horizon score as though they were the same benchmark.",
        "Environment/demographics are not inputs to the current delivered leader. Human built-up trials are separate.",
        "Ensemble weights used development data; final test targets remain sealed.", "",
        "QUICK QUESTIONS AND ANSWERS", "Why could the model help?", "Its design combines observed movement memory, season, multiple future modes and complementary predictors. Their individual causal contribution still needs controlled ablations.", "",
        "What does10.18km mean?", "It is an endpoint energy distribution score averaged over species/identity/horizon strata and two validation splits. Mean point error for the same ensemble panel is about13.84km. Neither number is distance travelled or metabolic energy.", "",
        "How good is one-hour prediction?", "On the full corrected development evaluation, the strongest single model's one-hour mean location errors are about3.22km for unseen animals and3.26km for the held-out year. The stationary baseline is about3.23/3.21km, so short-horizon advantage is limited and varies by cohort. The ensemble's final all-horizon comparison is still pending.", "",
        "Is this a whole-earth simulator?", "It forecasts conditional individual trajectories. It still needs initial populations, interactions, habitat constraints and validated recursive rollouts for a global population simulation.", "",
        "Which marine animals are trained?", "Examples include Chelonia mydas, Eretmochelys imbricata, Halichoerus grypus, Physeter macrocephalus and seabirds. The taxon inventory provides all108 labels. This does not establish broad coverage of sharks, fish or other missing marine groups.", "",
        "Can human infrastructure affect migration?", "Yes as a scientific mechanism, but a causal effect is not established by this model. Our current experiment uses retrospective GHSL built-up surface with dated metadata and matched controls; it is not a complete road/fence/barrier layer. Completed adaptation trials have not improved the leader.", "",
        "Can it run locally?", "The warm four-model ensemble took about41ms for108 prepared tracks on an RTX4060 Laptop GPU with8GB VRAM, generating64 futures at9 horizons. This excludes raw-track preparation and checkpoint loading.", "",
        "What are the next priorities?", "Independent assessment and comparable public runs; then issue-available weather/ocean/habitat/infrastructure and verified deployment traits; then broader independent populations and calibration/transfer checks.", "",
        "SOURCES", "evidence.json contains report and checkpoint fingerprints.", "study-inventory.csv lists actual study contributions by split.", "taxon-inventory.csv lists training labels and training-origin counts.",
        "Matched comparison: reports/migration-public-panel-v1/metrics.json", "Full development: reports/migration-repaired-dev-combined-v1/metrics.json", "Model: wildcast/migration_model.py; wildcast/migration_ensemble.py", "Inputs: wildcast/migration_repaired_features.py", "MoveFormer primary source: https://github.com/cifkao/moveformer", "deepSSF primary source: https://github.com/swforrest/deepSSF_package", ""])
    (OUT/"speaker-notes.txt").write_text("\n\n".join(lines))
    print(json.dumps({"main_slides":6,"total_slides":len(prs.slides),"talk_words":sum(len(item[2].split()) for item in TALK),"output":str(OUT),"training_studies":train_studies,"benchmarks":dict(zip(labels,map(float,values)))},indent=2))


if __name__ == "__main__":
    main()
