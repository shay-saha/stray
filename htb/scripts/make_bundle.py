"""Create a portable training bundle without virtualenvs or whole raw archives."""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parents[1]
out = root / "artifacts" / "wildcast-training.tar.gz"
out.parent.mkdir(exist_ok=True)
files = [root / p for p in ["README.md", "RESEARCH.md", "pyproject.toml", "requirements-validated.txt"]]
for pattern in ["wildcast/*.py", "tests/*.py", "scripts/*.py", "scripts/*.sh", "reports/*",
                "data/processed/*", "data/raw/*/source.json", "data/raw/*/README*", "data/catalogue.json",
                "runs/multispecies-seed*/best.pt", "runs/multispecies-seed*/config.json", "runs/multispecies-seed*/history.json",
                "runs/forecasts.json"]:
    files.extend(p for p in root.glob(pattern) if p.is_file())
with tarfile.open(out, "w:gz", compresslevel=3) as archive:
    for p in sorted(set(files)):
        archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
manifest = {"file": out.name, "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "files": [str(p.relative_to(root)) for p in sorted(set(files))],
            "note": "Includes clean observation/windows data, source attribution, code and selected weights. Raw event archives are not bundled."}
out.with_suffix(out.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
