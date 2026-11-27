"""Package a processed source (or a built version) as a folder ready for `kaggle datasets create` / web upload.

  python scripts/package_kaggle.py source vn_team                 -> dist/kaggle/vietsight-src-vn-team/processed/vn_team/...
  python scripts/package_kaggle.py version vietsight_v2           -> dist/kaggle/vietsight-vietsight-v2/{images,labels,data.yaml}
  then:  kaggle datasets create -p dist/kaggle/<folder> --dir-mode zip     (needs ~/.kaggle/kaggle.json, PRIVATE by default)
     or: kaggle.com > Datasets > New dataset > upload the folder (keep it Private: licenses are non-commercial / share-alike)

The prep notebook copies every attached `processed/<source>/` (meta.jsonl present) into its PROCESSED folder, so a
version config can merge team data with the public sources converted on Kaggle.
Only images with status ok and review/ drafts are packaged as they are; nothing is modified.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vslib import ROOT  # noqa: E402


def slug(s):
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["source", "version"])
    ap.add_argument("name")
    ap.add_argument("--owner", default="YOUR_KAGGLE_USERNAME", help="kaggle username for dataset-metadata.json")
    ap.add_argument("--out", default=str(ROOT / "dist" / "kaggle"))
    a = ap.parse_args()
    ds = f"vietsight-{'src-' if a.kind == 'source' else ''}{slug(a.name)}"
    out = Path(a.out) / ds
    if out.exists():
        shutil.rmtree(out)
    if a.kind == "source":
        src = ROOT / "data" / "processed" / a.name
        if not (src / "meta.jsonl").is_file():
            raise SystemExit(f"{src} is not a processed source")
        shutil.copytree(src, out / "processed" / a.name)
    else:
        src = ROOT / "data" / "versions" / a.name
        if not (src / "data.yaml").is_file():
            raise SystemExit(f"{src} is not a built version")
        shutil.copytree(src, out / a.name)
        rep = ROOT / "data" / "versions" / f"{a.name}_report"
        if rep.is_dir():
            shutil.copytree(rep, out / f"{a.name}_report")
    meta = {"title": ds, "id": f"{a.owner}/{ds}", "licenses": [{"name": "other"}],
            "description": "VietSight (AIInAction RoboTaxi, team 55). Provenance: docs/DATA_SOURCES.md. Private, competition use only."}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    n = sum(1 for _ in out.rglob("*.jpg"))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
    print(f"{out}: {n} images, {size:.0f} MiB\n  kaggle datasets create -p \"{out}\" --dir-mode zip")


if __name__ == "__main__":
    main()
