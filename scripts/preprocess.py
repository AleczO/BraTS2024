"""Preprocess the dataset for training, once, into data/processed.

Run from the repo root:
    python scripts/preprocess.py                 # skips cases already done
    python scripts/preprocess.py --workers 16
    python scripts/preprocess.py --overwrite     # redo everything

What happens to a single case is defined in src/dataset/preprocess.py.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# make `src` importable; must run before any `from src...` import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.manifest import build_manifest  # noqa: E402
from src.dataset.paths import get_processed_dir  # noqa: E402
from src.dataset.preprocess import check_version, preprocess_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BraTS 2024 GLI cases once.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel processes (default: 8)")
    parser.add_argument("--overwrite", action="store_true", help="Redo cases already processed")
    args = parser.parse_args()

    out_dir = get_processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    check_version(out_dir, args.overwrite)

    manifest_path = out_dir / "manifest.json"
    done = {} if args.overwrite or not manifest_path.exists() else {
        e["case_id"]: e for e in json.loads(manifest_path.read_text())
    }

    records = build_manifest()
    todo = [r for r in records
            if r["case_id"] not in done or not Path(done[r["case_id"]]["processed"]).exists()]
    print(f"{len(records)} cases, {len(records) - len(todo)} already done, {len(todo)} to process")

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(preprocess_case, r, out_dir): r["case_id"] for r in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                entry = fut.result()
                done[entry["case_id"]] = entry
            except Exception as e:  # keep going, report at the end
                failed.append((futures[fut], repr(e)))

    entries = [done[r["case_id"]] for r in records if r["case_id"] in done]
    manifest_path.write_text(json.dumps(entries, indent=1))

    size_gb = sum(Path(e["processed"]).stat().st_size for e in entries) / 1e9
    print(f"Saved {len(entries)} cases ({size_gb:.1f} GB) to {out_dir}")
    if failed:
        print(f"{len(failed)} cases failed:")
        for case_id, err in failed:
            print(f"  {case_id}: {err}")


if __name__ == "__main__":
    main()