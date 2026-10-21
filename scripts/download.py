"""Download BraTS 2024 Adult Glioma Post-Treatment data from Synapse.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --synapse-id synXXXXXXXX --no-extract

Requires SYNAPSE_TOKEN in the environment or in a .env file in the repo root.
Data goes to $DATA_DIR/raw (default: data/raw).
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import synapseclient
import synapseutils
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SYNAPSE_ID = "syn59059776"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument(
        "--synapse-id",
        default=DEFAULT_SYNAPSE_ID,
        help=f"Synapse ID of the file or folder to download (default: {DEFAULT_SYNAPSE_ID})",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Target directory (default: $DATA_DIR/raw, or data/raw)",
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not extract downloaded .zip archives",
    )

    return parser.parse_args()


def resolve_out_dir(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    data_dir = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
    return (data_dir / "raw").resolve()


def login() -> synapseclient.Synapse:
    load_dotenv(REPO_ROOT / ".env")

    token = os.getenv("SYNAPSE_TOKEN")

    if not token:
        sys.exit(
            "SYNAPSE_TOKEN not found. Put it in a .env file in the repo root "
            "or export it as an environment variable."
        )
    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    return syn


def extract_archives(paths: list[Path]) -> None:
    for archive in paths:
        if archive.suffix != ".zip":
            continue

        marker = archive.with_suffix(".extracted")

        if marker.exists():
            print(f"Already extracted, skipping: {archive.name}")
            continue

        print(f"Extracting {archive.name} ...")

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(archive.parent)
            
        marker.touch()


def main() -> None:
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syn = login()
    print(f"Downloading {args.synapse_id} to {out_dir} ...")

    # keep.local: files already present locally are not downloaded again
    files = synapseutils.syncFromSynapse(
        syn, args.synapse_id, path=str(out_dir), ifcollision="keep.local"
    )
    downloaded = [Path(f.path) for f in files if getattr(f, "path", None)]
    print(f"Done: {len(downloaded)} file(s) in {out_dir}")

    if not args.no_extract:
        extract_archives(downloaded)


if __name__ == "__main__":
    main()