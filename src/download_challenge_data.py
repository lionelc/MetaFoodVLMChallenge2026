#!/usr/bin/env python3
"""
Download the Dishcovery competition data with KaggleHub.

The script does not require credentials to be embedded in source. Provide them
using one of:
  - KAGGLE_USERNAME and KAGGLE_KEY environment variables
  - KAGGLE_API_TOKEN='{"username":"...","key":"..."}'
  - secrets/kaggle.json next to this script
  - kaggle.json next to this script
  - ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPETITION = "dishcovery-mission-ii-cvpr-2026"


def configure_kaggle_credentials(kaggle_json: Path | None) -> str:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username or key:
        if not username or not key:
            raise ValueError("Set both KAGGLE_USERNAME and KAGGLE_KEY, or neither.")
        return "KAGGLE_USERNAME/KAGGLE_KEY environment variables"

    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        token = token.strip()
        if token.startswith("{"):
            payload = json.loads(token)
            if payload.get("username") and payload.get("key"):
                os.environ["KAGGLE_USERNAME"] = payload["username"]
                os.environ["KAGGLE_KEY"] = payload["key"]
                return "KAGGLE_API_TOKEN JSON environment variable"
        return "KAGGLE_API_TOKEN environment variable"

    candidates = []
    if kaggle_json is not None:
        candidates.append(kaggle_json)
    candidates.extend([
        ROOT_DIR / "secrets" / "kaggle.json",
        ROOT_DIR / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ])

    for candidate in candidates:
        if candidate.exists():
            candidate = candidate.resolve()
            with candidate.open() as handle:
                payload = json.load(handle)
            if not payload.get("username") or not payload.get("key"):
                raise ValueError(f"{candidate} must contain username and key fields.")
            os.chmod(candidate, stat.S_IRUSR | stat.S_IWUSR)
            os.environ["KAGGLE_CONFIG_DIR"] = str(candidate.parent)
            return str(candidate)

    raise FileNotFoundError(
        "No Kaggle credentials found. Set KAGGLE_USERNAME/KAGGLE_KEY, set "
        "KAGGLE_API_TOKEN JSON, place kaggle.json in secrets/kaggle.json, "
        "or pass --kaggle-json."
    )


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        print(f"Keeping existing {dst.relative_to(ROOT_DIR)}")
        return
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def materialize_download(download_path: Path, mode: str) -> None:
    for name in ["Test1", "Test2"]:
        src = download_path / name
        if not src.exists():
            print(f"Did not find {name}/ under {download_path}; leaving downloaded files in place.")
            continue
        link_or_copy(src, ROOT_DIR / name, mode)
        print(f"Prepared {name}/ from {src}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--kaggle-json", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Set KAGGLEHUB_CACHE before importing kagglehub.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Pass KaggleHub output_dir to download directly into this folder.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--materialize",
        choices=["none", "symlink", "copy"],
        default="symlink",
        help="Expose downloaded Test1/Test2 in this folder for the reproduction pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    credential_source = configure_kaggle_credentials(args.kaggle_json)

    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = str(args.cache_dir.resolve())
    if args.download_dir is not None:
        args.download_dir.mkdir(parents=True, exist_ok=True)

    import kagglehub

    print(f"Using Kaggle credentials from: {credential_source}")
    if args.cache_dir is not None:
        print(f"KaggleHub cache: {args.cache_dir.resolve()}")
    output_dir = str(args.download_dir.resolve()) if args.download_dir is not None else None
    if output_dir is not None:
        print(f"KaggleHub output_dir: {output_dir}")

    path = Path(kagglehub.competition_download(
        args.competition,
        force_download=args.force_download,
        output_dir=output_dir,
    )).resolve()
    print("Path to competition files:", path)

    if args.materialize != "none":
        materialize_download(path, args.materialize)


if __name__ == "__main__":
    main()
