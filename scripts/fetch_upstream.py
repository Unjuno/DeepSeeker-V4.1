#!/usr/bin/env python3
"""Fetch the official DeepSeek-V4.1-Flash reference implementation.

Only reference code/configuration/license metadata are downloaded.
Model weight shards are deliberately excluded.

The exact resolved Hugging Face revision is recorded in the destination so
DeepSeeker experiments can be reproduced against a known upstream snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_DEST = Path("upstream/DeepSeek-V4.1-Flash")

ALLOW_PATTERNS = [
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "inference/*",
    "inference/**/*",
    "encoding/*",
    "encoding/**/*",
]

IGNORE_PATTERNS = [
    "*.safetensors",
    "*.gguf",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch DeepSeek-V4.1-Flash reference code without weights."
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch/tag/commit to resolve (default: main).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()

    info = api.model_info(REPO_ID, revision=args.revision)
    resolved_sha = info.sha
    if not resolved_sha:
        raise RuntimeError("Could not resolve an immutable upstream revision.")

    args.dest.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        revision=resolved_sha,
        local_dir=args.dest,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    manifest = {
        "repo_id": REPO_ID,
        "requested_revision": args.revision,
        "resolved_revision": resolved_sha,
        "source": f"https://huggingface.co/{REPO_ID}",
        "weights_downloaded": False,
    }
    (args.dest / "DEEPSEEKER_UPSTREAM.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Fetched {REPO_ID}")
    print(f"Resolved revision: {resolved_sha}")
    print(f"Destination: {args.dest}")
    print("Weights: excluded")


if __name__ == "__main__":
    main()
