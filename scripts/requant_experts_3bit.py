"""Rebuild routed experts as MLX-native int3 affine (group_size=64, bits=3).

Quality-first pipeline (docs/QUANT_RESEARCH.md): dequantize each routed
experts tensor out of the official MXFP4 checkpoint, requantize it with
mx.quantize (the MLX-native affine matmul path the runner consumes, no
custom packing), and write

  .weight  uint32 words (MLX internal 3-bit pack order - opaque)
  .scale   fp16, one scale per 64-wide group
  .bias    fp16, MLX affine zero-point per group

into a side payload file. Per expert the payload layout keeps the same
merged-run shape as MXFP4: a contiguous uint32 weight run (w1, w2, w3),
then a contiguous fp16 scale run, then a contiguous fp16 bias run. Two
kinds per run keeps `pread`-direct loading节气 valid (one width per run).

Unconverted tensors keep their original shard entries; the Q3 model root
symlinks the untouched shard files so every other tensor read stays
identical to the 1017 checkpoint.

Scope defaults to layers 20-39 (sensitivity studies put early blocks at
the highest FP4 risk, so 3-bit moves stay in the later layers). MTP
experts stay MXFP4, out of scope. Resumable: converted experts are
skipped, per-tensor relative RMS sits in --quality-log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepseeker.mlx_weights import (
    Q3_BITS,
    Q3_GROUP_SIZE,
    dequant_fp4_rowmajor,
    read_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLES = ("w1", "w2", "w3")
FP4_EXPERT_BYTES = 18800640


def parse_layers(arg: str) -> list[int]:
    out: list[int] = []
    for part in arg.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_experts(arg: str | None, total: int = 384) -> list[int]:
    if not arg:
        return list(range(total))
    return parse_layers(arg)


def convert_tensor(vals: np.ndarray, rows: int, cols: int,
                   bits: int) -> tuple:
    gpu = mx.array(vals.reshape(rows, cols).astype(np.float32))
    packed, scales, biases = mx.quantize(gpu, group_size=Q3_GROUP_SIZE,
                                         bits=bits)
    roundtrip = mx.dequantize(packed, scales=scales, biases=biases,
                              group_size=Q3_GROUP_SIZE, bits=bits)
    rt_scale = max(float(mx.abs(roundtrip).max()), 1e-9)
    err = np.asarray(gpu, dtype=np.float64) - np.asarray(roundtrip, np.float64)
    rms = float(np.sqrt(np.mean(np.square(err))) / rt_scale)
    os.environ["DEEPSEEKER_Q3_BITS"] = str(bits)
    words = np.asarray(packed).astype("<u4").tobytes()
    sraw = np.asarray(scales.astype(mx.float16)).astype("<f2").tobytes()
    braw = np.asarray(biases.astype(mx.float16)).astype("<f2").tobytes()
    return words, sraw, braw, rms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="20-39")
    parser.add_argument("--experts", default=None,
                        help="subset, e.g. 0-15 or 0,1,2 (default all 384)")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--quality-log", type=Path,
                        default=REPO_ROOT / "work" / "q3-quality.jsonl")
    parser.add_argument("--bits", type=int, default=int(
        os.environ.get("DEEPSEEKER_Q3_BITS", "3")),
        help="quantize bit depth (3 pilot default; 4 = near-lossless sanity probe)")
    parser.add_argument("--payload-name", default="routed-q3.payload")
    parser.add_argument("--fresh", action="store_true",
                        help="start from the pristine 1017 manifest (ignore prior q3 conversion state)")
    args = parser.parse_args()

    snap = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash-1017"
    revision = json.loads(
        (snap / "DEEPSEEKER_UPSTREAM.json").read_text())["resolved_revision"]
    profile_dir = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision
    manifest_name = "model-manifest-q3.json"
    if args.fresh or args.bits != 3 or args.payload_name != "routed-q3.payload":
        manifest_name = "model-manifest.json"
    manifest = json.loads((profile_dir / manifest_name).read_text())

    layers = parse_layers(args.layers)
    expert_ids = parse_experts(args.experts)

    by_name = {t["name"]: t for t in manifest["tensors"]}
    source_root = REPO_ROOT / "models" / snap.name
    if not source_root.exists():
        raise SystemExit(f"weights not found under {source_root}")

    out_root = args.out_root or (REPO_ROOT / "models" / f"{snap.name}-Q3")
    out_root.mkdir(exist_ok=True)
    for item in source_root.iterdir():
        link = out_root / item.name
        if not link.exists():
            link.symlink_to(item.resolve() if item.is_file() else item)

    payload_path = out_root / args.payload_name
    if not payload_path.exists():
        open(payload_path, "wb").close()
    payload_size = payload_path.stat().st_size
    payload = open(payload_path, "r+b")  # noqa: SIM115
    payload.seek(payload_size)

    quality_log = open(args.quality_log, "a")  # noqa: SIM115

    def tensor_dims(role: str) -> tuple[int, int, int, int, int]:
        """(rows, pair_cols, cols, words, groups) for a converted role.

        Fixed model geometry from config.json: w1/w3 map 2304 outputs over
        5120 inputs; w2 maps 5120 over 2304. Never read back from the
        in-flight manifest - converted expert entries have rewritten shapes.
        """
        if role == "w2":
            rows, cols = 5120, 2304
        else:
            rows, cols = 2304, 5120
        words = (cols * Q3_BITS + 31) // 32
        groups = cols // Q3_GROUP_SIZE
        return rows, cols // 2, cols, words, groups

    pending: list[tuple[int, int]] = []
    for layer in layers:
        for expert in expert_ids:
            base = f"layers.{layer}.ffn.experts.{expert}"
            if all(by_name[f"{base}.{r}.weight"].get("dtype")
                   == "U32_Q3_AFFINE" for r in ROLES):
                continue
            pending.append((layer, expert))
    target = len(layers) * len(expert_ids)
    print(f"{len(pending)}/{target} experts to convert; payload starts at "
          f"{payload_size / 1e9:.2f} GB")

    out_manifest = profile_dir / (
        "model-manifest-q3.json" if args.bits == 3
        and args.payload_name == "routed-q3.payload"
        and not args.fresh else "model-manifest-qprobe.json")
    start_clock = time.perf_counter()
    done = written_total = 0
    for (layer, expert) in pending:
        base = f"layers.{layer}.ffn.experts.{expert}"
        blocks: dict[str, bytes] = {}
        for role in ROLES:
            rows, _pair, cols, _words, _groups = tensor_dims(role)
            wentry = by_name[f"{base}.{role}.weight"]
            sentry = by_name[f"{base}.{role}.scale"]
            wdata = read_bytes(source_root, wentry["shard"],
                               *wentry["file_range"])
            sdata = read_bytes(source_root, sentry["shard"],
                               *sentry["file_range"])
            vals = dequant_fp4_rowmajor(
                np.frombuffer(wdata, dtype=np.uint8),
                np.frombuffer(sdata, dtype=np.uint8))
            words, sraw, braw, rms = convert_tensor(vals, rows, cols,
                                                    args.bits)
            blocks[f"{base}.{role}.weight"] = words
            blocks[f"{base}.{role}.scale"] = sraw
            blocks[f"{base}.{role}.bias"] = braw
            quality_log.write(json.dumps(
                {"tensor": f"{base}.{role}", "rms_rel": round(rms, 8)}) + "\n")
            quality_log.flush()

        # Payload block order per expert: weights, then scales, then biases
        # (two contiguous runs: uint32 and fp16), then manifest ranges.
        ordered = [f"{base}.{r}.{part}" for part in ("weight", "scale", "bias")
                   for r in ROLES]
        new_entries: dict[str, dict] = {}
        for name in ordered:
            role, part = name.rsplit(".", 2)[-2], name.rsplit(".", 2)[-1]
            start = payload.tell()
            blob = blocks[name]
            payload.write(blob)
            rows, _pair, cols, words, groups = tensor_dims(role)
            if part == "weight":
                shape, dims, dtype = [rows, words], len(blob) // 4, "U32_Q3_AFFINE"
            else:
                shape, dims, dtype = [rows, groups], len(blob) // 2, "F16"
            new_entries[part + role] = {
                "name": name, "shape": shape, "dtype": dtype,
                "elements": dims, "bytes": len(blob),
                "shard": args.payload_name,
                "data_range": [start, start + len(blob)],
                "file_range": [start, start + len(blob)],
                "subsystem": "routed-expert", "layer": layer,
                "expert": expert, "routed": True,
                "detail": f"{role}-{part}"}

        # Original weight/scale entries get rewritten in place; the six bias
        # entries are new list entries.
        for role in ROLES:
            for part in ("weight", "scale", "bias"):
                new = new_entries[part + role]
                if part == "bias":
                    manifest["tensors"].append(new)
                    by_name[new["name"]] = new
                else:
                    by_name[f"{base}.{role}.{part}"].update(new)

        done += 1
        expert_bytes = sum(len(b) for b in blocks.values())
        written_total += expert_bytes
        if done % 128 == 0:
            payload.flush()
            manifest["q3_requant"] = {
                "payload": args.payload_name,
                "group_size": Q3_GROUP_SIZE, "bits": Q3_BITS,
                "layer_range": [layers[0], layers[-1]],
                "experts_converted": done,
                "original_mxfp4_expert_bytes": done * FP4_EXPERT_BYTES,
                "q3_expert_bytes": written_total / done,
                "payload_bytes_written": written_total}
            out_manifest.write_text(json.dumps(manifest, separators=(",", ":")))
            dt = time.perf_counter() - start_clock
            rate = done / max(dt, 1e-9)
            eta = (len(pending) - done) / rate
            print(f"{done}/{len(pending)} {written_total / 1e9:.1f} GB "
                  f"{rate:.1f}/s eta {eta / 60:.0f}m", flush=True)
    payload.flush()

    manifest["q3_requant"] = {
        "payload": args.payload_name,
        "group_size": Q3_GROUP_SIZE, "bits": Q3_BITS,
        "layer_range": [layers[0], layers[-1]],
        "experts_converted": done,
        "original_mxfp4_expert_bytes": done * FP4_EXPERT_BYTES,
        "q3_expert_bytes": written_total // max(done, 1),
        "payload_bytes_written": written_total}
    out_manifest.write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"payload {payload_path.stat().st_size / 1e9:.2f} GB; "
          f"manifest -> {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
