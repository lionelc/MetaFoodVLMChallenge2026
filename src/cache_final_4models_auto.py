#!/usr/bin/env python3
"""
Cache the four embedding models used by the final reproduction pipeline.

Outputs:
  siglip2_cache/
    test1_image_features.npy, test1_text_features.npy,
    test2_image_features.npy, test2_text_features.npy
  qwen3_cache/
    qwen3vl8b_test1_img.npy, qwen3vl8b_test1_text.npy,
    qwen3vl8b_test2_img.npy, qwen3vl8b_test2_text.npy
  clip_cache/vit_h_14_dfn5b/
    test1_image_features.npy, test1_text_features.npy,
    test2_image_features.npy, test2_text_features.npy
  qwen3vl2b_cache/
    test1_image_features.npy, test1_text_features.npy,
    test2_image_features.npy, test2_text_features.npy

The parent process checks available GPU memory and chooses auto concurrency:
  >= 72 GiB free: run all 4 model cache jobs concurrently
  >= 40 GiB free: run 2 jobs concurrently
  otherwise:      run 1 job at a time

Use --concurrency to override.  Each child process loads exactly one model and
skips already-complete cache files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
TEST1_DIR = ROOT_DIR / "Test1"
TEST2_DIR = ROOT_DIR / "Test2"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self) -> int:
        return self.streams[0].fileno()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)


def load_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def load_captions(path: Path) -> list[str]:
    if path.suffix == ".json":
        return json.load(path.open())
    return load_lines(path)


def safe_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def gpu_memory_info() -> dict[str, Any]:
    torch = safe_import_torch()
    if torch is not None and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return {
            "source": "torch.cuda.mem_get_info",
            "cuda": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "free_bytes": int(free),
            "total_bytes": int(total),
            "free_gib": free / 1024**3,
            "total_gib": total / 1024**3,
        }

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        first = proc.stdout.strip().splitlines()[0]
        name, free_mib, total_mib = [item.strip() for item in first.split(",")]
        free = int(free_mib) * 1024**2
        total = int(total_mib) * 1024**2
        return {
            "source": "nvidia-smi",
            "cuda": True,
            "device_count": len(proc.stdout.strip().splitlines()),
            "device_name": name,
            "free_bytes": free,
            "total_bytes": total,
            "free_gib": free / 1024**3,
            "total_gib": total / 1024**3,
        }
    except Exception:
        return {
            "source": "none",
            "cuda": False,
            "device_count": 0,
            "device_name": None,
            "free_bytes": 0,
            "total_bytes": 0,
            "free_gib": 0.0,
            "total_gib": 0.0,
        }


def auto_concurrency(memory: dict[str, Any]) -> int:
    if not memory["cuda"]:
        return 1
    free_gib = float(memory["free_gib"])
    if free_gib >= 72:
        return 4
    if free_gib >= 40:
        return 2
    return 1


def shape_ok(path: Path, expected_rows: int, expected_dim: int) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (expected_rows, expected_dim)
    except Exception:
        return False


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    np.save(tmp, array.astype(np.float32, copy=False))
    tmp_npy = tmp if tmp.suffix == ".npy" else Path(str(tmp) + ".npy")
    os.replace(tmp_npy, path)


@dataclass(frozen=True)
class CacheFile:
    key: str
    path: Path
    rows: int
    dim: int
    kind: str
    split: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    display: str
    cache_dir: Path
    dim: int
    model_name: str
    pretrained: str | None
    default_image_batch: int
    h200_image_batch: int
    text_batch: int
    files: tuple[CacheFile, ...]


def build_specs() -> dict[str, ModelSpec]:
    n_t1_img = len(load_lines(TEST1_DIR / "images.txt"))
    n_t1_txt = len(load_captions(TEST1_DIR / "captions.txt"))
    n_t2_img = len(load_lines(TEST2_DIR / "images.txt"))
    n_t2_txt = len(load_captions(TEST2_DIR / "captions.json"))

    def files(cache_dir: Path, dim: int, names: dict[str, str]) -> tuple[CacheFile, ...]:
        return (
            CacheFile("test1_text", cache_dir / names["test1_text"], n_t1_txt, dim, "text", "test1"),
            CacheFile("test1_image", cache_dir / names["test1_image"], n_t1_img, dim, "image", "test1"),
            CacheFile("test2_text", cache_dir / names["test2_text"], n_t2_txt, dim, "text", "test2"),
            CacheFile("test2_image", cache_dir / names["test2_image"], n_t2_img, dim, "image", "test2"),
        )

    return {
        "siglip2": ModelSpec(
            key="siglip2",
            family="open_clip",
            display="SigLIP2 ViT-B-16-SigLIP2-256 webli",
            cache_dir=ROOT_DIR / "siglip2_cache",
            dim=768,
            model_name="ViT-B-16-SigLIP2-256",
            pretrained="webli",
            default_image_batch=64,
            h200_image_batch=256,
            text_batch=8192,
            files=files(
                ROOT_DIR / "siglip2_cache",
                768,
                {
                    "test1_text": "test1_text_features.npy",
                    "test1_image": "test1_image_features.npy",
                    "test2_text": "test2_text_features.npy",
                    "test2_image": "test2_image_features.npy",
                },
            ),
        ),
        "clip_h": ModelSpec(
            key="clip_h",
            family="open_clip",
            display="CLIP ViT-H-14 dfn5b",
            cache_dir=ROOT_DIR / "clip_cache" / "vit_h_14_dfn5b",
            dim=1024,
            model_name="ViT-H-14",
            pretrained="dfn5b",
            default_image_batch=32,
            h200_image_batch=128,
            text_batch=8192,
            files=files(
                ROOT_DIR / "clip_cache" / "vit_h_14_dfn5b",
                1024,
                {
                    "test1_text": "test1_text_features.npy",
                    "test1_image": "test1_image_features.npy",
                    "test2_text": "test2_text_features.npy",
                    "test2_image": "test2_image_features.npy",
                },
            ),
        ),
        "qwen3_8b": ModelSpec(
            key="qwen3_8b",
            family="qwen_st",
            display="Qwen3-VL-Embedding-8B",
            cache_dir=ROOT_DIR / "qwen3_cache",
            dim=4096,
            model_name="Qwen/Qwen3-VL-Embedding-8B",
            pretrained=None,
            default_image_batch=4,
            h200_image_batch=16,
            text_batch=64,
            files=files(
                ROOT_DIR / "qwen3_cache",
                4096,
                {
                    "test1_text": "qwen3vl8b_test1_text.npy",
                    "test1_image": "qwen3vl8b_test1_img.npy",
                    "test2_text": "qwen3vl8b_test2_text.npy",
                    "test2_image": "qwen3vl8b_test2_img.npy",
                },
            ),
        ),
        "qwen3_2b": ModelSpec(
            key="qwen3_2b",
            family="qwen_st",
            display="Qwen3-VL-Embedding-2B",
            cache_dir=ROOT_DIR / "qwen3vl2b_cache",
            dim=2048,
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            pretrained=None,
            default_image_batch=16,
            h200_image_batch=48,
            text_batch=128,
            files=files(
                ROOT_DIR / "qwen3vl2b_cache",
                2048,
                {
                    "test1_text": "test1_text_features.npy",
                    "test1_image": "test1_image_features.npy",
                    "test2_text": "test2_text_features.npy",
                    "test2_image": "test2_image_features.npy",
                },
            ),
        ),
    }


def missing_files(spec: ModelSpec, force: bool) -> list[CacheFile]:
    if force:
        return list(spec.files)
    return [item for item in spec.files if not shape_ok(item.path, item.rows, item.dim)]


def load_split_data(split: str) -> tuple[list[str], list[str], Path]:
    if split == "test1":
        return (
            load_captions(TEST1_DIR / "captions.txt"),
            load_lines(TEST1_DIR / "images.txt"),
            TEST1_DIR / "imgs",
        )
    if split == "test2":
        return (
            load_captions(TEST2_DIR / "captions.json"),
            load_lines(TEST2_DIR / "images.txt"),
            TEST2_DIR / "imgs",
        )
    raise ValueError(split)


def cache_open_clip(spec: ModelSpec, needed: list[CacheFile], image_batch: int, text_batch: int) -> None:
    import torch
    import open_clip
    from PIL import Image
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{spec.key}] Loading {spec.display} on {device}", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.model_name,
        pretrained=spec.pretrained,
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(spec.model_name)

    def text_features(captions: list[str]) -> np.ndarray:
        chunks = []
        for start in tqdm(range(0, len(captions), text_batch), desc=f"{spec.key} text", unit="batch"):
            chunk = captions[start:start + text_batch]
            tokens = tokenizer(chunk).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.cpu().float().numpy())
        return np.concatenate(chunks, axis=0)

    def image_features(image_dir: Path, image_names: list[str]) -> np.ndarray:
        num_workers = min(16, os.cpu_count() or 8)

        def load_and_preprocess(image_name: str):
            try:
                with Image.open(image_dir / image_name) as img:
                    return preprocess(img.convert("RGB"))
            except Exception as exc:
                print(f"[{spec.key}] warning: {image_name}: {exc}", flush=True)
                return preprocess(Image.new("RGB", (256, 256)))

        chunks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            for start in tqdm(range(0, len(image_names), image_batch), desc=f"{spec.key} images", unit="batch"):
                batch_names = image_names[start:start + image_batch]
                images = list(pool.map(load_and_preprocess, batch_names))
                tensor = torch.stack(images).to(device)
                with torch.no_grad():
                    feats = model.encode_image(tensor)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                chunks.append(feats.cpu().float().numpy())
        return np.concatenate(chunks, axis=0)

    data_cache: dict[str, tuple[list[str], list[str], Path]] = {}
    for item in needed:
        captions, image_names, image_dir = data_cache.setdefault(item.split, load_split_data(item.split))
        print(f"[{spec.key}] Computing {item.key} -> {item.path.relative_to(ROOT_DIR)}", flush=True)
        array = text_features(captions) if item.kind == "text" else image_features(image_dir, image_names)
        atomic_save_npy(item.path, array)
        print(f"[{spec.key}] Saved {item.path.relative_to(ROOT_DIR)} {array.shape}", flush=True)

    meta = {
        "model": spec.model_name,
        "pretrained": spec.pretrained,
        "embed_dim": spec.dim,
        "device": str(device),
        "image_batch": image_batch,
        "text_batch": text_batch,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (spec.cache_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def patch_sentence_transformer_length(model: Any) -> None:
    from PIL import Image

    length_attr = "_text_length" if hasattr(model, "_text_length") else "_input_length"
    original = getattr(model, length_attr)

    def safe_length(value):
        if isinstance(value, (dict, Image.Image)):
            return 1
        return original(value)

    setattr(model, length_attr, safe_length)


def cache_qwen_st(
    spec: ModelSpec,
    needed: list[CacheFile],
    image_batch: int,
    text_batch: int,
    qwen_dtype: str,
) -> None:
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "auto": torch.bfloat16 if torch.cuda.is_available() else None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[qwen_dtype]
    model_kwargs: dict[str, Any] = {
        "attn_implementation": "sdpa",
        "device_map": device,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    print(f"[{spec.key}] Loading {spec.model_name} on {device} dtype={qwen_dtype}", flush=True)
    model = SentenceTransformer(
        spec.model_name,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    patch_sentence_transformer_length(model)

    def text_features(captions: list[str]) -> np.ndarray:
        chunks = []
        for start in tqdm(range(0, len(captions), text_batch), desc=f"{spec.key} text", unit="batch"):
            chunk = captions[start:start + text_batch]
            embs = model.encode(chunk, show_progress_bar=False, normalize_embeddings=True)
            chunks.append(embs.astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)

    def image_features(image_dir: Path, image_names: list[str]) -> np.ndarray:
        num_workers = min(16, os.cpu_count() or 8)

        def load_one(image_name: str):
            try:
                with Image.open(image_dir / image_name) as img:
                    return {"image": img.convert("RGB")}
            except Exception as exc:
                print(f"[{spec.key}] warning: {image_name}: {exc}", flush=True)
                return {"image": Image.new("RGB", (256, 256))}

        chunks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            for start in tqdm(range(0, len(image_names), image_batch), desc=f"{spec.key} images", unit="batch"):
                batch_names = image_names[start:start + image_batch]
                images = list(pool.map(load_one, batch_names))
                embs = model.encode(images, show_progress_bar=False, normalize_embeddings=True)
                chunks.append(embs.astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)

    data_cache: dict[str, tuple[list[str], list[str], Path]] = {}
    for item in needed:
        captions, image_names, image_dir = data_cache.setdefault(item.split, load_split_data(item.split))
        print(f"[{spec.key}] Computing {item.key} -> {item.path.relative_to(ROOT_DIR)}", flush=True)
        array = text_features(captions) if item.kind == "text" else image_features(image_dir, image_names)
        atomic_save_npy(item.path, array)
        print(f"[{spec.key}] Saved {item.path.relative_to(ROOT_DIR)} {array.shape}", flush=True)

    meta = {
        "model": spec.model_name,
        "embed_dim": spec.dim,
        "device": device,
        "image_batch": image_batch,
        "text_batch": text_batch,
        "qwen_dtype": qwen_dtype,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (spec.cache_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def batch_for_model(spec: ModelSpec, memory: dict[str, Any], concurrency: int, overrides: dict[str, int]) -> int:
    if spec.key in overrides:
        return overrides[spec.key]
    if memory["cuda"] and float(memory["total_gib"]) >= 70 and concurrency <= 2:
        return spec.h200_image_batch
    if memory["cuda"] and float(memory["total_gib"]) >= 70 and concurrency == 4:
        return max(spec.default_image_batch, spec.h200_image_batch // 2)
    return spec.default_image_batch


def run_worker(args: argparse.Namespace) -> int:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = None
    if args.worker_log is not None:
        args.worker_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.worker_log.open("w", buffering=1)
        sys.stdout = Tee(original_stdout, log_handle)
        sys.stderr = Tee(original_stderr, log_handle)

    specs = build_specs()
    spec = specs[args.worker]
    try:
        if args.worker_log is not None:
            print(f"[{spec.key}] Worker log: {args.worker_log}", flush=True)
        needed = missing_files(spec, args.force)
        if not needed:
            print(f"[{spec.key}] All cache files already complete; skipping.", flush=True)
            return 0
        print(f"[{spec.key}] Starting; {len(needed)} cache file(s) to build.", flush=True)
        t_start = time.time()
        spec.cache_dir.mkdir(parents=True, exist_ok=True)
        if spec.family == "open_clip":
            cache_open_clip(spec, needed, args.image_batch, args.text_batch)
        elif spec.family == "qwen_st":
            cache_qwen_st(spec, needed, args.image_batch, args.text_batch, args.qwen_dtype)
        else:
            raise ValueError(spec.family)
        print(f"[{spec.key}] Done in {(time.time() - t_start) / 60:.1f} min", flush=True)
        return 0
    finally:
        if log_handle is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()


def run_child_process(
    spec: ModelSpec,
    args: argparse.Namespace,
    image_batch: int,
    text_batch: int,
    log_path: Path | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        spec.key,
        "--image-batch",
        str(image_batch),
        "--text-batch",
        str(text_batch),
        "--qwen-dtype",
        args.qwen_dtype,
    ]
    if args.force:
        cmd.append("--force")
    if log_path is not None:
        cmd.extend(["--worker-log", str(log_path)])

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    t_start = time.time()
    log_msg = f" log={display_path(log_path)}" if log_path is not None else ""
    print(
        f"[parent] launching {spec.key}: image_batch={image_batch} text_batch={text_batch}{log_msg}",
        flush=True,
    )
    proc = subprocess.run(cmd, cwd=ROOT_DIR, env=env)
    return {
        "model": spec.key,
        "returncode": proc.returncode,
        "elapsed_sec": time.time() - t_start,
        "image_batch": image_batch,
        "text_batch": text_batch,
        "cmd": cmd,
        "log_path": display_path(log_path) if log_path is not None else None,
    }


def parse_batch_overrides(raw: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Expected MODEL=BATCH for --batch-override, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = int(value)
    return overrides


def print_plan(
    specs: dict[str, ModelSpec],
    selected: list[str],
    memory: dict[str, Any],
    concurrency: int,
    batches: dict[str, int],
    force: bool,
) -> list[dict[str, Any]]:
    rows = []
    print("\nGPU/memory:")
    print(f"  cuda={memory['cuda']} device={memory['device_name']} free={memory['free_gib']:.1f}GiB total={memory['total_gib']:.1f}GiB")
    print(f"  selected concurrency={concurrency}")
    print("\nCache plan:")
    for key in selected:
        spec = specs[key]
        needed = missing_files(spec, force)
        status = "run" if needed else "skip"
        print(f"  {key:9s} {status:4s} image_batch={batches[key]:3d} text_batch={spec.text_batch:5d} missing={len(needed)}")
        for item in needed:
            print(f"      - {item.path.relative_to(ROOT_DIR)} expected=({item.rows}, {item.dim})")
        rows.append({
            "model": key,
            "status": status,
            "image_batch": batches[key],
            "text_batch": spec.text_batch,
            "missing": [str(item.path.relative_to(ROOT_DIR)) for item in needed],
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["qwen3_8b", "qwen3_2b", "siglip2", "clip_h"])
    parser.add_argument("--concurrency", choices=["auto", "1", "2", "4"], default="auto")
    parser.add_argument("--batch-override", nargs="*", default=[], help="Override image batch size, e.g. qwen3_8b=12 clip_h=96")
    parser.add_argument("--qwen-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--force", action="store_true", help="Recompute cache files even when complete files exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the auto plan without running model jobs.")
    parser.add_argument("--manifest-out", type=Path, default=ROOT_DIR / "optimization" / "experiment_results" / "final_4model_cache_manifest.json")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for per-model worker logs.")

    worker = parser.add_argument_group("internal worker options")
    worker.add_argument("--worker", choices=["siglip2", "clip_h", "qwen3_8b", "qwen3_2b"], default=None)
    worker.add_argument("--image-batch", type=int, default=0)
    worker.add_argument("--text-batch", type=int, default=0)
    worker.add_argument("--worker-log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        raise SystemExit(run_worker(args))

    t_start = time.time()
    specs = build_specs()
    selected = args.models
    unknown = [key for key in selected if key not in specs]
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}; choices are {sorted(specs)}")

    memory = gpu_memory_info()
    concurrency = auto_concurrency(memory) if args.concurrency == "auto" else int(args.concurrency)
    overrides = parse_batch_overrides(args.batch_override)
    batches = {
        key: batch_for_model(specs[key], memory, concurrency, overrides)
        for key in selected
    }
    plan = print_plan(specs, selected, memory, concurrency, batches, args.force)
    if args.dry_run:
        return

    runnable = [key for key in selected if missing_files(specs[key], args.force)]
    results: list[dict[str, Any]] = []
    log_dir = None
    if runnable:
        log_dir = args.log_dir
        if log_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            log_dir = args.manifest_out.parent / f"final_4model_cache_logs_{stamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRunning {len(runnable)} model cache job(s) with concurrency={concurrency}...\n", flush=True)
        print(f"Queue order: {', '.join(runnable)}", flush=True)
        print(f"Worker logs: {display_path(log_dir)}\n", flush=True)
    # Submit in waves of `concurrency` to guarantee pairing order.
    for wave_start in range(0, len(runnable), concurrency):
        wave = runnable[wave_start:wave_start + concurrency]
        print(f"[parent] Wave {wave_start // concurrency + 1}: {', '.join(wave)}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave)) as pool:
            future_to_key = {
                pool.submit(
                    run_child_process,
                    specs[key],
                    args,
                    batches[key],
                    specs[key].text_batch,
                    log_dir / f"{key}.log" if log_dir is not None else None,
                ): key
                for key in wave
            }
            for future in concurrent.futures.as_completed(future_to_key):
                result = future.result()
                results.append(result)
                if result["returncode"] != 0:
                    raise RuntimeError(f"{result['model']} failed with exit code {result['returncode']}")
                print(f"[parent] {result['model']} completed in {result['elapsed_sec'] / 60:.1f} min", flush=True)

    final_status = {}
    for key in selected:
        spec = specs[key]
        final_status[key] = {
            str(item.path.relative_to(ROOT_DIR)): shape_ok(item.path, item.rows, item.dim)
            for item in spec.files
        }
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT_DIR)),
        "memory": memory,
        "concurrency": concurrency,
        "qwen_dtype": args.qwen_dtype,
        "force": args.force,
        "log_dir": display_path(log_dir) if log_dir is not None else None,
        "plan": plan,
        "results": results,
        "final_status": final_status,
        "elapsed_sec": time.time() - t_start,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nSaved manifest: {args.manifest_out}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
