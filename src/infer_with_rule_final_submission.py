#!/usr/bin/env python3
"""
Clean final-submission reproducer.

This script is intentionally a fixed replay of the final selected pipeline.  It
does not rerun local Gemini-judge selection or broad search.  It expects the four
embedding cache folders to exist:

  siglip2_cache/
  qwen3_cache/
  clip_cache/vit_h_14_dfn5b/
  qwen3vl2b_cache/

The Qwen3/Gemma4 description and yes/no caches are resume-safe and are generated
only when the corresponding cache files are missing or incomplete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
OPT_DIR = ROOT_DIR / "optimization"
RESULT_DIR = OPT_DIR / "experiment_results"
SIGLIP_CACHE = ROOT_DIR / "siglip2_cache"
QWEN3_8B_CACHE = ROOT_DIR / "qwen3_cache"
CLIP_CACHE = ROOT_DIR / "clip_cache" / "vit_h_14_dfn5b"
QWEN3_2B_CACHE = ROOT_DIR / "qwen3vl2b_cache"
QWEN3_DESC_CACHE = ROOT_DIR / "description_cache" / "qwen3_de_cache"
GEMMA4_DESC_CACHE = ROOT_DIR / "description_cache" / "gemma4_cache"

FINAL_RECALL_MANIFEST = RESULT_DIR / "test1_recall_recovery_20260501212205.json"
FINAL_PRESET = "p0525_r80"
FINAL_REFERENCE = OPT_DIR / "submission-test1-recall-p0525_r80-test2-safe-20260501212237.csv"

BENCH_SCRIPT = OPT_DIR / "benchmark_test2_matching_signals-20260501.py"
TEST1_SELECTED_SCRIPT = OPT_DIR / "analyze_test1_description_signals-20260501.py"
TEST1_TRIGGERED_SCRIPT = OPT_DIR / "analyze_test1_triggered_disambig_rules-20260501.py"
TEST1_MARGIN_SCRIPT = OPT_DIR / "analyze_test1_triggered_margin_gates-20260501.py"
TEST1_QWEN_GEMMA_SCRIPT = OPT_DIR / "analyze_test1_qwen_gemma_description_ensemble-20260501.py"

TEST1_QWEN_GEMMA_MANIFEST = RESULT_DIR / "test1_qwen_gemma_desc_ensemble_20260501071543.json"
TEST1_SELECTED_MANIFEST = RESULT_DIR / "test1_description_signal_rules_20260501104024.json"
TEST1_MARGIN_MANIFEST = RESULT_DIR / "test1_triggered_margin_gates_20260501165741.json"
TEST1_CLUSTERED_MANIFEST = RESULT_DIR / "test1_clustered_ab_confusions_20260501171640.json"
TEST1_AGGRESSIVE_MANIFEST = RESULT_DIR / "test1_clustered_ab_confusions_aggressive_20260501185240.json"
TEST1_SECONDPASS_MANIFEST = RESULT_DIR / "test1_clustered_ab_confusions_secondpass_20260501191353.json"

TEST2_CONSERVATIVE_MANIFEST = RESULT_DIR / "test2_clustered_ab_confusions_20260501181702.json"
TEST2_AGGRESSIVE_MANIFEST = RESULT_DIR / "test2_clustered_ab_confusions_aggressive_20260501184656.json"
TEST2_SECONDPASS_MANIFEST = RESULT_DIR / "test2_clustered_ab_confusions_secondpass_20260501190646.json"
TEST2_SAFE_PRESET = "second_max4_safe"

T1_WEIGHTS = [0.25, 0.35, 0.30, 0.10]
T2_WEIGHTS = [0.20, 0.40, 0.15, 0.25]
JACCARD_BONUS = 0.20
SEASONING_THRESHOLD = 3.3
SEASONING_INDICES = frozenset([11, 16, 100, 121, 187, 188, 315, 329, 372])
CONTAINER_THRESHOLD = 3.2
CONTAINER_INDICES = frozenset([41, 357, 365, 382, 384])
GAP_SR = 5
GAP_MAX_K = 5

DESCRIBE_PROMPT = (
    "You are a food image annotator. List every distinct food item visible in "
    "this image. For each item, use the most specific common name (e.g. "
    "'snow peas' not 'greens', 'salmon' not 'fish', 'baguette' not 'bread').\n\n"
    "Organize your answer into these categories, skipping any category with "
    "nothing visible:\n"
    "  PROTEIN: (e.g. grilled chicken, bacon, shrimp, tofu)\n"
    "  STARCH: (e.g. white rice, pasta, baguette, waffle)\n"
    "  VEGETABLE/FRUIT: (e.g. broccoli, cherry tomatoes, watermelon, apple)\n"
    "  SAUCE/CONDIMENT: (e.g. tomato sauce, barbecue sauce, gravy, mayo)\n"
    "  GARNISH: (e.g. basil, sesame seeds, lemon wedge, parsley)\n"
    "  BEVERAGE: (e.g. coffee, orange juice, red wine)\n"
    "  OTHER: (e.g. wafer cone, breading, wrap)\n\n"
    "Output ONLY the categorized list. Do not describe the scene, do not add "
    "commentary, and do not invent items that are not clearly visible."
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "Usage", "class_ids", "Task"])
        writer.writeheader()
        writer.writerows(rows)


def parse_pred_set(value: str) -> set[int]:
    return {int(item) for item in value.split("-") if item}


def format_pred_set(values: set[int]) -> str:
    return "-".join(str(item) for item in sorted(values))


def z_norm(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    mu = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True) + 1e-12
    return (matrix - mu) / std


def adaptive_gap(row: np.ndarray, sr: int = GAP_SR, max_k: int = GAP_MAX_K) -> int:
    ranked = np.sort(row)[::-1]
    s = min(sr, len(ranked) - 1)
    gaps = ranked[:s] - ranked[1:s + 1]
    return max(1, min(int(np.argmax(gaps)) + 1, max_k))


def build_test1_scores() -> np.ndarray:
    print("Building Test1 4-model score matrix...")
    return (
        T1_WEIGHTS[0] * z_norm(np.load(SIGLIP_CACHE / "test1_image_features.npy") @ np.load(SIGLIP_CACHE / "test1_text_features.npy").T)
        + T1_WEIGHTS[1] * z_norm(np.load(QWEN3_8B_CACHE / "qwen3vl8b_test1_img.npy") @ np.load(QWEN3_8B_CACHE / "qwen3vl8b_test1_text.npy").T)
        + T1_WEIGHTS[2] * z_norm(np.load(CLIP_CACHE / "test1_image_features.npy") @ np.load(CLIP_CACHE / "test1_text_features.npy").T)
        + T1_WEIGHTS[3] * z_norm(np.load(QWEN3_2B_CACHE / "test1_image_features.npy") @ np.load(QWEN3_2B_CACHE / "test1_text_features.npy").T)
    )


def build_rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    row_idx = np.arange(scores.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, scores.shape[1] + 1)
    return ranks


def rows_from_pred_sets(preds: list[set[int]]) -> list[dict[str, str]]:
    return [
        {
            "row_id": str(image_idx),
            "Usage": "Public",
            "class_ids": format_pred_set(pred_set),
            "Task": "multi",
        }
        for image_idx, pred_set in enumerate(preds)
    ]


def preds_from_rows(rows: list[dict[str, str]], image_count: int | None = None) -> list[set[int]]:
    count = image_count if image_count is not None else len(rows)
    return [parse_pred_set(rows[idx]["class_ids"]) for idx in range(count)]


def validate_embedding_caches() -> dict[str, Any]:
    test1_images = load_lines(ROOT_DIR / "Test1" / "images.txt")
    test1_caps = load_lines(ROOT_DIR / "Test1" / "captions.txt")
    test2_images = load_lines(ROOT_DIR / "Test2" / "images.txt")
    test2_caps = load_json(ROOT_DIR / "Test2" / "captions.json")

    required = {
        "siglip2": [
            (SIGLIP_CACHE / "test1_image_features.npy", len(test1_images)),
            (SIGLIP_CACHE / "test1_text_features.npy", len(test1_caps)),
            (SIGLIP_CACHE / "test2_image_features.npy", len(test2_images)),
            (SIGLIP_CACHE / "test2_text_features.npy", len(test2_caps)),
        ],
        "qwen3_vl_8b": [
            (QWEN3_8B_CACHE / "qwen3vl8b_test1_img.npy", len(test1_images)),
            (QWEN3_8B_CACHE / "qwen3vl8b_test1_text.npy", len(test1_caps)),
            (QWEN3_8B_CACHE / "qwen3vl8b_test2_img.npy", len(test2_images)),
            (QWEN3_8B_CACHE / "qwen3vl8b_test2_text.npy", len(test2_caps)),
        ],
        "clip_h_dfn5b": [
            (CLIP_CACHE / "test1_image_features.npy", len(test1_images)),
            (CLIP_CACHE / "test1_text_features.npy", len(test1_caps)),
            (CLIP_CACHE / "test2_image_features.npy", len(test2_images)),
            (CLIP_CACHE / "test2_text_features.npy", len(test2_caps)),
        ],
        "qwen3_vl_2b": [
            (QWEN3_2B_CACHE / "test1_image_features.npy", len(test1_images)),
            (QWEN3_2B_CACHE / "test1_text_features.npy", len(test1_caps)),
            (QWEN3_2B_CACHE / "test2_image_features.npy", len(test2_images)),
            (QWEN3_2B_CACHE / "test2_text_features.npy", len(test2_caps)),
        ],
    }

    summary: dict[str, Any] = {}
    missing: list[str] = []
    bad_shape: list[str] = []
    for model_name, files in required.items():
        summary[model_name] = []
        for path, expected_rows in files:
            if not path.exists():
                missing.append(str(path.relative_to(ROOT_DIR)))
                continue
            arr = np.load(path, mmap_mode="r")
            row = {
                "path": str(path.relative_to(ROOT_DIR)),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
            summary[model_name].append(row)
            if arr.shape[0] != expected_rows:
                bad_shape.append(f"{path.relative_to(ROOT_DIR)} expected {expected_rows} rows, got {arr.shape[0]}")

    if missing or bad_shape:
        message = []
        if missing:
            message.append("Missing embedding cache files:\n  " + "\n  ".join(missing))
        if bad_shape:
            message.append("Embedding cache shape mismatches:\n  " + "\n  ".join(bad_shape))
        raise FileNotFoundError("\n".join(message))

    print("Validated four embedding cache folders.")
    return summary


def ensure_description_cache(
    *,
    model_name: str,
    model_id: str,
    cache_path: Path,
    image_dir: Path,
    image_names: list[str],
    generate: bool,
    save_every: int,
    batch_size: int = 4,
) -> dict[str, str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, str] = load_json(cache_path) if cache_path.exists() else {}
    missing = [name for name in image_names if name not in cached]
    print(f"{model_name} descriptions: cached={len(image_names) - len(missing)} missing={len(missing)}")
    if not missing or not generate:
        if missing:
            print(f"  WARNING: {cache_path.relative_to(ROOT_DIR)} has {len(missing)} missing entries — those images will get empty descriptions.")
        return cached

    import torch
    from transformers import AutoProcessor

    if model_name == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration

        print(f"Loading {model_id} for missing Qwen3 descriptions (batch_size={batch_size})...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
    elif model_name == "gemma4":
        from transformers import Gemma4ForConditionalGeneration

        print(f"Loading {model_id} for missing Gemma4 descriptions (batch_size={batch_size})...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
    else:
        raise ValueError(model_name)

    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.eval()

    def _describe_single(img):
        """Fallback: generate description for a single image."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": DESCRIBE_PROMPT},
        ]}]
        if model_name == "qwen3":
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            trimmed = out[:, inputs.input_ids.shape[1]:]
            return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        else:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    processed_count = 0
    try:
        for batch_start in tqdm(range(0, len(missing), batch_size),
                                desc=f"{model_name} descriptions", unit="batch"):
            batch_names = missing[batch_start:batch_start + batch_size]
            batch_images = []
            valid_names = []
            for image_name in batch_names:
                try:
                    img = Image.open(image_dir / image_name).convert("RGB")
                    batch_images.append(img)
                    valid_names.append(image_name)
                except Exception as exc:
                    cached[image_name] = f"ERROR: {type(exc).__name__}: {exc}"

            if not batch_images:
                processed_count += len(batch_names)
                continue

            try:
                batch_texts = []
                for img in batch_images:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": DESCRIBE_PROMPT},
                    ]}]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    batch_texts.append(text)

                inputs = processor(
                    text=batch_texts, images=batch_images,
                    return_tensors="pt", padding=True,
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                prompt_len = inputs.input_ids.shape[1]
                for i, name in enumerate(valid_names):
                    desc = processor.decode(
                        out[i, prompt_len:], skip_special_tokens=True
                    ).strip()
                    cached[name] = desc
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{model_name}] OOM on batch of {len(batch_images)}, falling back to single", flush=True)
                for img, name in zip(batch_images, valid_names):
                    if name in cached:
                        continue
                    try:
                        cached[name] = _describe_single(img)
                    except torch.cuda.OutOfMemoryError as exc2:
                        torch.cuda.empty_cache()
                        cached[name] = f"OOM: {exc2}"
                    except Exception as exc2:
                        cached[name] = f"ERROR: {type(exc2).__name__}: {exc2}"
            except Exception:
                # Batch API error — fall back to single-image processing
                for img, name in zip(batch_images, valid_names):
                    if name in cached:
                        continue
                    try:
                        cached[name] = _describe_single(img)
                    except torch.cuda.OutOfMemoryError as exc2:
                        torch.cuda.empty_cache()
                        cached[name] = f"OOM: {exc2}"
                    except Exception as exc2:
                        cached[name] = f"ERROR: {type(exc2).__name__}: {exc2}"

            processed_count += len(batch_names)
            if processed_count % save_every < batch_size:
                cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True))
        cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True))
    finally:
        del model, processor
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cached


def ensure_description_caches(generate: bool, save_every: int) -> dict[str, Any]:
    test1_images = load_lines(ROOT_DIR / "Test1" / "images.txt")
    test2_images = load_lines(ROOT_DIR / "Test2" / "images.txt")

    print("Building description caches (Qwen3 then Gemma4)...", flush=True)
    summary = {}
    summary["qwen3_test1"] = len(ensure_description_cache(
        model_name="qwen3",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        cache_path=QWEN3_DESC_CACHE / "test1_desc.json",
        image_dir=ROOT_DIR / "Test1" / "imgs",
        image_names=test1_images,
        generate=generate,
        save_every=save_every,
        batch_size=1,
    ))
    summary["qwen3_test2"] = len(ensure_description_cache(
        model_name="qwen3",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        cache_path=QWEN3_DESC_CACHE / "test2_desc.json",
        image_dir=ROOT_DIR / "Test2" / "imgs",
        image_names=test2_images,
        generate=generate,
        save_every=save_every,
        batch_size=1,
    ))
    summary["gemma4_test1"] = len(ensure_description_cache(
        model_name="gemma4",
        model_id="google/gemma-4-E4B-it",
        cache_path=GEMMA4_DESC_CACHE / "test1_desc.json",
        image_dir=ROOT_DIR / "Test1" / "imgs",
        image_names=test1_images,
        generate=generate,
        save_every=save_every,
        batch_size=1,
    ))

    return summary

def build_jaccard_from_sets(img_entities: list[set[str]], cap_entities: list[set[str]]) -> np.ndarray:
    matrix = np.zeros((len(img_entities), len(cap_entities)), dtype=np.float32)
    for image_idx, image_set in enumerate(img_entities):
        if not image_set:
            continue
        for cap_idx, cap_set in enumerate(cap_entities):
            if not cap_set:
                continue
            union = image_set | cap_set
            if union:
                matrix[image_idx, cap_idx] = len(image_set & cap_set) / len(union)
    return matrix


def has_description_caches() -> bool:
    """Check if pre-computed description caches are available."""
    required = [
        QWEN3_DESC_CACHE / "test1_desc.json",
        QWEN3_DESC_CACHE / "test2_desc.json",
        GEMMA4_DESC_CACHE / "test1_desc.json",
    ]
    return all(p.exists() for p in required)


def selected_from_manifest(manifest: dict[str, Any], preset_name: str) -> list[dict[str, Any]]:
    if preset_name == "best":
        return manifest["best"]["selected"]
    for result in manifest["preset_results"]:
        if result["preset"]["name"] == preset_name:
            return result["selected"]
    raise ValueError(f"Preset {preset_name!r} not found")


def apply_test2_candidate(
    score: np.ndarray,
    candidate: dict[str, Any],
    cap_has: dict[str, np.ndarray],
    img_has: dict[str, np.ndarray],
) -> np.ndarray:
    remove_entity = candidate["remove_entity"]
    anchor_entity = candidate["anchor_entity"]
    rows = img_has[anchor_entity] & ~img_has[remove_entity]
    remove_cols = cap_has[remove_entity] & ~cap_has[anchor_entity]
    anchor_cols = cap_has[anchor_entity] & ~cap_has[remove_entity]
    out = score.copy()
    if candidate["op"] in ("penalty", "both"):
        out[np.ix_(rows, remove_cols)] -= float(candidate["weight"])
    if candidate["op"] in ("boost", "both"):
        out[np.ix_(rows, anchor_cols)] += float(candidate["weight"])
    return out


def build_test2_safe_assignment() -> np.ndarray:
    bench = load_module("final_test2_bench", BENCH_SCRIPT)

    image_names = load_lines(ROOT_DIR / "Test2" / "images.txt")
    captions = load_json(ROOT_DIR / "Test2" / "captions.json")

    print("Building Test2 raw 4-model score matrix...")
    model_scores = {
        "siglip2": bench.load_dot_signal(SIGLIP_CACHE / "test2_image_features.npy", SIGLIP_CACHE / "test2_text_features.npy"),
        "qwen3_8b": bench.load_dot_signal(QWEN3_8B_CACHE / "qwen3vl8b_test2_img.npy", QWEN3_8B_CACHE / "qwen3vl8b_test2_text.npy"),
        "clip_h": bench.load_dot_signal(CLIP_CACHE / "test2_image_features.npy", CLIP_CACHE / "test2_text_features.npy"),
        "qwen3_2b": bench.load_dot_signal(QWEN3_2B_CACHE / "test2_image_features.npy", QWEN3_2B_CACHE / "test2_text_features.npy"),
    }
    score = (
        T2_WEIGHTS[0] * model_scores["siglip2"]
        + T2_WEIGHTS[1] * model_scores["qwen3_8b"]
        + T2_WEIGHTS[2] * model_scores["clip_h"]
        + T2_WEIGHTS[3] * model_scores["qwen3_2b"]
    )

    desc_path = QWEN3_DESC_CACHE / "test2_desc.json"
    if desc_path.exists():
        descs = load_json(desc_path)
        cap_entities = [bench.extract_entities(caption) for caption in captions]
        img_entities = [bench.extract_entities(descs.get(name, "")) for name in image_names]
        score = score + JACCARD_BONUS * build_jaccard_from_sets(img_entities, cap_entities)
    else:
        print("  (no Test2 description cache — skipping Jaccard bonus)")
        cap_entities = [bench.extract_entities(caption) for caption in captions]
        img_entities = [set() for _ in image_names]

    selected = []
    selected.extend(load_json(TEST2_CONSERVATIVE_MANIFEST)["best"]["selected"])
    selected.extend(load_json(TEST2_AGGRESSIVE_MANIFEST)["best"]["selected"])
    selected.extend(selected_from_manifest(load_json(TEST2_SECONDPASS_MANIFEST), TEST2_SAFE_PRESET))

    needed_entities = sorted({
        entity
        for candidate in selected
        for entity in (candidate["remove_entity"], candidate["anchor_entity"])
    })
    cap_has = {
        entity: np.array([entity in item for item in cap_entities], dtype=bool)
        for entity in needed_entities
    }
    img_has = {
        entity: np.array([entity in item for item in img_entities], dtype=bool)
        for entity in needed_entities
    }
    for candidate in selected:
        score = apply_test2_candidate(score, candidate, cap_has, img_has)

    return bench.sinkhorn_assign(score)


def build_combined_rows(test1_rows: list[dict[str, str]], test2_assignment: np.ndarray) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row_id, row in enumerate(test1_rows):
        rows.append({
            "row_id": str(row_id),
            "Usage": "Public",
            "class_ids": row["class_ids"],
            "Task": "multi",
        })
    offset = len(test1_rows)
    for image_idx, caption_idx in enumerate(test2_assignment):
        rows.append({
            "row_id": str(offset + image_idx),
            "Usage": "Public",
            "class_ids": str(int(caption_idx)),
            "Task": "single",
        })
    return rows


def build_embedding_base_test1_rows(
    image_names: list[str],
    scores: np.ndarray,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    preds: list[set[int]] = []
    n_seasoning = 0
    n_container = 0
    for image_idx, row in enumerate(scores):
        k = adaptive_gap(row)
        selected = set(np.argsort(-row)[:k].astype(int).tolist())
        for label_idx in SEASONING_INDICES:
            if label_idx not in selected and row[label_idx] > SEASONING_THRESHOLD:
                selected.add(label_idx)
                n_seasoning += 1
        for label_idx in CONTAINER_INDICES:
            if label_idx not in selected and row[label_idx] > CONTAINER_THRESHOLD:
                selected.add(label_idx)
                n_container += 1
        preds.append(selected)
    return rows_from_pred_sets(preds), {
        "layer": "embedding_adaptive_base",
        "seasoning_additions": n_seasoning,
        "container_additions": n_container,
    }


def add_description_ensemble_rows(
    base_rows: list[dict[str, str]],
    image_names: list[str],
    labels: list[str],
    patterns: list[Any],
    mode: str = "bestlocal",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    ensemble = load_module("final_dynamic_qwen_gemma_ensemble", TEST1_QWEN_GEMMA_SCRIPT)
    analysis = load_json(TEST1_QWEN_GEMMA_MANIFEST)
    threshold = analysis["threshold_results"][0]
    if mode != "bestlocal":
        for candidate in analysis["threshold_results"]:
            if mode == "ensemble70" and candidate["allowed_rules"] == ["qwen", "gemma", "and"] and candidate["min_precision"] == 0.70 and candidate["min_tp"] == 1:
                threshold = candidate
                break
            if mode == "and70" and candidate["allowed_rules"] == ["and"] and candidate["min_precision"] == 0.70 and candidate["min_tp"] == 1:
                threshold = candidate
                break

    qwen_descs = load_json(QWEN3_DESC_CACHE / "test1_desc.json")
    gemma_descs = load_json(GEMMA4_DESC_CACHE / "test1_desc.json")
    qwen = ensemble.detect_mentions(image_names, labels, patterns, qwen_descs)
    gemma = ensemble.detect_mentions(image_names, labels, patterns, gemma_descs)
    rule_mentions = {
        "qwen": qwen,
        "gemma": gemma,
        "or": {idx: qwen[idx] | gemma[idx] for idx in range(len(labels))},
        "and": {idx: qwen[idx] & gemma[idx] for idx in range(len(labels))},
    }

    rows = [dict(row) for row in base_rows]
    additions = 0
    for item in threshold["selected"]:
        rule = item["rule"]
        label_idx = int(item["idx"])
        for image_idx in rule_mentions[rule][label_idx]:
            pred = parse_pred_set(rows[image_idx]["class_ids"])
            before = len(pred)
            pred.add(label_idx)
            if len(pred) != before:
                additions += 1
                rows[image_idx]["class_ids"] = format_pred_set(pred)
    return rows, {
        "layer": "qwen_gemma_description_ensemble",
        "mode": mode,
        "selected_rule_labels": len(threshold["selected"]),
        "additions": additions,
        "manifest": str(TEST1_QWEN_GEMMA_MANIFEST.relative_to(ROOT_DIR)),
    }


def build_test1_context() -> dict[str, Any]:
    triggered = load_module("final_dynamic_triggered", TEST1_TRIGGERED_SCRIPT)
    image_names = load_lines(ROOT_DIR / "Test1" / "images.txt")
    captions = load_lines(ROOT_DIR / "Test1" / "captions.txt")
    labels = [triggered.qa.label_from_caption(caption) for caption in captions]
    patterns = [triggered.qa.build_pattern(label) for label in labels]
    scores = build_test1_scores()
    ranks = build_rank_positions(scores)
    return {
        "triggered": triggered,
        "image_names": image_names,
        "captions": captions,
        "labels": labels,
        "patterns": patterns,
        "scores": scores,
        "ranks": ranks,
    }


def build_all_test1_signals(ctx: dict[str, Any]) -> dict[str, dict[int, set[int]]]:
    if has_description_caches():
        selected = load_module("final_dynamic_description_signals", TEST1_SELECTED_SCRIPT)
        signals = selected.build_all_signals(ctx["image_names"], ctx["labels"], ctx["patterns"])
    else:
        print("  (no description caches — skipping description signals)")
        signals = {}
    return signals


def build_topk_images_by_label(scores: np.ndarray, top_k: int) -> dict[int, set[int]]:
    top = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    by_label = {idx: set() for idx in range(scores.shape[1])}
    for image_idx, labels in enumerate(top):
        for label_idx in labels:
            by_label[int(label_idx)].add(int(image_idx))
    return by_label


def apply_triggered_selected_rows(
    base_rows: list[dict[str, str]],
    ctx: dict[str, Any],
    signals: dict[str, dict[int, set[int]]],
    manifest_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    best = manifest["best"]
    topk = build_topk_images_by_label(ctx["scores"], ctx["triggered"].TOP_K)
    rows = [dict(row) for row in base_rows]
    additions = 0
    for item in best["selected"]:
        rule = item["rule"]
        label_idx = int(item["idx"])
        for image_idx in signals.get(rule, {}).get(label_idx, set()) & topk[label_idx]:
            pred = parse_pred_set(rows[image_idx]["class_ids"])
            before = len(pred)
            pred.add(label_idx)
            if len(pred) != before:
                additions += 1
                rows[image_idx]["class_ids"] = format_pred_set(pred)
    return rows, {
        "layer": "triggered_selected",
        "manifest": str(manifest_path.relative_to(ROOT_DIR)),
        "base": best["base"],
        "selected_rule_labels": len(best["selected"]),
        "additions": additions,
    }


def apply_margin_gates_rows(
    base_rows: list[dict[str, str]],
    ctx: dict[str, Any],
    signals: dict[str, dict[int, set[int]]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    margin = load_module("final_dynamic_margin_gates", TEST1_MARGIN_SCRIPT)
    best = load_json(TEST1_MARGIN_MANIFEST)["best"]
    selected_gate_names = {item["gate"] for item in best["selected"]}
    gate_config_list = [config for config in margin.gate_configs() if config["gate"] in selected_gate_names]
    triggered_mentions = margin.build_triggered_mentions(signals, ctx["scores"], gate_config_list)
    rows, additions = margin.apply_result_to_submission(best, base_rows, triggered_mentions)
    return rows[: len(ctx["image_names"])], {
        "layer": "margin_gates",
        "manifest": str(TEST1_MARGIN_MANIFEST.relative_to(ROOT_DIR)),
        "base": best["base"],
        "selected_rule_labels": len(best["selected"]),
        "selected_gate_count": len(selected_gate_names),
        "additions": additions,
    }


def build_signal_sets_for_clustered(signals: dict[str, dict[int, set[int]]], n_labels: int) -> dict[str, dict[int, set[int]]]:
    if not signals:
        empty = {idx: set() for idx in range(n_labels)}
        return {"desc_or": empty, "desc_and": empty, "vlm_or": empty}
    return {
        "desc_or": {
            idx: signals["desc_qwen"][idx] | signals["desc_gemma"][idx]
            for idx in range(n_labels)
        },
        "desc_and": signals["desc_and"],
        "vlm_or": {
            idx: (
                signals["desc_qwen"][idx]
                | signals["desc_gemma"][idx]
            )
            for idx in range(n_labels)
        },
    }


def reconstruct_clustered_images(
    item: dict[str, Any],
    base_preds: list[set[int]],
    scores: np.ndarray,
    ranks: np.ndarray,
    signal_sets: dict[str, dict[int, set[int]]],
) -> set[int]:
    if item["op"] == "remove":
        remove_idx = int(item["remove_idx"])
        anchor_idx = int(item["anchor_idx"])
        min_score_delta = float(item["min_score_delta"])
        mode = item["mode"]
        images: set[int] = set()
        for image_idx, pred_set in enumerate(base_preds):
            if remove_idx not in pred_set or anchor_idx not in pred_set:
                continue
            if scores[image_idx, anchor_idx] - scores[image_idx, remove_idx] < min_score_delta:
                continue
            if mode == "desc_anchor" and image_idx not in signal_sets["desc_or"][anchor_idx]:
                continue
            if mode == "desc_anchor_no_remove" and (
                image_idx not in signal_sets["desc_or"][anchor_idx]
                or image_idx in signal_sets["desc_or"][remove_idx]
            ):
                continue
            if mode == "and_anchor_no_remove" and (
                image_idx not in signal_sets["desc_and"][anchor_idx]
                or image_idx in signal_sets["desc_or"][remove_idx]
            ):
                continue
            images.add(image_idx)
        return images

    from_idx = int(item["from_idx"])
    to_idx = int(item["to_idx"])
    evidence = item["evidence"]
    min_score_delta = float(item["min_score_delta"])
    max_rank = int(item["max_rank"])
    images = set()
    for image_idx in signal_sets[evidence][to_idx]:
        if from_idx not in base_preds[image_idx] or to_idx in base_preds[image_idx]:
            continue
        if ranks[image_idx, to_idx] > max_rank:
            continue
        score_delta = scores[image_idx, to_idx] - scores[image_idx, from_idx]
        if score_delta < min_score_delta:
            continue
        if image_idx in signal_sets[evidence][from_idx] and score_delta < 0.25:
            continue
        images.add(image_idx)
    return images


def apply_clustered_manifest_rows(
    base_rows: list[dict[str, str]],
    ctx: dict[str, Any],
    signal_sets: dict[str, dict[int, set[int]]],
    manifest_path: Path,
    preset_name: str = "best",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    selected = selected_from_manifest(manifest, preset_name)
    base_preds = preds_from_rows(base_rows, len(ctx["image_names"]))
    preds = [set(item) for item in base_preds]
    additions = 0
    removals = 0
    for item in selected:
        images = reconstruct_clustered_images(item, base_preds, ctx["scores"], ctx["ranks"], signal_sets)
        if item["op"] == "remove":
            remove_idx = int(item["remove_idx"])
            anchor_idx = int(item["anchor_idx"])
            for image_idx in images:
                if remove_idx in preds[image_idx] and anchor_idx in preds[image_idx]:
                    preds[image_idx].discard(remove_idx)
                    removals += 1
        else:
            from_idx = int(item["from_idx"])
            to_idx = int(item["to_idx"])
            for image_idx in images:
                if from_idx in preds[image_idx] and to_idx not in preds[image_idx]:
                    preds[image_idx].add(to_idx)
                    additions += 1
    rows = rows_from_pred_sets(preds)
    return rows, {
        "layer": f"clustered_{manifest_path.stem}",
        "manifest": str(manifest_path.relative_to(ROOT_DIR)),
        "preset": preset_name,
        "selected_rules": len(selected),
        "additions": additions,
        "removals": removals,
    }


def recall_candidate_key(item: dict[str, Any], mode: str) -> tuple[Any, ...]:
    label_idx = int(item["label_idx"])
    if mode == "source_label":
        return item["source"], label_idx
    if mode == "source_label_rank":
        rank_bin = 1 if item["rank"] <= 1 else 3 if item["rank"] <= 3 else 5 if item["rank"] <= 5 else 10 if item["rank"] <= 10 else 20
        return item["source"], label_idx, rank_bin
    if mode == "source_label_evidence":
        evidence = "desc_and" if item["desc_and"] else "vlm_or" if item["vlm_or"] else "none"
        rank_bin = 3 if item["rank"] <= 3 else 5 if item["rank"] <= 5 else 10 if item["rank"] <= 10 else 20
        return item["source"], label_idx, evidence, rank_bin
    raise ValueError(mode)


def build_recall_additions_from_source(
    current_preds: list[set[int]],
    source_preds: list[set[int]],
    source_name: str,
    ctx: dict[str, Any],
    signal_sets: dict[str, dict[int, set[int]]],
) -> list[dict[str, Any]]:
    out = []
    for image_idx in range(len(ctx["image_names"])):
        for label_idx in source_preds[image_idx] - current_preds[image_idx]:
            out.append({
                "image_idx": image_idx,
                "label_idx": int(label_idx),
                "source": source_name,
                "rank": int(ctx["ranks"][image_idx, label_idx]),
                "score": float(ctx["scores"][image_idx, label_idx]),
                "desc_or": image_idx in signal_sets["desc_or"][label_idx],
                "desc_and": image_idx in signal_sets["desc_and"][label_idx],
                "vlm_or": image_idx in signal_sets["vlm_or"][label_idx],
            })
    return out


def build_recall_vlm_additions(
    current_preds: list[set[int]],
    ctx: dict[str, Any],
    signal_sets: dict[str, dict[int, set[int]]],
) -> list[dict[str, Any]]:
    out = []
    for label_idx in range(len(ctx["labels"])):
        for image_idx in signal_sets["desc_and"][label_idx]:
            if label_idx in current_preds[image_idx]:
                continue
            rank = int(ctx["ranks"][image_idx, label_idx])
            if rank > 12:
                continue
            out.append({
                "image_idx": image_idx,
                "label_idx": int(label_idx),
                "source": "desc_and_rank12",
                "rank": rank,
                "score": float(ctx["scores"][image_idx, label_idx]),
                "desc_or": True,
                "desc_and": True,
                "vlm_or": True,
            })
    return out


def apply_dynamic_recall_rows(
    current_rows: list[dict[str, str]],
    source_rows: dict[str, list[dict[str, str]]],
    ctx: dict[str, Any],
    signal_sets: dict[str, dict[int, set[int]]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    final_result = None
    for result in load_json(FINAL_RECALL_MANIFEST)["results"]:
        if result["preset"]["name"] == FINAL_PRESET:
            final_result = result
            break
    if final_result is None:
        raise RuntimeError(f"Could not find {FINAL_PRESET} in {FINAL_RECALL_MANIFEST}")

    current_preds = preds_from_rows(current_rows, len(ctx["image_names"]))
    additions: list[dict[str, Any]] = []
    for source_name, rows in source_rows.items():
        additions.extend(build_recall_additions_from_source(
            current_preds,
            preds_from_rows(rows, len(ctx["image_names"])),
            source_name,
            ctx,
            signal_sets,
        ))
    additions.extend(build_recall_vlm_additions(current_preds, ctx, signal_sets))

    by_mode_key: dict[tuple[str, tuple[Any, ...]], set[tuple[int, int]]] = {}
    for item in additions:
        for mode in ["source_label", "source_label_rank", "source_label_evidence"]:
            by_mode_key.setdefault((mode, recall_candidate_key(item, mode)), set()).add(
                (int(item["image_idx"]), int(item["label_idx"]))
            )

    preds = [set(item) for item in current_preds]
    selected_pairs: set[tuple[int, int]] = set()
    additions_total = 0
    for rule in final_result["selected"]:
        mode = rule["mode"]
        key = tuple(rule["key"])
        pairs = sorted(by_mode_key.get((mode, key), set()))
        for image_idx, label_idx in pairs:
            if (image_idx, label_idx) in selected_pairs or label_idx in preds[image_idx]:
                continue
            preds[image_idx].add(label_idx)
            selected_pairs.add((image_idx, label_idx))
            additions_total += 1
    return rows_from_pred_sets(preds), {
        "layer": "dynamic_recall_addbacks",
        "manifest": str(FINAL_RECALL_MANIFEST.relative_to(ROOT_DIR)),
        "preset": FINAL_PRESET,
        "selected_rules": len(final_result["selected"]),
        "additions": additions_total,
    }


def build_dynamic_private_safe_test1_rows() -> tuple[list[dict[str, str]], dict[str, Any]]:
    ctx = build_test1_context()
    layers: list[dict[str, Any]] = []

    base_rows, summary = build_embedding_base_test1_rows(ctx["image_names"], ctx["scores"])
    layers.append(summary)

    if has_description_caches():
        qwen_gemma_rows, summary = add_description_ensemble_rows(base_rows, ctx["image_names"], ctx["labels"], ctx["patterns"])
        layers.append(summary)
    else:
        print("  (no description caches — skipping description ensemble layer)")
        qwen_gemma_rows = base_rows

    signals = build_all_test1_signals(ctx)
    selected_rows, summary = apply_triggered_selected_rows(qwen_gemma_rows, ctx, signals, TEST1_SELECTED_MANIFEST)
    layers.append(summary)
    margin_rows, summary = apply_margin_gates_rows(selected_rows, ctx, signals)
    layers.append(summary)

    signal_sets = build_signal_sets_for_clustered(signals, len(ctx["labels"]))
    clustered_rows, summary = apply_clustered_manifest_rows(margin_rows, ctx, signal_sets, TEST1_CLUSTERED_MANIFEST, "best")
    layers.append(summary)
    aggressive_rows, summary = apply_clustered_manifest_rows(clustered_rows, ctx, signal_sets, TEST1_AGGRESSIVE_MANIFEST, "best")
    layers.append(summary)
    secondpass_rows, summary = apply_clustered_manifest_rows(aggressive_rows, ctx, signal_sets, TEST1_SECONDPASS_MANIFEST, "best")
    layers.append(summary)

    recall_rows, summary = apply_dynamic_recall_rows(
        secondpass_rows,
        {
            "public_ab": clustered_rows,
            "aggressive_t1": aggressive_rows,
            "margin_t1": margin_rows,
        },
        ctx,
        signal_sets,
    )
    layers.append(summary)
    return recall_rows, {
        "mode": "dynamic_private_safe",
        "layers": layers,
    }


def compare_to_reference(path: Path) -> dict[str, Any]:
    if not FINAL_REFERENCE.exists():
        return {"reference_exists": False}
    same = path.read_bytes() == FINAL_REFERENCE.read_bytes()
    result = {
        "reference_exists": True,
        "reference": str(FINAL_REFERENCE.relative_to(ROOT_DIR)),
        "exact_match": same,
        "reference_sha256": sha256_file(FINAL_REFERENCE),
    }
    if same:
        return result

    with path.open() as a, FINAL_REFERENCE.open() as b:
        rows_a = list(csv.DictReader(a))
        rows_b = list(csv.DictReader(b))
    changed = [
        idx for idx, (ra, rb) in enumerate(zip(rows_a, rows_b))
        if ra["class_ids"] != rb["class_ids"] or ra["Task"] != rb["Task"]
    ]
    result["changed_rows"] = len(changed)
    result["first_changed_rows"] = changed[:20]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "submission.csv",
        help="Output submission CSV path.",
    )
    parser.add_argument("--skip-description-cache-build", action="store_true")
    parser.add_argument("--skip-cache-validation", action="store_true")
    parser.add_argument("--description-save-every", type=int, default=50)
    parser.add_argument("--manifest-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cache_summary = {} if args.skip_cache_validation else validate_embedding_caches()
    if not args.skip_description_cache_build:
        if has_description_caches():
            print("Description cache files found — will check for missing entries.")
        desc_summary = ensure_description_caches(
            generate=True,
            save_every=args.description_save_every,
        )
    elif has_description_caches():
        print("Description caches found (skip-build mode).")
    else:
        print("Description caches not found — running with embedding-only mode.")
    test1_rows, test1_summary = build_dynamic_private_safe_test1_rows()
    recall_summary = test1_summary["layers"][-1]
    test2_assignment = build_test2_safe_assignment()
    rows = build_combined_rows(test1_rows, test2_assignment)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.output, rows)
    compare = compare_to_reference(args.output)
    out_sha = sha256_file(args.output)

    manifest = {
        "script": str(Path(__file__).relative_to(ROOT_DIR)),
        "output": str(args.output.relative_to(ROOT_DIR) if args.output.is_relative_to(ROOT_DIR) else args.output),
        "output_sha256": out_sha,
        "row_count": len(rows),
        "final_preset": FINAL_PRESET,
        "embedding_cache_summary": cache_summary,
        "description_caches_available": has_description_caches(),
        "test1_summary": test1_summary,
        "recall_summary": recall_summary,
        "test2_safe_preset": TEST2_SAFE_PRESET,
        "reference_compare": compare,
        "elapsed_sec": time.time() - t_start,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_out = args.manifest_out or RESULT_DIR / f"final_reproduction_{time.strftime('%Y%m%d%H%M%S')}.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"Saved submission: {args.output}")
    print(f"SHA256: {out_sha}")
    if compare.get("reference_exists"):
        print(f"Reference exact match: {compare['exact_match']}")
        if not compare["exact_match"]:
            print(f"Changed rows vs reference: {compare.get('changed_rows')}")
    print(f"Saved manifest: {manifest_out}")
    print(f"Time: {manifest['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
