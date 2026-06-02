#!/usr/bin/env python3
"""
Benchmark Test2 matching signals and persist every attempt.

This script is intentionally standalone so each experiment can be reproduced.
It writes:
  - optimization/experiment_results/<run_id>.json
  - optimization/experiment_results/<run_id>.csv
  - optimization/test2_experiment_log.jsonl (append-only, one row per attempt)

Gemini judge vectors can be included only as a diagnostic upper-bound signal with
--include-judge; do not use judge-vector variants for Kaggle submissions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parent.parent
OPT_DIR = ROOT_DIR / "optimization"
RESULT_DIR = OPT_DIR / "experiment_results"
LOG_PATH = OPT_DIR / "test2_experiment_log.jsonl"

SIGLIP_CACHE = ROOT_DIR / "siglip2_cache"
QWEN3_8B_CACHE = ROOT_DIR / "qwen3_cache"
CLIP_CACHE = ROOT_DIR / "clip_cache" / "vit_h_14_dfn5b"
QWEN3_2B_CACHE = ROOT_DIR / "qwen3vl2b_cache"
QWEN3_2B_FT_CACHE = ROOT_DIR / "qwen3vl2b_ft_web_cache"
GME_CACHE = ROOT_DIR / "gme_cache"
METACLIP2_CACHE = ROOT_DIR / "metaclip2_cache"
DE_CACHE = ROOT_DIR / "qwen3_de_cache"
JUDGE_CACHE = ROOT_DIR / "gemini_judge_cache"

T2_WEIGHTS = [0.20, 0.40, 0.15, 0.25]
JACCARD_BONUS = 0.20
SINKHORN_TAU = 0.15
SINKHORN_ITER = 50

ENTITIES = {
    "chicken": ["chicken", "poultry"],
    "beef": ["beef", "steak", "brisket"],
    "pork": ["pork", "ham", "bacon", "prosciutto"],
    "lamb": ["lamb"],
    "duck": ["duck"],
    "turkey": ["turkey"],
    "fish": ["fish", "salmon", "tuna", "cod", "tilapia"],
    "shrimp": ["shrimp", "prawn", "prawns"],
    "crab": ["crab"],
    "lobster": ["lobster"],
    "meatball": ["meatball"],
    "sausage": ["sausage"],
    "patty": ["patty", "patties"],
    "tofu": ["tofu"],
    "rice": ["rice", "risotto"],
    "pasta": ["pasta", "spaghetti", "fettuccine", "penne", "linguine", "macaroni", "ravioli"],
    "noodles": ["noodle", "noodles", "ramen", "udon"],
    "bread": ["bread", "baguette"],
    "toast": ["toast"],
    "bun": ["bun", "buns"],
    "tortilla": ["tortilla"],
    "potato": ["potato", "potatoes", "fries"],
    "broccoli": ["broccoli"],
    "carrots": ["carrot", "carrots"],
    "green_beans": ["green beans"],
    "peas": ["peas"],
    "corn": ["corn"],
    "spinach": ["spinach"],
    "lettuce": ["lettuce"],
    "tomato": ["tomato", "tomatoes", "cherry tomato"],
    "onion": ["onion", "onions"],
    "mushroom": ["mushroom", "mushrooms"],
    "pepper": ["pepper", "peppers"],
    "zucchini": ["zucchini"],
    "asparagus": ["asparagus"],
    "cabbage": ["cabbage"],
    "cucumber": ["cucumber"],
    "celery": ["celery"],
    "avocado": ["avocado"],
    "apple": ["apple", "apples"],
    "banana": ["banana", "bananas"],
    "orange": ["orange", "oranges"],
    "lemon": ["lemon"],
    "berries": ["berry", "berries", "blueberries", "strawberries", "raspberries"],
    "grapes": ["grape", "grapes"],
    "mango": ["mango"],
    "pineapple": ["pineapple"],
    "cheese": ["cheese", "mozzarella", "cheddar", "parmesan", "feta"],
    "cream_sauce": ["cream sauce", "creamy sauce"],
    "tomato_sauce": ["tomato sauce", "marinara"],
    "gravy": ["gravy"],
    "sushi": ["sushi", "nigiri", "sashimi"],
    "pizza": ["pizza"],
    "burger": ["burger", "hamburger"],
    "sandwich": ["sandwich"],
    "taco": ["taco", "tacos"],
    "soup": ["soup", "stew", "broth", "chowder"],
    "salad": ["salad"],
    "cake": ["cake"],
    "ice_cream": ["ice cream", "gelato"],
    "eggs": ["egg", "eggs", "omelette"],
    "fried": ["fried", "crispy"],
    "grilled": ["grilled"],
    "roasted": ["roasted"],
    "steamed": ["steamed"],
    "baked": ["baked"],
}

ENTITY_PATTERNS = {
    ent: re.compile("|".join(rf"\b{re.escape(form)}\b" for form in forms), re.IGNORECASE)
    for ent, forms in ENTITIES.items()
}


def z_norm(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    mu = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True) + 1e-12
    return (matrix - mu) / std


def sinkhorn_assign(score: np.ndarray, tau: float = SINKHORN_TAU, n_iter: int = SINKHORN_ITER) -> np.ndarray:
    log_k = score.astype(np.float32, copy=False) / tau
    log_k = log_k - log_k.max()
    k_matrix = np.exp(log_k).astype(np.float32)
    u = np.ones(k_matrix.shape[0], dtype=np.float32)
    v = np.ones(k_matrix.shape[1], dtype=np.float32)
    for _ in range(n_iter):
        u = 1.0 / (k_matrix @ v + 1e-12)
        v = 1.0 / (k_matrix.T @ u + 1e-12)
    p_matrix = (u[:, None] * k_matrix) * v[None, :]

    assignment = np.full(k_matrix.shape[0], -1, dtype=np.int32)
    used_caps: set[int] = set()
    for img_idx in np.argsort(-p_matrix.max(axis=1)):
        row = p_matrix[img_idx].copy()
        if used_caps:
            row[list(used_caps)] = -1
        cap_idx = int(np.argmax(row))
        assignment[int(img_idx)] = cap_idx
        used_caps.add(cap_idx)
    return assignment


def extract_entities(text: str) -> set[str]:
    return {entity for entity, pattern in ENTITY_PATTERNS.items() if pattern.search(text)}


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def load_dot_signal(img_path: Path, text_path: Path) -> np.ndarray:
    image_features = np.load(img_path)
    text_features = np.load(text_path)
    return z_norm(image_features @ text_features.T)


def load_desc_vector_signal(cache_dir: Path, image_names: list[str]) -> np.ndarray:
    cap_vecs = np.load(cache_dir / "test2_cap_vecs.npy")
    desc_raw = np.load(cache_dir / "test2_desc_vecs.npy", allow_pickle=True)
    if desc_raw.dtype == object and desc_raw.shape == ():
        desc_dict = desc_raw.item()
        desc_mat = np.zeros((len(image_names), cap_vecs.shape[1]), dtype=np.float32)
        for img_idx, image_name in enumerate(image_names):
            if image_name in desc_dict:
                desc_mat[img_idx] = desc_dict[image_name]
    elif desc_raw.ndim == 2:
        desc_mat = desc_raw.astype(np.float32, copy=False)
        if desc_mat.shape[0] != len(image_names):
            raise ValueError(f"{cache_dir}/test2_desc_vecs.npy row count does not match Test2 images")
    else:
        raise ValueError(f"Unsupported desc vector format in {cache_dir}/test2_desc_vecs.npy")
    return z_norm(desc_mat @ cap_vecs.T)


def build_entity_matrices(
    captions: list[str],
    image_descs: dict[str, str],
    image_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[set[str]], list[set[str]], dict[str, float]]:
    cap_entities = [extract_entities(caption) for caption in captions]
    img_entities = [extract_entities(image_descs.get(image_name, "")) for image_name in image_names]

    n_caps = len(captions)
    doc_freq = {
        entity: sum(1 for entities in cap_entities if entity in entities)
        for entity in ENTITIES
    }
    idf = {
        entity: float(np.log((n_caps + 1) / (doc_freq[entity] + 1)) + 1.0)
        for entity in ENTITIES
    }

    jaccard = np.zeros((len(image_names), len(captions)), dtype=np.float32)
    weighted = np.zeros_like(jaccard)
    for img_idx, img_set in enumerate(img_entities):
        if not img_set:
            continue
        for cap_idx, cap_set in enumerate(cap_entities):
            if not cap_set:
                continue
            inter = img_set & cap_set
            union = img_set | cap_set
            if union:
                jaccard[img_idx, cap_idx] = len(inter) / len(union)
            if inter:
                weighted[img_idx, cap_idx] = sum(idf[e] for e in inter) / (sum(idf[e] for e in union) + 1e-9)
    return jaccard, weighted, cap_entities, img_entities, idf


def build_tfidf_signal(captions: list[str], image_texts: list[str], analyzer: str) -> np.ndarray:
    if analyzer == "word":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
    elif analyzer == "char_wb":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
    else:
        raise ValueError(f"Unsupported analyzer: {analyzer}")

    combined_texts = image_texts + captions
    features = vectorizer.fit_transform(combined_texts)
    image_features = features[: len(image_texts)]
    caption_features = features[len(image_texts) :]
    return z_norm(cosine_similarity(image_features, caption_features).astype(np.float32))


def accuracy(assignment: np.ndarray, gold: np.ndarray) -> tuple[float, int]:
    correct = int(np.sum(assignment == gold))
    return correct / len(gold), correct


def compare_assignments(assignment: np.ndarray, baseline: np.ndarray, gold: np.ndarray) -> dict[str, int]:
    changed = assignment != baseline
    return {
        "changed": int(np.sum(changed)),
        "helpful": int(np.sum(changed & (assignment == gold) & (baseline != gold))),
        "harmful": int(np.sum(changed & (assignment != gold) & (baseline == gold))),
        "neutral_wrong": int(np.sum(changed & (assignment != gold) & (baseline != gold))),
        "neutral_correct": int(np.sum(changed & (assignment == gold) & (baseline == gold))),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_outputs(run_json: Path, run_csv: Path, manifest: dict[str, Any]) -> None:
    with run_json.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    rows = manifest["results"]
    fieldnames = [
        "attempt",
        "accuracy",
        "correct",
        "changed",
        "helpful",
        "harmful",
        "neutral_wrong",
        "neutral_correct",
        "leaky",
        "notes",
        "params_json",
    ]
    with run_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "attempt": row["attempt"],
                "accuracy": row["accuracy"],
                "correct": row["correct"],
                "leaky": row["leaky"],
                "notes": row["notes"],
                "params_json": json.dumps(row["params"], sort_keys=True),
            }
            flat.update(row["delta_vs_baseline"])
            writer.writerow(flat)


def add_result(
    manifest: dict[str, Any],
    run_json: Path,
    run_csv: Path,
    attempt: str,
    score: np.ndarray,
    gold: np.ndarray,
    baseline_assignment: np.ndarray | None,
    params: dict[str, Any],
    notes: str = "",
    leaky: bool = False,
) -> np.ndarray:
    assignment = sinkhorn_assign(score)
    acc, correct = accuracy(assignment, gold)
    if baseline_assignment is None:
        delta = {
            "changed": 0,
            "helpful": 0,
            "harmful": 0,
            "neutral_wrong": 0,
            "neutral_correct": 0,
        }
    else:
        delta = compare_assignments(assignment, baseline_assignment, gold)

    row = {
        "run_id": manifest["run_id"],
        "script": manifest["script"],
        "attempt": attempt,
        "accuracy": acc,
        "correct": correct,
        "scored": int(len(gold)),
        "delta_vs_baseline": delta,
        "params": params,
        "leaky": leaky,
        "notes": notes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest["results"].append(row)
    manifest["best_nonleaky"] = max(
        (r for r in manifest["results"] if not r["leaky"]),
        key=lambda item: item["accuracy"],
    )
    manifest["best_overall"] = max(manifest["results"], key=lambda item: item["accuracy"])
    write_outputs(run_json, run_csv, manifest)
    append_jsonl(LOG_PATH, row)
    print(
        f"{attempt:45s} acc={acc:.4f} correct={correct} "
        f"changed={delta['changed']} helpful={delta['helpful']} harmful={delta['harmful']}"
        f"{' LEAKY' if leaky else ''}",
        flush=True,
    )
    return assignment


def oracle_rank_summary(score: np.ndarray, gold: np.ndarray, top_ks: list[int]) -> dict[str, float]:
    max_k = max(top_ks)
    top = np.argpartition(-score, kth=max_k - 1, axis=1)[:, :max_k]
    summary = {}
    for top_k in top_ks:
        hits = np.array([gold[i] in set(top[i, :top_k].tolist()) for i in range(len(gold))])
        summary[f"gold_in_top_{top_k}"] = float(np.mean(hits))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-judge", action="store_true", help="Include Gemini judge vectors as leaky diagnostics.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller sweep.")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"test2_matching_signals_{time.strftime('%Y%m%d%H%M%S')}"
    run_json = RESULT_DIR / f"{run_id}.json"
    run_csv = RESULT_DIR / f"{run_id}.csv"

    with (ROOT_DIR / "Test2" / "images.txt").open() as handle:
        image_names = [line.strip() for line in handle if line.strip()]
    captions = load_json(ROOT_DIR / "Test2" / "captions.json")
    qwen_descs = load_json(DE_CACHE / "test2_desc.json")
    gold_map = load_json(JUDGE_CACHE / "gold_test2_hungarian.json")
    gold = np.array([gold_map.get(image_name, -1) for image_name in image_names], dtype=np.int32)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "script": str(Path(__file__).relative_to(ROOT_DIR)),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "t2_weights": T2_WEIGHTS,
            "jaccard_bonus": JACCARD_BONUS,
            "sinkhorn_tau": SINKHORN_TAU,
            "sinkhorn_iter": SINKHORN_ITER,
            "include_judge": args.include_judge,
            "quick": args.quick,
        },
        "data": {
            "n_images": len(image_names),
            "n_captions": len(captions),
        },
        "results": [],
        "best_nonleaky": None,
        "best_overall": None,
    }
    write_outputs(run_json, run_csv, manifest)
    print(f"Saving results to {run_json} and {run_csv}", flush=True)

    signals: dict[str, np.ndarray] = {}

    print("Loading base embedding signals...", flush=True)
    signals["siglip2"] = load_dot_signal(SIGLIP_CACHE / "test2_image_features.npy", SIGLIP_CACHE / "test2_text_features.npy")
    signals["qwen3_8b"] = load_dot_signal(QWEN3_8B_CACHE / "qwen3vl8b_test2_img.npy", QWEN3_8B_CACHE / "qwen3vl8b_test2_text.npy")
    signals["clip_dfn5b"] = load_dot_signal(CLIP_CACHE / "test2_image_features.npy", CLIP_CACHE / "test2_text_features.npy")
    signals["qwen2b"] = load_dot_signal(QWEN3_2B_CACHE / "test2_image_features.npy", QWEN3_2B_CACHE / "test2_text_features.npy")

    print("Building entity signals...", flush=True)
    jaccard_raw, weighted_entity_raw, _, _, idf = build_entity_matrices(captions, qwen_descs, image_names)
    manifest["entity_idf_top20"] = sorted(idf.items(), key=lambda item: item[1], reverse=True)[:20]
    signals["weighted_entity"] = z_norm(weighted_entity_raw)

    baseline_score = (
        T2_WEIGHTS[0] * signals["siglip2"]
        + T2_WEIGHTS[1] * signals["qwen3_8b"]
        + T2_WEIGHTS[2] * signals["clip_dfn5b"]
        + T2_WEIGHTS[3] * signals["qwen2b"]
        + JACCARD_BONUS * jaccard_raw
    )
    manifest["oracle_baseline"] = oracle_rank_summary(baseline_score, gold, [1, 3, 5, 10, 20, 50])
    baseline_assignment = add_result(
        manifest,
        run_json,
        run_csv,
        "baseline_current",
        baseline_score,
        gold,
        None,
        {"score": "4-model ensemble + raw entity Jaccard"},
    )

    print("Loading additional nonleaky signals...", flush=True)
    optional_dot_signals = {
        "qwen2b_ft": (QWEN3_2B_FT_CACHE / "test2_image_features.npy", QWEN3_2B_FT_CACHE / "test2_text_features.npy"),
        "gme": (GME_CACHE / "test2_image_features.npy", GME_CACHE / "test2_text_features.npy"),
        "metaclip2": (METACLIP2_CACHE / "test2_image_features.npy", METACLIP2_CACHE / "test2_text_features.npy"),
    }
    for name, (img_path, text_path) in optional_dot_signals.items():
        if img_path.exists() and text_path.exists():
            signals[name] = load_dot_signal(img_path, text_path)
            add_result(
                manifest,
                run_json,
                run_csv,
                f"single_{name}",
                signals[name],
                gold,
                baseline_assignment,
                {"score": name},
            )

    signals["qwen3_de_vec"] = load_desc_vector_signal(DE_CACHE, image_names)
    add_result(
        manifest,
        run_json,
        run_csv,
        "single_qwen3_de_vec",
        signals["qwen3_de_vec"],
        gold,
        baseline_assignment,
        {"score": "qwen3 description vectors vs captions"},
    )

    print("Building TF-IDF signals...", flush=True)
    image_texts = [qwen_descs.get(image_name, "") for image_name in image_names]
    signals["tfidf_qwen_word"] = build_tfidf_signal(captions, image_texts, "word")
    signals["tfidf_qwen_char_wb"] = build_tfidf_signal(captions, image_texts, "char_wb")
    for name in ["weighted_entity", "tfidf_qwen_word", "tfidf_qwen_char_wb"]:
        add_result(
            manifest,
            run_json,
            run_csv,
            f"single_{name}",
            signals[name],
            gold,
            baseline_assignment,
            {"score": name},
        )

    if args.include_judge:
        print("Loading leaky Gemini-judge diagnostics...", flush=True)
        signals["gemini_vec"] = load_desc_vector_signal(JUDGE_CACHE, image_names)
        add_result(
            manifest,
            run_json,
            run_csv,
            "diagnostic_single_gemini_vec",
            signals["gemini_vec"],
            gold,
            baseline_assignment,
            {"score": "Gemini judge vectors vs captions"},
            notes="Leaky diagnostic only; do not submit.",
            leaky=True,
        )
        judge_descs = load_json(JUDGE_CACHE / "test2_desc.json")
        judge_texts = [judge_descs.get(image_name, "") for image_name in image_names]
        signals["tfidf_gemini_word"] = build_tfidf_signal(captions, judge_texts, "word")
        add_result(
            manifest,
            run_json,
            run_csv,
            "diagnostic_single_tfidf_gemini_word",
            signals["tfidf_gemini_word"],
            gold,
            baseline_assignment,
            {"score": "TF-IDF over Gemini judge descriptions"},
            notes="Leaky diagnostic only; do not submit.",
            leaky=True,
        )

    single_weights = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
    if not args.quick:
        single_weights += [0.80, 1.00]

    candidate_signals = [
        "qwen3_de_vec",
        "tfidf_qwen_word",
        "tfidf_qwen_char_wb",
        "weighted_entity",
        "qwen2b_ft",
        "gme",
        "metaclip2",
    ]
    candidate_signals = [name for name in candidate_signals if name in signals]

    print("Sweeping single additions to the baseline...", flush=True)
    for signal_name in candidate_signals:
        for weight in single_weights:
            add_result(
                manifest,
                run_json,
                run_csv,
                f"baseline_plus_{weight:g}_{signal_name}",
                baseline_score + weight * signals[signal_name],
                gold,
                baseline_assignment,
                {"base": "baseline_current", "add": signal_name, "weight": weight},
            )

    pair_weights = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    if args.quick:
        pair_weights = [0.05, 0.10, 0.20]
    print("Sweeping two-signal additions to the baseline...", flush=True)
    pair_candidates = [
        name
        for name in ["qwen3_de_vec", "tfidf_qwen_word", "tfidf_qwen_char_wb", "weighted_entity", "gme", "metaclip2"]
        if name in signals
    ]
    for idx, signal_a in enumerate(pair_candidates):
        for signal_b in pair_candidates[idx + 1 :]:
            for weight_a in pair_weights:
                for weight_b in pair_weights:
                    add_result(
                        manifest,
                        run_json,
                        run_csv,
                        f"baseline_plus_{weight_a:g}_{signal_a}_plus_{weight_b:g}_{signal_b}",
                        baseline_score + weight_a * signals[signal_a] + weight_b * signals[signal_b],
                        gold,
                        baseline_assignment,
                        {
                            "base": "baseline_current",
                            "add_a": signal_a,
                            "weight_a": weight_a,
                            "add_b": signal_b,
                            "weight_b": weight_b,
                        },
                    )

    if args.include_judge:
        for weight in [0.03, 0.05, 0.10, 0.20]:
            add_result(
                manifest,
                run_json,
                run_csv,
                f"diagnostic_baseline_plus_{weight:g}_gemini_vec",
                baseline_score + weight * signals["gemini_vec"],
                gold,
                baseline_assignment,
                {"base": "baseline_current", "add": "gemini_vec", "weight": weight},
                notes="Leaky diagnostic only; do not submit.",
                leaky=True,
            )

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_outputs(run_json, run_csv, manifest)
    print("Best nonleaky:", manifest["best_nonleaky"], flush=True)
    print(f"Finished. Results saved to {run_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": str(Path(__file__).relative_to(ROOT_DIR)),
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        append_jsonl(LOG_PATH, failure)
        print(traceback.format_exc(), file=sys.stderr)
        raise
