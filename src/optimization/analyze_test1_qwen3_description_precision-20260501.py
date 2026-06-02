#!/usr/bin/env python3
"""
Analyze whether Qwen3-8B Test1 descriptions can extend the 30 disambiguation targets.

For each Test1 ingredient label:
  - detect exact/near-exact ingredient mentions in qwen3_de_cache/test1_desc.json
  - compare those mentions against local Gemini-derived Test1 gold
  - measure the precision of ADDING a label only when the current all30
    submission is missing it
  - simulate thresholded extension sets

Outputs:
  optimization/experiment_results/test1_qwen3_desc_precision_<timestamp>.json
  optimization/experiment_results/test1_qwen3_desc_precision_<timestamp>.csv
  optimization/test2_experiment_log.jsonl append row
"""

from __future__ import annotations

import csv
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
OPT_DIR = ROOT_DIR / "optimization"
RESULT_DIR = OPT_DIR / "experiment_results"
LOG_PATH = OPT_DIR / "test2_experiment_log.jsonl"

ALL30_SUBMISSION = OPT_DIR / "submission-entity-jaccard-20260430200009.csv"
QWEN_DESC_PATH = ROOT_DIR / "qwen3_de_cache" / "test1_desc.json"
JUDGE_CACHE = ROOT_DIR / "gemini_judge_cache"

PROBE_TARGETS = {
    "beef": 20, "breading": 41, "broth": 44, "cake layers": 50,
    "cherry tomatoes": 60, "crawfish": 97, "cream sauce": 100,
    "eggs": 140, "garlic": 168, "green grapes": 178, "lemon": 209,
    "noodles": 239, "orange": 246, "orange juice": 247, "pasta": 254,
    "peaches": 256, "peanut sauce": 257, "peas": 260, "rice": 308,
    "sesame seeds": 329, "shrimp": 333, "soft serve ice cream": 340,
    "spicy sauce": 345, "steamed buns": 353, "strawberries": 354,
    "tomato": 371, "tomato sauce": 372, "waffle cone": 384,
    "whole chicken": 391, "yellow tomatoes": 396,
}
PROBE_TARGET_INDICES = set(PROBE_TARGETS.values())

# Some labels need unambiguous surface forms beyond naive singular/plural.
# Keep this conservative: precision matters more than recall for public transfer.
EXTRA_FORMS = {
    "beef": ["steak", "brisket"],
    "pork": ["pork belly", "bacon", "ham", "prosciutto"],
    "shrimp": ["prawn", "prawns"],
    "noodles": ["ramen", "udon"],
    "pasta": ["spaghetti", "fettuccine", "penne", "linguine", "macaroni", "ravioli"],
    "tomato sauce": ["marinara"],
    "cream sauce": ["creamy sauce"],
    "soft serve ice cream": ["soft serve"],
    "steamed buns": ["bao", "bao buns"],
    "scallions": ["green onions", "spring onions"],
    "green onion": ["green onions", "scallions", "spring onions"],
    "fried egg": ["fried eggs"],
    "french fries": ["fries"],
    "bell pepper": ["bell peppers"],
    "chili pepper": ["chili peppers", "chile pepper", "chile peppers"],
}

# Terms that are too ambiguous as exact description mentions for this analysis.
AMBIGUOUS_LABELS = {
    "orange",  # often color, sauce, lighting, garnish color
    "pepper",  # seasoning vs vegetable
    "chips",  # potato chips vs tortilla chips vs fragments
}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def label_from_caption(caption: str) -> str:
    marker = " containing "
    if marker not in caption:
        raise ValueError(f"Unexpected caption format: {caption}")
    return caption.split(marker, 1)[1].strip().lower()


def singular_plural_forms(label: str) -> set[str]:
    forms = {label}
    tokens = label.split()
    last = tokens[-1]
    candidates = {last}
    if last.endswith("ies") and len(last) > 3:
        candidates.add(last[:-3] + "y")
    elif last.endswith("y"):
        candidates.add(last[:-1] + "ies")
    if last.endswith("oes"):
        candidates.add(last[:-2])
    elif last.endswith("o"):
        candidates.add(last + "es")
    if last.endswith("s") and not last.endswith(("ss", "us")):
        candidates.add(last[:-1])
    else:
        candidates.add(last + "s")
    for cand in candidates:
        if cand != last:
            forms.add(" ".join(tokens[:-1] + [cand]))
    return {form for form in forms if len(form) >= 3}


def build_pattern(label: str) -> re.Pattern[str] | None:
    if label in AMBIGUOUS_LABELS:
        return None
    forms = singular_plural_forms(label)
    forms.update(EXTRA_FORMS.get(label, []))
    escaped = sorted((re.escape(form) for form in forms), key=len, reverse=True)
    return re.compile("|".join(rf"\b{form}\b" for form in escaped), re.IGNORECASE)


def build_test1_gold(image_names: list[str]) -> list[set[int]]:
    cap_vecs = np.load(JUDGE_CACHE / "test1_cap_vecs.npy")
    desc_vecs = np.load(JUDGE_CACHE / "test1_desc_vecs.npy", allow_pickle=True).item()

    gold: list[set[int]] = []
    for image_name in image_names:
        if image_name not in desc_vecs:
            gold.append(set())
            continue
        sims = cap_vecs @ desc_vecs[image_name]
        sorted_sims = np.sort(sims)[::-1]
        gaps = sorted_sims[:-1] - sorted_sims[1:]
        search_range = min(30, len(gaps))
        elbow = int(np.argmax(gaps[:search_range])) + 1
        elbow = max(1, min(elbow, 30))
        gold.append(set(int(idx) for idx in np.argsort(-sims)[:elbow]))
    return gold


def load_submission_preds(path: Path, n_images: int) -> list[set[int]]:
    preds: list[set[int]] = [set() for _ in range(n_images)]
    with path.open() as handle:
        for row in csv.DictReader(handle):
            row_id = int(row["row_id"])
            if row_id >= n_images:
                continue
            preds[row_id] = {int(x) for x in row["class_ids"].split("-") if x}
    return preds


def score_pred_sets(preds: list[set[int]], gold: list[set[int]]) -> dict[str, float]:
    tp = sum(len(p & g) for p, g in zip(preds, gold))
    fp = sum(len(p - g) for p, g in zip(preds, gold))
    fn = sum(len(g - p) for p, g in zip(preds, gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "avg_pred_size": (tp + fp) / max(len(preds), 1),
    }


def add_label_mentions(
    base_preds: list[set[int]],
    mentions_by_label: dict[int, set[int]],
    label_indices: set[int],
) -> list[set[int]]:
    preds = deepcopy(base_preds)
    for label_idx in label_indices:
        for image_idx in mentions_by_label[label_idx]:
            preds[image_idx].add(label_idx)
    return preds


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"test1_qwen3_desc_precision_{time.strftime('%Y%m%d%H%M%S')}"
    out_json = RESULT_DIR / f"{run_id}.json"
    out_csv = RESULT_DIR / f"{run_id}.csv"

    image_names = load_lines(ROOT_DIR / "Test1" / "images.txt")
    captions = load_lines(ROOT_DIR / "Test1" / "captions.txt")
    labels = [label_from_caption(caption) for caption in captions]
    qwen_descs = json.load(open(QWEN_DESC_PATH))
    gold = build_test1_gold(image_names)
    base_preds = load_submission_preds(ALL30_SUBMISSION, len(image_names))
    base_score = score_pred_sets(base_preds, gold)

    patterns = [build_pattern(label) for label in labels]
    mentions_by_label: dict[int, set[int]] = {idx: set() for idx in range(len(labels))}
    for image_idx, image_name in enumerate(image_names):
        desc = qwen_descs.get(image_name, "")
        for label_idx, pattern in enumerate(patterns):
            if pattern is not None and pattern.search(desc):
                mentions_by_label[label_idx].add(image_idx)

    rows: list[dict[str, Any]] = []
    for label_idx, label in enumerate(labels):
        mention_images = mentions_by_label[label_idx]
        add_images = {idx for idx in mention_images if label_idx not in base_preds[idx]}
        add_tp = sum(1 for idx in add_images if label_idx in gold[idx])
        add_fp = len(add_images) - add_tp
        mention_tp = sum(1 for idx in mention_images if label_idx in gold[idx])
        mention_fp = len(mention_images) - mention_tp
        missed_gold = {idx for idx in range(len(image_names)) if label_idx in gold[idx] and label_idx not in base_preds[idx]}

        trial_preds = add_label_mentions(base_preds, mentions_by_label, {label_idx})
        trial_score = score_pred_sets(trial_preds, gold)
        row = {
            "label_idx": label_idx,
            "label": label,
            "in_existing_30": label_idx in PROBE_TARGET_INDICES,
            "pattern_enabled": patterns[label_idx] is not None,
            "gold_count": sum(1 for g in gold if label_idx in g),
            "current_pred_count": sum(1 for p in base_preds if label_idx in p),
            "current_missed_gold": len(missed_gold),
            "qwen_mention_count": len(mention_images),
            "qwen_mention_tp": mention_tp,
            "qwen_mention_fp": mention_fp,
            "qwen_mention_precision": mention_tp / len(mention_images) if mention_images else None,
            "add_candidate_count": len(add_images),
            "add_tp": add_tp,
            "add_fp": add_fp,
            "add_precision": add_tp / len(add_images) if add_images else None,
            "add_recall_of_missed": add_tp / len(missed_gold) if missed_gold else None,
            "single_label_f1_delta": trial_score["f1"] - base_score["f1"],
        }
        rows.append(row)

    # Threshold simulations are selected on the same local judge, so they are diagnostic.
    threshold_results: list[dict[str, Any]] = []
    for min_precision in [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
        for min_tp in [1, 2, 3, 5, 10]:
            selected = {
                int(row["label_idx"])
                for row in rows
                if not row["in_existing_30"]
                and row["pattern_enabled"]
                and row["add_candidate_count"] > 0
                and row["add_tp"] >= min_tp
                and row["add_precision"] is not None
                and row["add_precision"] >= min_precision
            }
            if not selected:
                continue
            preds = add_label_mentions(base_preds, mentions_by_label, selected)
            score = score_pred_sets(preds, gold)
            threshold_results.append({
                "min_precision": min_precision,
                "min_tp": min_tp,
                "n_labels": len(selected),
                "labels": [{"idx": idx, "label": labels[idx]} for idx in sorted(selected)],
                "score": score,
                "delta_f1": score["f1"] - base_score["f1"],
                "delta_precision": score["precision"] - base_score["precision"],
                "delta_recall": score["recall"] - base_score["recall"],
            })

    threshold_results.sort(key=lambda item: (item["score"]["f1"], item["score"]["precision"]), reverse=True)

    existing_30 = [row for row in rows if row["in_existing_30"] and row["add_candidate_count"] > 0]
    candidates = [
        row for row in rows
        if not row["in_existing_30"]
        and row["pattern_enabled"]
        and row["add_candidate_count"] > 0
        and row["single_label_f1_delta"] > 0
    ]
    candidates.sort(
        key=lambda row: (
            row["single_label_f1_delta"],
            row["add_precision"] if row["add_precision"] is not None else -1,
            row["add_tp"],
        ),
        reverse=True,
    )

    manifest = {
        "run_id": run_id,
        "script": str(Path(__file__).relative_to(ROOT_DIR)),
        "submission_reference": str(ALL30_SUBMISSION.relative_to(ROOT_DIR)),
        "description_reference": str(QWEN_DESC_PATH.relative_to(ROOT_DIR)),
        "base_score": base_score,
        "n_images": len(image_names),
        "n_labels": len(labels),
        "existing_30_summary": {
            "n_with_add_candidates": len(existing_30),
            "micro_add_tp": sum(row["add_tp"] for row in existing_30),
            "micro_add_fp": sum(row["add_fp"] for row in existing_30),
            "micro_add_precision": (
                sum(row["add_tp"] for row in existing_30)
                / max(sum(row["add_candidate_count"] for row in existing_30), 1)
            ),
        },
        "top_new_candidates": candidates[:80],
        "threshold_results": threshold_results[:50],
        "all_label_metrics": rows,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with out_json.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    with out_csv.open("w", newline="") as handle:
        fieldnames = [
            "label_idx", "label", "in_existing_30", "pattern_enabled",
            "gold_count", "current_pred_count", "current_missed_gold",
            "qwen_mention_count", "qwen_mention_tp", "qwen_mention_fp", "qwen_mention_precision",
            "add_candidate_count", "add_tp", "add_fp", "add_precision",
            "add_recall_of_missed", "single_label_f1_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    append_jsonl(LOG_PATH, {
        "run_id": run_id,
        "script": manifest["script"],
        "attempt": "test1_qwen3_description_precision_analysis",
        "base_f1": base_score["f1"],
        "existing_30_micro_add_precision": manifest["existing_30_summary"]["micro_add_precision"],
        "best_threshold": threshold_results[0] if threshold_results else None,
        "notes": "Diagnostic local-gold analysis for extending Test1 disambiguation beyond 30 targets.",
        "timestamp": manifest["finished_at"],
    })

    print(f"Base all30 Test1 F1: {base_score['f1']:.6f} precision={base_score['precision']:.6f} recall={base_score['recall']:.6f}")
    print("Existing 30 Qwen-description add-candidate micro precision:",
          f"{manifest['existing_30_summary']['micro_add_precision']:.4f}",
          f"({manifest['existing_30_summary']['micro_add_tp']} TP /",
          f"{manifest['existing_30_summary']['micro_add_fp']} FP)")
    print("\nTop new candidates by single-label local F1 delta:")
    for row in candidates[:25]:
        precision = row["add_precision"]
        print(
            f"  {row['label_idx']:3d} {row['label'][:28]:28s} "
            f"add={row['add_candidate_count']:3d} tp={row['add_tp']:3d} fp={row['add_fp']:3d} "
            f"prec={precision:.3f} delta_f1={row['single_label_f1_delta']:.6f}"
        )
    print("\nBest threshold simulations:")
    for result in threshold_results[:10]:
        score = result["score"]
        print(
            f"  min_prec={result['min_precision']:.2f} min_tp={result['min_tp']:2d} "
            f"labels={result['n_labels']:2d} F1={score['f1']:.6f} "
            f"P={score['precision']:.6f} R={score['recall']:.6f} "
            f"delta={result['delta_f1']:.6f}"
        )
    print(f"\nSaved {out_json}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
