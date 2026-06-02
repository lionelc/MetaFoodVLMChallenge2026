#!/usr/bin/env python3
"""
Analyze Qwen3 + Gemma4 cached-description ensembles for Test1 disambiguation.

Rules evaluated per ingredient label:
  qwen   - Qwen3 description mentions the label
  gemma  - Gemma4 structured description mentions the label
  or     - either description mentions the label
  and    - both descriptions mention the label

All metrics are measured as ADD candidates on top of the best public-safe all30
submission, and compared against the local Gemini-derived Test1 gold. This is
diagnostic; no Kaggle submission is implied.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
OPT_DIR = ROOT_DIR / "optimization"
RESULT_DIR = OPT_DIR / "experiment_results"
QWEN_ANALYSIS_PATH = OPT_DIR / "analyze_test1_qwen3_description_precision-20260501.py"
ALL30_SUBMISSION = OPT_DIR / "submission-entity-jaccard-20260430200009.csv"
QWEN_DESC_PATH = ROOT_DIR / "qwen3_de_cache" / "test1_desc.json"
GEMMA_DESC_PATH = ROOT_DIR / "gemma4_cache" / "test1_desc.json"


def load_qwen_analysis_module():
    spec = importlib.util.spec_from_file_location("test1_qwen_analysis", QWEN_ANALYSIS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helpers from {QWEN_ANALYSIS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qa = load_qwen_analysis_module()


def detect_mentions(
    image_names: list[str],
    labels: list[str],
    patterns: list[Any],
    descs: dict[str, str],
) -> dict[int, set[int]]:
    mentions = {idx: set() for idx in range(len(labels))}
    for image_idx, image_name in enumerate(image_names):
        desc = descs.get(image_name, "")
        for label_idx, pattern in enumerate(patterns):
            if pattern is not None and pattern.search(desc):
                mentions[label_idx].add(image_idx)
    return mentions


def score_added_rule(
    label_idx: int,
    label: str,
    mention_images: set[int],
    base_preds: list[set[int]],
    gold: list[set[int]],
    base_score: dict[str, float],
) -> dict[str, Any]:
    add_images = {idx for idx in mention_images if label_idx not in base_preds[idx]}
    add_tp = sum(1 for idx in add_images if label_idx in gold[idx])
    add_fp = len(add_images) - add_tp
    mention_tp = sum(1 for idx in mention_images if label_idx in gold[idx])
    mention_fp = len(mention_images) - mention_tp
    missed_gold = {idx for idx in range(len(gold)) if label_idx in gold[idx] and label_idx not in base_preds[idx]}

    trial_preds = deepcopy(base_preds)
    for image_idx in add_images:
        trial_preds[image_idx].add(label_idx)
    trial_score = qa.score_pred_sets(trial_preds, gold)

    return {
        "label_idx": label_idx,
        "label": label,
        "in_existing_30": label_idx in qa.PROBE_TARGET_INDICES,
        "gold_count": sum(1 for item in gold if label_idx in item),
        "current_pred_count": sum(1 for item in base_preds if label_idx in item),
        "current_missed_gold": len(missed_gold),
        "mention_count": len(mention_images),
        "mention_tp": mention_tp,
        "mention_fp": mention_fp,
        "mention_precision": mention_tp / len(mention_images) if mention_images else None,
        "add_candidate_count": len(add_images),
        "add_tp": add_tp,
        "add_fp": add_fp,
        "add_precision": add_tp / len(add_images) if add_images else None,
        "add_recall_of_missed": add_tp / len(missed_gold) if missed_gold else None,
        "single_label_f1_delta": trial_score["f1"] - base_score["f1"],
    }


def add_mentions_to_preds(
    base_preds: list[set[int]],
    mentions_by_rule_label: dict[tuple[str, int], set[int]],
    selected: set[tuple[str, int]],
) -> list[set[int]]:
    preds = deepcopy(base_preds)
    for rule, label_idx in selected:
        for image_idx in mentions_by_rule_label[(rule, label_idx)]:
            preds[image_idx].add(label_idx)
    return preds


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"test1_qwen_gemma_desc_ensemble_{time.strftime('%Y%m%d%H%M%S')}"
    out_json = RESULT_DIR / f"{run_id}.json"
    out_csv = RESULT_DIR / f"{run_id}.csv"

    image_names = qa.load_lines(ROOT_DIR / "Test1" / "images.txt")
    captions = qa.load_lines(ROOT_DIR / "Test1" / "captions.txt")
    labels = [qa.label_from_caption(caption) for caption in captions]
    patterns = [qa.build_pattern(label) for label in labels]
    gold = qa.build_test1_gold(image_names)
    base_preds = qa.load_submission_preds(ALL30_SUBMISSION, len(image_names))
    base_score = qa.score_pred_sets(base_preds, gold)

    qwen_descs = json.load(open(QWEN_DESC_PATH))
    gemma_descs = json.load(open(GEMMA_DESC_PATH))
    qwen_mentions = detect_mentions(image_names, labels, patterns, qwen_descs)
    gemma_mentions = detect_mentions(image_names, labels, patterns, gemma_descs)

    rule_mentions: dict[str, dict[int, set[int]]] = {
        "qwen": qwen_mentions,
        "gemma": gemma_mentions,
        "or": {idx: qwen_mentions[idx] | gemma_mentions[idx] for idx in range(len(labels))},
        "and": {idx: qwen_mentions[idx] & gemma_mentions[idx] for idx in range(len(labels))},
    }

    rows: list[dict[str, Any]] = []
    mentions_by_rule_label: dict[tuple[str, int], set[int]] = {}
    for rule, mentions_by_label in rule_mentions.items():
        for label_idx, label in enumerate(labels):
            if patterns[label_idx] is None:
                continue
            mentions_by_rule_label[(rule, label_idx)] = mentions_by_label[label_idx]
            row = score_added_rule(label_idx, label, mentions_by_label[label_idx], base_preds, gold, base_score)
            row["rule"] = rule
            rows.append(row)

    threshold_results: list[dict[str, Any]] = []
    for min_precision in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        for min_tp in [1, 2, 3, 5, 10]:
            for allowed_rules in [["and"], ["qwen"], ["gemma"], ["qwen", "gemma", "and"]]:
                selected: set[tuple[str, int]] = set()
                selected_labels: set[int] = set()
                for row in rows:
                    label_idx = int(row["label_idx"])
                    if row["rule"] not in allowed_rules:
                        continue
                    if row["in_existing_30"]:
                        continue
                    if row["add_candidate_count"] <= 0:
                        continue
                    if row["add_tp"] < min_tp:
                        continue
                    if row["add_precision"] is None or row["add_precision"] < min_precision:
                        continue
                    # Do not add the same label by multiple rules; take the highest precision,
                    # then highest TP, rule if multiple rules qualify.
                    if label_idx in selected_labels:
                        current = [item for item in selected if item[1] == label_idx][0]
                        cur_row = next(r for r in rows if r["rule"] == current[0] and r["label_idx"] == label_idx)
                        cur_key = (cur_row["add_precision"] or -1, cur_row["add_tp"], cur_row["single_label_f1_delta"])
                        new_key = (row["add_precision"] or -1, row["add_tp"], row["single_label_f1_delta"])
                        if new_key > cur_key:
                            selected.remove(current)
                            selected.add((row["rule"], label_idx))
                        continue
                    selected.add((row["rule"], label_idx))
                    selected_labels.add(label_idx)

                if not selected:
                    continue
                preds = add_mentions_to_preds(base_preds, mentions_by_rule_label, selected)
                score = qa.score_pred_sets(preds, gold)
                threshold_results.append({
                    "allowed_rules": allowed_rules,
                    "min_precision": min_precision,
                    "min_tp": min_tp,
                    "n_rule_labels": len(selected),
                    "n_labels": len(selected_labels),
                    "selected": [
                        {
                            "rule": rule,
                            "idx": label_idx,
                            "label": labels[label_idx],
                            "add_candidate_count": next(
                                r["add_candidate_count"] for r in rows
                                if r["rule"] == rule and r["label_idx"] == label_idx
                            ),
                            "add_tp": next(
                                r["add_tp"] for r in rows
                                if r["rule"] == rule and r["label_idx"] == label_idx
                            ),
                            "add_fp": next(
                                r["add_fp"] for r in rows
                                if r["rule"] == rule and r["label_idx"] == label_idx
                            ),
                            "add_precision": next(
                                r["add_precision"] for r in rows
                                if r["rule"] == rule and r["label_idx"] == label_idx
                            ),
                        }
                        for rule, label_idx in sorted(selected, key=lambda item: (item[1], item[0]))
                    ],
                    "score": score,
                    "delta_f1": score["f1"] - base_score["f1"],
                    "delta_precision": score["precision"] - base_score["precision"],
                    "delta_recall": score["recall"] - base_score["recall"],
                })

    threshold_results.sort(key=lambda item: (item["score"]["f1"], item["score"]["precision"]), reverse=True)

    rule_summaries: dict[str, dict[str, Any]] = {}
    for rule in rule_mentions:
        rule_rows = [
            row for row in rows
            if row["rule"] == rule
            and not row["in_existing_30"]
            and row["add_candidate_count"] > 0
        ]
        add_total = sum(row["add_candidate_count"] for row in rule_rows)
        tp_total = sum(row["add_tp"] for row in rule_rows)
        fp_total = sum(row["add_fp"] for row in rule_rows)
        positive_rows = [row for row in rule_rows if row["single_label_f1_delta"] > 0]
        rule_summaries[rule] = {
            "n_labels_with_add_candidates": len(rule_rows),
            "micro_add_tp": tp_total,
            "micro_add_fp": fp_total,
            "micro_add_precision": tp_total / add_total if add_total else None,
            "n_positive_delta_labels": len(positive_rows),
            "top_positive_delta": sorted(
                positive_rows,
                key=lambda row: (row["single_label_f1_delta"], row["add_precision"] or -1, row["add_tp"]),
                reverse=True,
            )[:30],
        }

    manifest = {
        "run_id": run_id,
        "script": str(Path(__file__).relative_to(ROOT_DIR)),
        "base_submission": str(ALL30_SUBMISSION.relative_to(ROOT_DIR)),
        "qwen_desc": str(QWEN_DESC_PATH.relative_to(ROOT_DIR)),
        "gemma_desc": str(GEMMA_DESC_PATH.relative_to(ROOT_DIR)),
        "base_score": base_score,
        "rule_summaries": rule_summaries,
        "threshold_results": threshold_results[:80],
        "all_rule_label_metrics": rows,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with out_json.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    with out_csv.open("w", newline="") as handle:
        fieldnames = [
            "rule", "label_idx", "label", "in_existing_30", "gold_count", "current_pred_count",
            "current_missed_gold", "mention_count", "mention_tp", "mention_fp", "mention_precision",
            "add_candidate_count", "add_tp", "add_fp", "add_precision", "add_recall_of_missed",
            "single_label_f1_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    qa.append_jsonl(qa.LOG_PATH, {
        "run_id": run_id,
        "script": manifest["script"],
        "attempt": "test1_qwen_gemma_description_ensemble_analysis",
        "base_f1": base_score["f1"],
        "best_threshold": threshold_results[0] if threshold_results else None,
        "rule_summaries": {
            rule: {
                "micro_add_precision": summary["micro_add_precision"],
                "n_positive_delta_labels": summary["n_positive_delta_labels"],
            }
            for rule, summary in rule_summaries.items()
        },
        "timestamp": manifest["finished_at"],
    })

    print(f"Base all30 Test1 F1: {base_score['f1']:.6f} P={base_score['precision']:.6f} R={base_score['recall']:.6f}")
    print("\nRule summaries for non-30 add candidates:")
    for rule, summary in rule_summaries.items():
        precision = summary["micro_add_precision"]
        print(
            f"  {rule:5s}: labels={summary['n_labels_with_add_candidates']:3d} "
            f"TP={summary['micro_add_tp']:4d} FP={summary['micro_add_fp']:4d} "
            f"precision={(precision if precision is not None else 0):.4f} "
            f"positive_labels={summary['n_positive_delta_labels']:3d}"
        )
    print("\nBest threshold simulations:")
    for result in threshold_results[:12]:
        score = result["score"]
        print(
            f"  rules={'+'.join(result['allowed_rules']):14s} "
            f"minP={result['min_precision']:.2f} minTP={result['min_tp']:2d} "
            f"labels={result['n_labels']:2d} F1={score['f1']:.6f} "
            f"P={score['precision']:.6f} R={score['recall']:.6f} "
            f"delta={result['delta_f1']:.6f}"
        )
    print("\nTop positive AND candidates:")
    for row in rule_summaries["and"]["top_positive_delta"][:25]:
        print(
            f"  {row['label_idx']:3d} {row['label'][:28]:28s} "
            f"add={row['add_candidate_count']:3d} tp={row['add_tp']:3d} fp={row['add_fp']:3d} "
            f"prec={row['add_precision']:.3f} delta={row['single_label_f1_delta']:.6f}"
        )
    print(f"\nSaved {out_json}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
