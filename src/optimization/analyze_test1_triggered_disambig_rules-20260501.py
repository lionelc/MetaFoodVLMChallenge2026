#!/usr/bin/env python3
"""
Tune Test1 disambiguation rules only on top-K triggered ingredients.

For each Test1 image, an ingredient label is "triggered" if it appears in the
embedding ensemble's top-K candidates. Cached disambiguation signals are only
allowed to add that label when it is triggered.

Signals evaluated:
  - Qwen3/Gemma4 description mention rules: qwen, gemma, and, or
  - cached yes/no rules from probe_yesno_results.json for the 30 probed labels:
    qwen yes, gemma yes, both yes, and conservative consensus yes

This script is diagnostic. It writes a best local submission preserving Test2
rows from the chosen base submission.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
OPT_DIR = ROOT_DIR / "optimization"
RESULT_DIR = OPT_DIR / "experiment_results"

ALL30_SCRIPT = OPT_DIR / "generate_preds_probe_disambig_all30-20260430.py"
ENSEMBLE_ANALYSIS_SCRIPT = OPT_DIR / "analyze_test1_qwen_gemma_description_ensemble-20260501.py"

BASE_SUBMISSIONS = {
    "all30": OPT_DIR / "submission-entity-jaccard-20260430200009.csv",
    "qwen_gemma_ensemble70": OPT_DIR / "submission-probe-all30-qwen-gemma-ensemble70-20260501071817.csv",
    "qwen_gemma_bestlocal": OPT_DIR / "submission-probe-all30-qwen-gemma-bestlocal-20260501071759.csv",
}

QWEN_DESC_PATH = ROOT_DIR / "qwen3_de_cache" / "test1_desc.json"
GEMMA_DESC_PATH = ROOT_DIR / "gemma4_cache" / "test1_desc.json"

TOP_K = 10


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


all30 = load_module("test1_all30_generator", ALL30_SCRIPT)
ensemble = load_module("test1_qwen_gemma_ensemble", ENSEMBLE_ANALYSIS_SCRIPT)
qa = ensemble.qa


def load_submission_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_submission_preds(path: Path, n_images: int) -> list[set[int]]:
    preds = [set() for _ in range(n_images)]
    with path.open() as handle:
        for row in csv.DictReader(handle):
            row_id = int(row["row_id"])
            if row_id >= n_images:
                continue
            preds[row_id] = {int(item) for item in row["class_ids"].split("-") if item}
    return preds


def format_class_ids(class_ids: set[int]) -> str:
    return "-".join(str(item) for item in sorted(class_ids))


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "Usage", "class_ids", "Task"])
        writer.writeheader()
        writer.writerows(rows)


def build_test1_scores() -> np.ndarray:
    s1n = all30.z_norm(
        np.load(all30.SIGLIP_CACHE / "test1_image_features.npy")
        @ np.load(all30.SIGLIP_CACHE / "test1_text_features.npy").T
    )
    q1n = all30.z_norm(
        np.load(all30.QWEN3_8B_CACHE / "qwen3vl8b_test1_img.npy")
        @ np.load(all30.QWEN3_8B_CACHE / "qwen3vl8b_test1_text.npy").T
    )
    c1n = all30.z_norm(
        np.load(all30.CLIP_CACHE / "test1_image_features.npy")
        @ np.load(all30.CLIP_CACHE / "test1_text_features.npy").T
    )
    b1n = all30.z_norm(
        np.load(all30.QWEN3_2B_CACHE / "test1_image_features.npy")
        @ np.load(all30.QWEN3_2B_CACHE / "test1_text_features.npy").T
    )
    return (
        all30.T1_WEIGHTS[0] * s1n
        + all30.T1_WEIGHTS[1] * q1n
        + all30.T1_WEIGHTS[2] * c1n
        + all30.T1_WEIGHTS[3] * b1n
    )


def build_topk_images_by_label(scores: np.ndarray, top_k: int) -> dict[int, set[int]]:
    top = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    by_label = {idx: set() for idx in range(scores.shape[1])}
    for image_idx, labels in enumerate(top):
        for label_idx in labels:
            by_label[int(label_idx)].add(int(image_idx))
    return by_label


def build_rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    row_idx = np.arange(scores.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, scores.shape[1] + 1)
    return ranks


def build_description_signals(
    image_names: list[str],
    labels: list[str],
    patterns: list[Any],
) -> dict[str, dict[int, set[int]]]:
    qwen_descs = json.load(open(QWEN_DESC_PATH))
    gemma_descs = json.load(open(GEMMA_DESC_PATH))
    qwen = ensemble.detect_mentions(image_names, labels, patterns, qwen_descs)
    gemma = ensemble.detect_mentions(image_names, labels, patterns, gemma_descs)
    return {
        "desc_qwen": qwen,
        "desc_gemma": gemma,
        "desc_and": {idx: qwen[idx] & gemma[idx] for idx in range(len(labels))},
        "desc_or": {idx: qwen[idx] | gemma[idx] for idx in range(len(labels))},
    }


def score_added_rule(
    base_name: str,
    rule: str,
    label_idx: int,
    label: str,
    signal_images: set[int],
    triggered_images: set[int],
    base_preds: list[set[int]],
    gold: list[set[int]],
    base_score: dict[str, float],
    rank_positions: np.ndarray,
) -> dict[str, Any]:
    triggered_signal = signal_images & triggered_images
    add_images = {idx for idx in triggered_signal if label_idx not in base_preds[idx]}
    add_tp = sum(1 for idx in add_images if label_idx in gold[idx])
    add_fp = len(add_images) - add_tp
    missed_gold = {
        idx for idx in range(len(gold))
        if label_idx in gold[idx] and label_idx not in base_preds[idx]
    }
    trial_preds = deepcopy(base_preds)
    for image_idx in add_images:
        trial_preds[image_idx].add(label_idx)
    trial_score = qa.score_pred_sets(trial_preds, gold)
    ranks = [int(rank_positions[idx, label_idx]) for idx in triggered_signal]
    return {
        "base": base_name,
        "rule": rule,
        "label_idx": label_idx,
        "label": label,
        "is_probe30": label_idx in qa.PROBE_TARGET_INDICES,
        "gold_count": sum(1 for item in gold if label_idx in item),
        "base_pred_count": sum(1 for item in base_preds if label_idx in item),
        "base_missed_gold": len(missed_gold),
        "signal_count": len(signal_images),
        "triggered_signal_count": len(triggered_signal),
        "triggered_signal_tp": sum(1 for idx in triggered_signal if label_idx in gold[idx]),
        "triggered_signal_fp": sum(1 for idx in triggered_signal if label_idx not in gold[idx]),
        "triggered_signal_precision": (
            sum(1 for idx in triggered_signal if label_idx in gold[idx]) / len(triggered_signal)
            if triggered_signal else None
        ),
        "add_candidate_count": len(add_images),
        "add_tp": add_tp,
        "add_fp": add_fp,
        "add_precision": add_tp / len(add_images) if add_images else None,
        "add_recall_of_missed": add_tp / len(missed_gold) if missed_gold else None,
        "mean_trigger_rank": float(np.mean(ranks)) if ranks else None,
        "single_rule_f1_delta": trial_score["f1"] - base_score["f1"],
    }


def add_selected_rules(
    base_preds: list[set[int]],
    triggered_mentions: dict[tuple[str, int], set[int]],
    selected: set[tuple[str, int]],
) -> list[set[int]]:
    preds = deepcopy(base_preds)
    for rule, label_idx in selected:
        for image_idx in triggered_mentions[(rule, label_idx)]:
            preds[image_idx].add(label_idx)
    return preds


def choose_threshold_results(
    rows: list[dict[str, Any]],
    labels: list[str],
    base_preds_by_name: dict[str, list[set[int]]],
    gold: list[set[int]],
    triggered_mentions_by_base: dict[str, dict[tuple[str, int], set[int]]],
    base_scores: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    allowed_groups = {
        "desc_and": ["desc_and"],
        "desc_ensemble": ["desc_qwen", "desc_gemma", "desc_and"],
        "yesno_strict": ["yesno_and", "yesno_consensus"],
        "yesno_all": ["yesno_qwen", "yesno_gemma", "yesno_and", "yesno_consensus"],
        "safe_all": ["desc_and", "yesno_and", "yesno_consensus"],
        "all_rules": ["desc_qwen", "desc_gemma", "desc_and", "yesno_qwen", "yesno_gemma", "yesno_and", "yesno_consensus"],
    }
    results: list[dict[str, Any]] = []
    for base_name, base_preds in base_preds_by_name.items():
        base_rows = [row for row in rows if row["base"] == base_name]
        for group_name, allowed_rules in allowed_groups.items():
            for min_precision in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
                for min_tp in [1, 2, 3, 5, 10]:
                    candidates = [
                        row for row in base_rows
                        if row["rule"] in allowed_rules
                        and row["add_candidate_count"] > 0
                        and row["add_tp"] >= min_tp
                        and row["add_precision"] is not None
                        and row["add_precision"] >= min_precision
                    ]
                    if not candidates:
                        continue
                    best_by_label: dict[int, dict[str, Any]] = {}
                    for row in candidates:
                        label_idx = int(row["label_idx"])
                        key = (
                            row["add_precision"] or -1,
                            row["add_tp"],
                            row["single_rule_f1_delta"],
                            -row["add_fp"],
                        )
                        old = best_by_label.get(label_idx)
                        if old is None:
                            best_by_label[label_idx] = row
                        else:
                            old_key = (
                                old["add_precision"] or -1,
                                old["add_tp"],
                                old["single_rule_f1_delta"],
                                -old["add_fp"],
                            )
                            if key > old_key:
                                best_by_label[label_idx] = row
                    selected = {(row["rule"], int(row["label_idx"])) for row in best_by_label.values()}
                    preds = add_selected_rules(base_preds, triggered_mentions_by_base[base_name], selected)
                    score = qa.score_pred_sets(preds, gold)
                    results.append({
                        "base": base_name,
                        "group": group_name,
                        "allowed_rules": allowed_rules,
                        "min_precision": min_precision,
                        "min_tp": min_tp,
                        "n_rule_labels": len(selected),
                        "n_labels": len(best_by_label),
                        "selected": [
                            {
                                "rule": row["rule"],
                                "idx": int(row["label_idx"]),
                                "label": labels[int(row["label_idx"])],
                                "add_candidate_count": row["add_candidate_count"],
                                "add_tp": row["add_tp"],
                                "add_fp": row["add_fp"],
                                "add_precision": row["add_precision"],
                            }
                            for row in sorted(best_by_label.values(), key=lambda item: (int(item["label_idx"]), item["rule"]))
                        ],
                        "score": score,
                        "delta_f1": score["f1"] - base_scores[base_name]["f1"],
                        "delta_precision": score["precision"] - base_scores[base_name]["precision"],
                        "delta_recall": score["recall"] - base_scores[base_name]["recall"],
                    })
    results.sort(key=lambda item: (item["score"]["f1"], item["score"]["precision"], item["delta_f1"]), reverse=True)
    return results


def apply_result_to_submission(
    result: dict[str, Any],
    base_rows: list[dict[str, str]],
    triggered_mentions: dict[tuple[str, int], set[int]],
) -> tuple[list[dict[str, str]], int]:
    rows = deepcopy(base_rows)
    additions = 0
    for item in result["selected"]:
        key = (item["rule"], int(item["idx"]))
        for image_idx in triggered_mentions[key]:
            class_ids = {int(x) for x in rows[image_idx]["class_ids"].split("-") if x}
            if int(item["idx"]) in class_ids:
                continue
            class_ids.add(int(item["idx"]))
            rows[image_idx]["class_ids"] = format_class_ids(class_ids)
            additions += 1
    return rows, additions


def main() -> None:
    t_start = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"test1_triggered_disambig_rules_{time.strftime('%Y%m%d%H%M%S')}"
    out_json = RESULT_DIR / f"{run_id}.json"
    out_csv = RESULT_DIR / f"{run_id}.csv"
    out_submission = OPT_DIR / f"submission-test1-triggered-disambig-{time.strftime('%Y%m%d%H%M%S')}.csv"

    image_names = qa.load_lines(ROOT_DIR / "Test1" / "images.txt")
    captions = qa.load_lines(ROOT_DIR / "Test1" / "captions.txt")
    labels = [qa.label_from_caption(caption) for caption in captions]
    patterns = [qa.build_pattern(label) for label in labels]
    gold = qa.build_test1_gold(image_names)
    scores = build_test1_scores()
    topk_images = build_topk_images_by_label(scores, TOP_K)
    rank_positions = build_rank_positions(scores)

    desc_signals = build_description_signals(image_names, labels, patterns)
    yesno_signals = build_yesno_signals(image_names, labels)
    signals = {**desc_signals, **yesno_signals}

    base_preds_by_name: dict[str, list[set[int]]] = {
        name: load_submission_preds(path, len(image_names))
        for name, path in BASE_SUBMISSIONS.items()
    }
    base_rows_by_name = {name: load_submission_rows(path) for name, path in BASE_SUBMISSIONS.items()}
    base_scores = {name: qa.score_pred_sets(preds, gold) for name, preds in base_preds_by_name.items()}

    all_rows: list[dict[str, Any]] = []
    triggered_mentions_by_base: dict[str, dict[tuple[str, int], set[int]]] = {}
    for base_name, base_preds in base_preds_by_name.items():
        base_score = base_scores[base_name]
        triggered_mentions_by_base[base_name] = {}
        for rule, mentions_by_label in signals.items():
            for label_idx, label in enumerate(labels):
                signal_images = mentions_by_label.get(label_idx, set())
                if not signal_images:
                    continue
                triggered_images = topk_images[label_idx]
                triggered_mentions = signal_images & triggered_images
                triggered_mentions_by_base[base_name][(rule, label_idx)] = triggered_mentions
                row = score_added_rule(
                    base_name=base_name,
                    rule=rule,
                    label_idx=label_idx,
                    label=label,
                    signal_images=signal_images,
                    triggered_images=triggered_images,
                    base_preds=base_preds,
                    gold=gold,
                    base_score=base_score,
                    rank_positions=rank_positions,
                )
                all_rows.append(row)

    threshold_results = choose_threshold_results(
        all_rows,
        labels,
        base_preds_by_name,
        gold,
        triggered_mentions_by_base,
        base_scores,
    )

    best = threshold_results[0] if threshold_results else None
    score_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    additions_total = 0
    if best is not None:
        best_rows, additions_total = apply_result_to_submission(
            best,
            base_rows_by_name[best["base"]],
            triggered_mentions_by_base[best["base"]],
        )
        write_submission(out_submission, best_rows)
        score_proc = subprocess.run(
            ["python3.13", str(ROOT_DIR / "score_submission.py"), str(out_submission)],
            capture_output=True,
            text=True,
            check=False,
        )
        if score_proc.stdout:
            print(score_proc.stdout, end="")
        if score_proc.stderr:
            print(score_proc.stderr, end="")

    # Keep CSV concise: rows with actual add candidates or notable trigger precision.
    csv_rows = [
        row for row in all_rows
        if row["add_candidate_count"] > 0
        or row["triggered_signal_count"] >= 5
    ]
    csv_rows.sort(
        key=lambda row: (
            row["base"],
            row["single_rule_f1_delta"],
            row["add_precision"] or -1,
            row["add_tp"],
        ),
        reverse=True,
    )

    with out_csv.open("w", newline="") as handle:
        fieldnames = [
            "base", "rule", "label_idx", "label", "is_probe30", "gold_count",
            "base_pred_count", "base_missed_gold", "signal_count",
            "triggered_signal_count", "triggered_signal_tp", "triggered_signal_fp",
            "triggered_signal_precision", "add_candidate_count", "add_tp", "add_fp",
            "add_precision", "add_recall_of_missed", "mean_trigger_rank",
            "single_rule_f1_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({field: row[field] for field in fieldnames})

    manifest = {
        "run_id": run_id,
        "script": str(Path(__file__).relative_to(ROOT_DIR)),
        "top_k": TOP_K,
        "base_submissions": {name: str(path.relative_to(ROOT_DIR)) for name, path in BASE_SUBMISSIONS.items()},
        "base_scores": base_scores,
        "signal_names": sorted(signals),
        "threshold_results": threshold_results[:100],
        "best": best,
        "best_submission": str(out_submission.relative_to(ROOT_DIR)) if best is not None else None,
        "best_additions_total": additions_total,
        "local_score_stdout": score_proc.stdout,
        "local_score_stderr": score_proc.stderr,
        "score_returncode": score_proc.returncode,
        "top_rule_rows": csv_rows[:200],
        "outputs": {"csv": str(out_csv.relative_to(ROOT_DIR))},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": time.time() - t_start,
    }
    with out_json.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    qa.append_jsonl(
        qa.LOG_PATH,
        {
            "run_id": run_id,
            "script": manifest["script"],
            "attempt": "test1_triggered_disambig_rules",
            "top_k": TOP_K,
            "base_scores": base_scores,
            "best": best,
            "timestamp": manifest["finished_at"],
        },
    )

    print("Base scores:")
    for name, score in base_scores.items():
        print(f"  {name:22s} F1={score['f1']:.6f} P={score['precision']:.6f} R={score['recall']:.6f}")
    print("\nTop triggered disambiguation variants:")
    for result in threshold_results[:20]:
        score = result["score"]
        print(
            f"  {result['base']:22s} {result['group']:13s} "
            f"minP={result['min_precision']:.2f} minTP={result['min_tp']:2d} "
            f"rules={result['n_rule_labels']:2d} F1={score['f1']:.6f} "
            f"delta={result['delta_f1']:.6f} P={score['precision']:.6f} R={score['recall']:.6f}"
        )
    print(f"\nSaved {out_json}")
    print(f"Saved {out_csv}")
    if best is not None:
        print(f"Best diagnostic submission: {out_submission}")
    print(f"Time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
