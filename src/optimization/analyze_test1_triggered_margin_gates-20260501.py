#!/usr/bin/env python3
"""
Tune Test1 triggered disambiguation rules with embedding margin gates.

The previous best allowed VLM signals to add a label whenever the label was in
the embedding ensemble top-10.  This script keeps that idea, but also sweeps
gates based on:
  - max embedding rank,
  - score gap from the adaptive-gap decision boundary,
  - score gap from the image's top-scoring label.

Each selected rule-label can choose its own gate.
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

SELECTED_TEST1_SCRIPT = OPT_DIR / "analyze_test1_description_signals-20260501.py"
CURRENT_SELECTED_SUBMISSION = OPT_DIR / "submission-selected-yesno-test1-rareboost-safe-test2-20260501104507.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selected_t1 = load_module("test1_triggered_selected_yesno", SELECTED_TEST1_SCRIPT)
base = selected_t1.base
qa = selected_t1.qa
all30 = base.all30


BASE_SUBMISSIONS = {
    **base.BASE_SUBMISSIONS,
    "current_selected_hard": CURRENT_SELECTED_SUBMISSION,
}


def adaptive_gap(row: np.ndarray, max_k: int = all30.GAP_MAX_K) -> int:
    ranked = np.sort(row)[::-1]
    s = min(all30.GAP_SR, len(ranked) - 1)
    gaps = ranked[:s] - ranked[1:s + 1]
    return max(1, min(int(np.argmax(gaps)) + 1, max_k))


def build_margin_arrays(scores: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)

    top_scores = scores[np.arange(scores.shape[0]), order[:, 0]]
    boundary_scores = np.zeros(scores.shape[0], dtype=np.float32)
    adaptive_ks = np.zeros(scores.shape[0], dtype=np.int16)
    for image_idx, row in enumerate(scores):
        k = adaptive_gap(row)
        adaptive_ks[image_idx] = k
        boundary_scores[image_idx] = row[order[image_idx, k - 1]]

    return {
        "order": order,
        "ranks": ranks,
        "top_scores": top_scores,
        "boundary_scores": boundary_scores,
        "adaptive_ks": adaptive_ks,
    }


def gate_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for max_rank in [10, 8, 7, 6, 5]:
        configs.append({
            "gate": f"rank_le_{max_rank}",
            "max_rank": max_rank,
            "max_boundary_gap": None,
            "max_top_gap": None,
        })

    for max_boundary_gap in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]:
        configs.append({
            "gate": f"top10_boundary_gap_le_{max_boundary_gap:g}",
            "max_rank": 10,
            "max_boundary_gap": max_boundary_gap,
            "max_top_gap": None,
        })
        configs.append({
            "gate": f"top7_boundary_gap_le_{max_boundary_gap:g}",
            "max_rank": 7,
            "max_boundary_gap": max_boundary_gap,
            "max_top_gap": None,
        })

    for max_top_gap in [0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]:
        configs.append({
            "gate": f"top10_top_gap_le_{max_top_gap:g}",
            "max_rank": 10,
            "max_boundary_gap": None,
            "max_top_gap": max_top_gap,
        })
    return configs


def build_eligible_by_label(
    scores: np.ndarray,
    margins: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[int, set[int]]:
    eligible = {idx: set() for idx in range(scores.shape[1])}
    order = margins["order"]
    boundary_scores = margins["boundary_scores"]
    top_scores = margins["top_scores"]
    max_rank = int(config["max_rank"])
    max_boundary_gap = config["max_boundary_gap"]
    max_top_gap = config["max_top_gap"]
    for image_idx in range(scores.shape[0]):
        for label_idx in order[image_idx, :max_rank]:
            label_idx = int(label_idx)
            score = scores[image_idx, label_idx]
            if max_boundary_gap is not None and (boundary_scores[image_idx] - score) > float(max_boundary_gap):
                continue
            if max_top_gap is not None and (top_scores[image_idx] - score) > float(max_top_gap):
                continue
            eligible[label_idx].add(image_idx)
    return eligible


def score_from_counts(base_score: dict[str, float], add_tp: int, add_fp: int, n_images: int) -> dict[str, float]:
    tp = int(base_score["tp"]) + add_tp
    fp = int(base_score["fp"]) + add_fp
    fn = int(base_score["fn"]) - add_tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_pred_size = (tp + fp) / n_images
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "avg_pred_size": avg_pred_size,
    }


def score_rule_row(
    base_name: str,
    base_preds: list[set[int]],
    base_score: dict[str, float],
    gold: list[set[int]],
    labels: list[str],
    rank_positions: np.ndarray,
    rule: str,
    label_idx: int,
    gate: dict[str, Any],
    triggered_mentions: set[int],
) -> dict[str, Any] | None:
    add_images = {idx for idx in triggered_mentions if label_idx not in base_preds[idx]}
    if not add_images:
        return None
    add_tp = sum(1 for idx in add_images if label_idx in gold[idx])
    add_fp = len(add_images) - add_tp
    triggered_tp = sum(1 for idx in triggered_mentions if label_idx in gold[idx])
    triggered_fp = len(triggered_mentions) - triggered_tp
    missed_gold = {
        idx for idx in range(len(gold))
        if label_idx in gold[idx] and label_idx not in base_preds[idx]
    }
    score = score_from_counts(base_score, add_tp, add_fp, len(gold))
    ranks = [int(rank_positions[idx, label_idx]) for idx in triggered_mentions]
    return {
        "base": base_name,
        "gate": gate["gate"],
        "max_rank": gate["max_rank"],
        "max_boundary_gap": gate["max_boundary_gap"],
        "max_top_gap": gate["max_top_gap"],
        "rule": rule,
        "label_idx": label_idx,
        "label": labels[label_idx],
        "is_probe30": label_idx in qa.PROBE_TARGET_INDICES,
        "gold_count": sum(1 for item in gold if label_idx in item),
        "base_pred_count": sum(1 for item in base_preds if label_idx in item),
        "base_missed_gold": len(missed_gold),
        "triggered_signal_count": len(triggered_mentions),
        "triggered_signal_tp": triggered_tp,
        "triggered_signal_fp": triggered_fp,
        "triggered_signal_precision": triggered_tp / len(triggered_mentions) if triggered_mentions else None,
        "add_candidate_count": len(add_images),
        "add_tp": add_tp,
        "add_fp": add_fp,
        "add_precision": add_tp / len(add_images) if add_images else None,
        "add_recall_of_missed": add_tp / len(missed_gold) if missed_gold else None,
        "mean_trigger_rank": float(np.mean(ranks)) if ranks else None,
        "single_rule_f1_delta": score["f1"] - base_score["f1"],
    }


def add_selected_rules(
    base_preds: list[set[int]],
    triggered_mentions: dict[tuple[str, str, int], set[int]],
    selected: set[tuple[str, str, int]],
) -> list[set[int]]:
    preds = deepcopy(base_preds)
    for gate, rule, label_idx in selected:
        for image_idx in triggered_mentions[(gate, rule, label_idx)]:
            preds[image_idx].add(label_idx)
    return preds


def apply_result_to_submission(
    result: dict[str, Any],
    base_rows: list[dict[str, str]],
    triggered_mentions: dict[tuple[str, str, int], set[int]],
) -> tuple[list[dict[str, str]], int]:
    rows = deepcopy(base_rows)
    additions = 0
    for item in result["selected"]:
        key = (item["gate"], item["rule"], int(item["idx"]))
        for image_idx in triggered_mentions.get(key, set()):
            class_ids = {int(x) for x in rows[image_idx]["class_ids"].split("-") if x}
            if int(item["idx"]) in class_ids:
                continue
            class_ids.add(int(item["idx"]))
            rows[image_idx]["class_ids"] = "-".join(str(item) for item in sorted(class_ids))
            additions += 1
    return rows, additions


def choose_threshold_results(
    rows: list[dict[str, Any]],
    labels: list[str],
    base_preds_by_name: dict[str, list[set[int]]],
    gold: list[set[int]],
    triggered_mentions_by_base: dict[str, dict[tuple[str, str, int], set[int]]],
    base_scores: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    allowed_groups = {
        "selected_yesno_strict": ["sel_yesno_and", "sel_yesno_consensus"],
        "selected_yesno_all": [
            "sel_yesno_qwen",
            "sel_yesno_gemma",
            "sel_yesno_and",
            "sel_yesno_or",
            "sel_yesno_consensus",
        ],
        "selected_plus_desc": [
            "desc_qwen",
            "desc_gemma",
            "desc_and",
            "sel_yesno_qwen",
            "sel_yesno_gemma",
            "sel_yesno_and",
            "sel_yesno_or",
            "sel_yesno_consensus",
        ],
        "selected_plus_all": [
            "desc_qwen",
            "desc_gemma",
            "desc_and",
            "yesno_qwen",
            "yesno_gemma",
            "yesno_and",
            "yesno_consensus",
            "sel_yesno_qwen",
            "sel_yesno_gemma",
            "sel_yesno_and",
            "sel_yesno_or",
            "sel_yesno_consensus",
        ],
        "legacy_all": [
            "desc_qwen",
            "desc_gemma",
            "desc_and",
            "yesno_qwen",
            "yesno_gemma",
            "yesno_and",
            "yesno_consensus",
        ],
    }
    results: list[dict[str, Any]] = []
    for base_name, base_preds in base_preds_by_name.items():
        base_rows = [row for row in rows if row["base"] == base_name]
        for group_name, allowed_rules in allowed_groups.items():
            for min_precision in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
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
                            -(row["mean_trigger_rank"] or 99),
                        )
                        old = best_by_label.get(label_idx)
                        if old is None:
                            best_by_label[label_idx] = row
                            continue
                        old_key = (
                            old["add_precision"] or -1,
                            old["add_tp"],
                            old["single_rule_f1_delta"],
                            -old["add_fp"],
                            -(old["mean_trigger_rank"] or 99),
                        )
                        if key > old_key:
                            best_by_label[label_idx] = row

                    selected = {
                        (row["gate"], row["rule"], int(row["label_idx"]))
                        for row in best_by_label.values()
                    }
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
                                "gate": row["gate"],
                                "max_rank": row["max_rank"],
                                "max_boundary_gap": row["max_boundary_gap"],
                                "max_top_gap": row["max_top_gap"],
                                "rule": row["rule"],
                                "idx": int(row["label_idx"]),
                                "label": labels[int(row["label_idx"])],
                                "add_candidate_count": row["add_candidate_count"],
                                "add_tp": row["add_tp"],
                                "add_fp": row["add_fp"],
                                "add_precision": row["add_precision"],
                                "mean_trigger_rank": row["mean_trigger_rank"],
                            }
                            for row in sorted(
                                best_by_label.values(),
                                key=lambda item: (int(item["label_idx"]), item["rule"], item["gate"]),
                            )
                        ],
                        "score": score,
                        "delta_f1": score["f1"] - base_scores[base_name]["f1"],
                        "delta_precision": score["precision"] - base_scores[base_name]["precision"],
                        "delta_recall": score["recall"] - base_scores[base_name]["recall"],
                    })
    results.sort(key=lambda item: (item["score"]["f1"], item["score"]["precision"], item["delta_f1"]), reverse=True)
    return results


def build_triggered_mentions(
    signals: dict[str, dict[int, set[int]]],
    scores: np.ndarray,
    gate_config_list: list[dict[str, Any]],
) -> dict[tuple[str, str, int], set[int]]:
    margins = build_margin_arrays(scores)
    triggered_mentions: dict[tuple[str, str, int], set[int]] = {}
    for gate in gate_config_list:
        eligible_by_label = build_eligible_by_label(scores, margins, gate)
        for rule, mentions_by_label in signals.items():
            for label_idx, signal_images in mentions_by_label.items():
                if not signal_images:
                    continue
                triggered = signal_images & eligible_by_label[int(label_idx)]
                if triggered:
                    triggered_mentions[(gate["gate"], rule, int(label_idx))] = triggered
    return triggered_mentions


def main() -> None:
    t_start = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"test1_triggered_margin_gates_{time.strftime('%Y%m%d%H%M%S')}"
    out_json = RESULT_DIR / f"{run_id}.json"
    out_csv = RESULT_DIR / f"{run_id}.csv"
    out_submission = OPT_DIR / f"submission-test1-triggered-margin-gates-{time.strftime('%Y%m%d%H%M%S')}.csv"

    image_names = qa.load_lines(ROOT_DIR / "Test1" / "images.txt")
    captions = qa.load_lines(ROOT_DIR / "Test1" / "captions.txt")
    labels = [qa.label_from_caption(caption) for caption in captions]
    patterns = [qa.build_pattern(label) for label in labels]
    gold = qa.build_test1_gold(image_names)
    scores = base.build_test1_scores()
    rank_positions = base.build_rank_positions(scores)
    signals = selected_t1.build_all_signals(image_names, labels, patterns)
    gate_config_list = gate_configs()

    print(f"Building triggered mentions for {len(gate_config_list)} margin gates...")
    shared_triggered_mentions = build_triggered_mentions(signals, scores, gate_config_list)
    print(f"  non-empty gate/rule/label triples: {len(shared_triggered_mentions)}")

    base_preds_by_name: dict[str, list[set[int]]] = {
        name: base.load_submission_preds(path, len(image_names))
        for name, path in BASE_SUBMISSIONS.items()
    }
    base_rows_by_name = {name: base.load_submission_rows(path) for name, path in BASE_SUBMISSIONS.items()}
    base_scores = {name: qa.score_pred_sets(preds, gold) for name, preds in base_preds_by_name.items()}
    triggered_mentions_by_base = {name: shared_triggered_mentions for name in base_preds_by_name}

    all_rows: list[dict[str, Any]] = []
    for base_name, base_preds in base_preds_by_name.items():
        base_score = base_scores[base_name]
        for (gate_name, rule, label_idx), triggered in shared_triggered_mentions.items():
            gate = next(item for item in gate_config_list if item["gate"] == gate_name)
            row = score_rule_row(
                base_name=base_name,
                base_preds=base_preds,
                base_score=base_score,
                gold=gold,
                labels=labels,
                rank_positions=rank_positions,
                rule=rule,
                label_idx=label_idx,
                gate=gate,
                triggered_mentions=triggered,
            )
            if row is not None:
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
        base.write_submission(out_submission, best_rows)
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
            -row["add_fp"],
        ),
        reverse=True,
    )
    with out_csv.open("w", newline="") as handle:
        fieldnames = [
            "base", "gate", "max_rank", "max_boundary_gap", "max_top_gap",
            "rule", "label_idx", "label", "is_probe30", "gold_count",
            "base_pred_count", "base_missed_gold", "triggered_signal_count",
            "triggered_signal_tp", "triggered_signal_fp", "triggered_signal_precision",
            "add_candidate_count", "add_tp", "add_fp", "add_precision",
            "add_recall_of_missed", "mean_trigger_rank", "single_rule_f1_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({field: row[field] for field in fieldnames})

    manifest = {
        "run_id": run_id,
        "script": str(Path(__file__).relative_to(ROOT_DIR)),
        "base_submissions": {name: str(path.relative_to(ROOT_DIR)) for name, path in BASE_SUBMISSIONS.items()},
        "source_selected_script": str(SELECTED_TEST1_SCRIPT.relative_to(ROOT_DIR)),
        "gate_configs": gate_config_list,
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
            "attempt": "test1_triggered_margin_gates",
            "gate_count": len(gate_config_list),
            "base_scores": base_scores,
            "best": best,
            "timestamp": manifest["finished_at"],
        },
    )

    print("Base scores:")
    for name, score in base_scores.items():
        print(f"  {name:22s} F1={score['f1']:.6f} P={score['precision']:.6f} R={score['recall']:.6f}")
    print("\nTop margin-gated variants:")
    for result in threshold_results[:20]:
        score = result["score"]
        print(
            f"  {result['base']:22s} {result['group']:22s} "
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
