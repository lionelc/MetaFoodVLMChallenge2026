#!/usr/bin/env python3
"""
Tune Test1 top-K triggered rules with the new selective yes/no caches.

The selective caches only cover image-label pairs that were already chosen by the
previous best triggered-disambiguation run.  They are therefore evaluated as
optional replacement signals, not mandatory vetoes.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
OPT_DIR = ROOT_DIR / "optimization"
RESULT_DIR = OPT_DIR / "experiment_results"
DISAMBIG_CACHE = ROOT_DIR / "disambiguation_cache"

BASE_TEST1_SCRIPT = OPT_DIR / "analyze_test1_triggered_disambig_rules-20260501.py"
CACHE_MANIFEST = RESULT_DIR / "test1_description_signal_cache_20260501102452.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("test1_triggered_disambig_base", BASE_TEST1_SCRIPT)
qa = base.qa


def safe_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def cache_path(model: str, label_idx: int, label: str) -> Path:
    return DISAMBIG_CACHE / f"{model}_test1_triggered_yesno_{label_idx:03d}_{safe_label(label)}.json"


def load_cache(model: str, label_idx: int, label: str) -> dict[str, str]:
    path = cache_path(model, label_idx, label)
    if not path.exists():
        return {}
    loaded = json.load(open(path))
    cache: dict[str, str] = {}
    for image_name, value in loaded.items():
        if isinstance(value, str):
            cache[image_name] = value
        elif isinstance(value, dict) and isinstance(value.get("answer"), str):
            cache[image_name] = value["answer"]
    return cache


def build_selected_yesno_signals(
    image_names: list[str],
    labels: list[str],
) -> dict[str, dict[int, set[int]]]:
    image_to_idx = {name: idx for idx, name in enumerate(image_names)}
    signals = {
        "sel_yesno_qwen": {idx: set() for idx in range(len(labels))},
        "sel_yesno_gemma": {idx: set() for idx in range(len(labels))},
        "sel_yesno_and": {idx: set() for idx in range(len(labels))},
        "sel_yesno_or": {idx: set() for idx in range(len(labels))},
        "sel_yesno_consensus": {idx: set() for idx in range(len(labels))},
    }

    for label_idx, label in enumerate(labels):
        qwen = load_cache("qwen3", label_idx, label)
        gemma = load_cache("gemma4", label_idx, label)
        if not qwen and not gemma:
            continue
        for image_name in sorted(set(qwen) | set(gemma)):
            image_idx = image_to_idx.get(image_name)
            if image_idx is None:
                continue
            qwen_yes = qwen.get(image_name) == "yes"
            gemma_yes = gemma.get(image_name) == "yes"
            if qwen_yes:
                signals["sel_yesno_qwen"][label_idx].add(image_idx)
            if gemma_yes:
                signals["sel_yesno_gemma"][label_idx].add(image_idx)
            if qwen_yes and gemma_yes:
                signals["sel_yesno_and"][label_idx].add(image_idx)
            if qwen_yes or gemma_yes:
                signals["sel_yesno_or"][label_idx].add(image_idx)

            valid = [answer for answer in (qwen.get(image_name), gemma.get(image_name)) if answer in ("yes", "no")]
            if valid and all(answer == "yes" for answer in valid):
                signals["sel_yesno_consensus"][label_idx].add(image_idx)
    return signals


def build_all_signals(
    image_names: list[str],
    labels: list[str],
    patterns: list[Any],
) -> dict[str, dict[int, set[int]]]:
    desc_signals = base.build_description_signals(image_names, labels, patterns)
    return desc_signals


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
            for min_precision in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
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
                            continue
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
    return base.apply_result_to_submission(result, base_rows, triggered_mentions)


def main() -> None:
    t_start = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"test1_triggered_selected_yesno_rules_{time.strftime('%Y%m%d%H%M%S')}"
    out_json = RESULT_DIR / f"{run_id}.json"
    out_csv = RESULT_DIR / f"{run_id}.csv"
    out_submission = OPT_DIR / f"submission-test1-triggered-selected-yesno-{time.strftime('%Y%m%d%H%M%S')}.csv"

    image_names = qa.load_lines(ROOT_DIR / "Test1" / "images.txt")
    captions = qa.load_lines(ROOT_DIR / "Test1" / "captions.txt")
    labels = [qa.label_from_caption(caption) for caption in captions]
    patterns = [qa.build_pattern(label) for label in labels]
    gold = qa.build_test1_gold(image_names)
    scores = base.build_test1_scores()
    topk_images = base.build_topk_images_by_label(scores, base.TOP_K)
    rank_positions = base.build_rank_positions(scores)
    signals = build_all_signals(image_names, labels, patterns)

    base_preds_by_name: dict[str, list[set[int]]] = {
        name: base.load_submission_preds(path, len(image_names))
        for name, path in base.BASE_SUBMISSIONS.items()
    }
    base_rows_by_name = {name: base.load_submission_rows(path) for name, path in base.BASE_SUBMISSIONS.items()}
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
                all_rows.append(base.score_added_rule(
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
                ))

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
        "top_k": base.TOP_K,
        "base_script": str(BASE_TEST1_SCRIPT.relative_to(ROOT_DIR)),
        "cache_manifest": str(CACHE_MANIFEST.relative_to(ROOT_DIR)),
        "base_submissions": {name: str(path.relative_to(ROOT_DIR)) for name, path in base.BASE_SUBMISSIONS.items()},
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
            "attempt": "test1_triggered_selected_yesno_rules",
            "top_k": base.TOP_K,
            "base_scores": base_scores,
            "best": best,
            "timestamp": manifest["finished_at"],
        },
    )

    print("Base scores:")
    for name, score in base_scores.items():
        print(f"  {name:22s} F1={score['f1']:.6f} P={score['precision']:.6f} R={score['recall']:.6f}")
    print("\nTop triggered selected-yes/no variants:")
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
