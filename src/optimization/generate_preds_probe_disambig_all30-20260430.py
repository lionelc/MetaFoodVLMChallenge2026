#!/usr/bin/env python3
"""
4-model ensemble + entity Jaccard + probe-based disambiguation for Test1.

Uses all 30 ingredients from probe_yesno_results.json.
- If both Gemma4 and Qwen3 agree "yes" → add the ingredient
- If only one model has a result and it says "yes" → add the ingredient
- If either says "no" → don't add

Usage:
    python3 generate_preds_probe_disambig_all30-20260430.py
"""

import csv
import json
import re
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
SIGLIP_CACHE = ROOT_DIR / "siglip2_cache"
QWEN3_8B_CACHE = ROOT_DIR / "qwen3_cache"
CLIP_CACHE = ROOT_DIR / "clip_cache" / "vit_h_14_dfn5b"
QWEN3_2B_CACHE = ROOT_DIR / "qwen3vl2b_cache"
DE_CACHE = ROOT_DIR / "qwen3_de_cache"

T1_WEIGHTS = [0.25, 0.35, 0.30, 0.10]
T2_WEIGHTS = [0.20, 0.40, 0.15, 0.25]
GAP_SR = 5
GAP_MAX_K = 5
SINKHORN_TAU = 0.15
SINKHORN_ITER = 50

SEASONING_THRESHOLD = 3.3
SEASONING_INDICES = frozenset([11, 16, 100, 121, 187, 188, 315, 329, 372])
CONTAINER_THRESHOLD = 3.2
CONTAINER_INDICES = frozenset([41, 357, 365, 382, 384])

JACCARD_BONUS = 0.20

# All 30 probe ingredients → caption index
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

# Entity vocabulary (for Test2 Jaccard)
ENTITIES = {
    "chicken": ["chicken", "poultry"], "beef": ["beef", "steak", "brisket"],
    "pork": ["pork", "ham", "bacon", "prosciutto"], "lamb": ["lamb"],
    "duck": ["duck"], "turkey": ["turkey"],
    "fish": ["fish", "salmon", "tuna", "cod", "tilapia"],
    "shrimp": ["shrimp", "prawn", "prawns"], "crab": ["crab"],
    "lobster": ["lobster"], "meatball": ["meatball"], "sausage": ["sausage"],
    "patty": ["patty", "patties"], "tofu": ["tofu"],
    "rice": ["rice", "risotto"],
    "pasta": ["pasta", "spaghetti", "fettuccine", "penne", "linguine",
              "macaroni", "ravioli"],
    "noodles": ["noodle", "noodles", "ramen", "udon"],
    "bread": ["bread", "baguette"], "toast": ["toast"], "bun": ["bun", "buns"],
    "tortilla": ["tortilla"], "potato": ["potato", "potatoes", "fries"],
    "broccoli": ["broccoli"], "carrots": ["carrot", "carrots"],
    "green_beans": ["green beans"], "peas": ["peas"], "corn": ["corn"],
    "spinach": ["spinach"], "lettuce": ["lettuce"],
    "tomato": ["tomato", "tomatoes", "cherry tomato"],
    "onion": ["onion", "onions"], "mushroom": ["mushroom", "mushrooms"],
    "pepper": ["pepper", "peppers"], "zucchini": ["zucchini"],
    "asparagus": ["asparagus"], "cabbage": ["cabbage"],
    "cucumber": ["cucumber"], "celery": ["celery"], "avocado": ["avocado"],
    "apple": ["apple", "apples"], "banana": ["banana", "bananas"],
    "orange": ["orange", "oranges"], "lemon": ["lemon"],
    "berries": ["berry", "berries", "blueberries", "strawberries", "raspberries"],
    "grapes": ["grape", "grapes"], "mango": ["mango"],
    "pineapple": ["pineapple"],
    "cheese": ["cheese", "mozzarella", "cheddar", "parmesan", "feta"],
    "cream_sauce": ["cream sauce", "creamy sauce"],
    "tomato_sauce": ["tomato sauce", "marinara"], "gravy": ["gravy"],
    "sushi": ["sushi", "nigiri", "sashimi"], "pizza": ["pizza"],
    "burger": ["burger", "hamburger"], "sandwich": ["sandwich"],
    "taco": ["taco", "tacos"],
    "soup": ["soup", "stew", "broth", "chowder"], "salad": ["salad"],
    "cake": ["cake"], "ice_cream": ["ice cream", "gelato"],
    "eggs": ["egg", "eggs", "omelette"],
    "fried": ["fried", "crispy"], "grilled": ["grilled"],
    "roasted": ["roasted"], "steamed": ["steamed"], "baked": ["baked"],
}
ENTITY_PATTERNS = {
    e: re.compile("|".join(rf"\b{re.escape(f)}\b" for f in forms), re.IGNORECASE)
    for e, forms in ENTITIES.items()
}


def z_norm(m):
    mu = m.mean(axis=1, keepdims=True)
    std = m.std(axis=1, keepdims=True) + 1e-12
    return (m - mu) / std


def adaptive_gap(row, sr=GAP_SR, max_k=GAP_MAX_K):
    ranked = np.sort(row)[::-1]
    s = min(sr, len(ranked) - 1)
    gaps = ranked[:s] - ranked[1:s + 1]
    return max(1, min(int(np.argmax(gaps)) + 1, max_k))


def build_probe_lookup():
    """Build lookup from probe_yesno_results.json.
    Returns: {(image, ingredient): "yes"/"no"} using consensus logic."""
    probes = json.load(open(ROOT_DIR / "probe_yesno_results.json"))["probes"]
    lookup = {}  # (fname, ingredient) → "yes" / "no"

    for p in probes:
        fname = p["image"]
        ing = p["ingredient"]
        g = p.get("gemma4_answer", "")
        q = p.get("qwen3_answer", "")

        # Both have valid answers
        if g in ("yes", "no") and q in ("yes", "no"):
            if g == q:
                lookup[(fname, ing)] = g  # agree → take it
            else:
                lookup[(fname, ing)] = "no"  # disagree → conservative
        # Only one model has result
        elif g in ("yes", "no"):
            lookup[(fname, ing)] = g
        elif q in ("yes", "no"):
            lookup[(fname, ing)] = q

    return lookup


def main():
    t_start = time.time()
    pred_csv = Path(__file__).resolve().parent / f"submission-entity-jaccard-{time.strftime('%Y%m%d%H%M%S')}.csv"

    # Build probe lookup
    probe_lookup = build_probe_lookup()
    n_yes = sum(1 for v in probe_lookup.values() if v == "yes")
    n_no = sum(1 for v in probe_lookup.values() if v == "no")
    print(f"Probe lookup: {len(probe_lookup)} entries ({n_yes} yes, {n_no} no)")
    print(f"Covering {len(PROBE_TARGETS)} ingredients")

    # Load Test1 images
    with open(ROOT_DIR / "Test1/images.txt") as f:
        imgs_t1 = [l.strip() for l in f if l.strip()]

    # Test1: base predictions
    print("\nLoading Test1 embeddings...")
    s1n = z_norm(np.load(SIGLIP_CACHE/"test1_image_features.npy") @ np.load(SIGLIP_CACHE/"test1_text_features.npy").T)
    q1n = z_norm(np.load(QWEN3_8B_CACHE/"qwen3vl8b_test1_img.npy") @ np.load(QWEN3_8B_CACHE/"qwen3vl8b_test1_text.npy").T)
    c1n = z_norm(np.load(CLIP_CACHE/"test1_image_features.npy") @ np.load(CLIP_CACHE/"test1_text_features.npy").T)
    b1n = z_norm(np.load(QWEN3_2B_CACHE/"test1_image_features.npy") @ np.load(QWEN3_2B_CACHE/"test1_text_features.npy").T)
    combined1 = T1_WEIGHTS[0]*s1n + T1_WEIGHTS[1]*q1n + T1_WEIGHTS[2]*c1n + T1_WEIGHTS[3]*b1n

    # Build predictions with disambiguation
    multi_rows = []
    n_seas = n_cont = n_disambig = 0
    per_ingredient_added = {ing: 0 for ing in PROBE_TARGETS}

    for i in range(combined1.shape[0]):
        row = combined1[i]
        k = adaptive_gap(row)
        selected = set(np.argsort(-row)[:k].tolist())

        # Seasoning/container boosts
        for cidx in SEASONING_INDICES:
            if cidx not in selected and row[cidx] > SEASONING_THRESHOLD:
                selected.add(cidx); n_seas += 1
        for cidx in CONTAINER_INDICES:
            if cidx not in selected and row[cidx] > CONTAINER_THRESHOLD:
                selected.add(cidx); n_cont += 1

        # Probe disambiguation
        fname = imgs_t1[i]
        for ingredient, cap_idx in PROBE_TARGETS.items():
            if cap_idx in selected:
                continue
            ans = probe_lookup.get((fname, ingredient))
            if ans == "yes":
                selected.add(cap_idx)
                n_disambig += 1
                per_ingredient_added[ingredient] += 1

        selected = np.sort(list(selected))
        multi_rows.append((i, "-".join(str(int(c)) for c in selected), "multi"))

    print(f"\nTest1: {n_seas} seas + {n_cont} cont + {n_disambig} disambig labels added")
    print(f"  Per ingredient:")
    for ing, count in sorted(per_ingredient_added.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {ing:25s}: +{count}")

    # Test2: Sinkhorn with entity Jaccard (unchanged)
    print("\nLoading Test2...")
    captions = json.load(open(ROOT_DIR / "Test2/captions.json"))
    with open(ROOT_DIR / "Test2/images.txt") as f:
        imgs_t2 = [l.strip() for l in f if l.strip()]
    descs = json.load(open(DE_CACHE / "test2_desc.json"))

    cap_ents = [{e for e, p in ENTITY_PATTERNS.items() if p.search(c)} for c in captions]
    img_ents = [{e for e, p in ENTITY_PATTERNS.items() if p.search(descs.get(f, ""))} for f in imgs_t2]
    jaccard = np.zeros((len(imgs_t2), len(captions)), dtype=np.float32)
    for i in range(len(imgs_t2)):
        ie = img_ents[i]
        if not ie: continue
        for j in range(len(captions)):
            ce = cap_ents[j]
            if ce:
                u = ie | ce
                if u: jaccard[i, j] = len(ie & ce) / len(u)

    s2n = z_norm(np.load(SIGLIP_CACHE/"test2_image_features.npy") @ np.load(SIGLIP_CACHE/"test2_text_features.npy").T)
    q2n = z_norm(np.load(QWEN3_8B_CACHE/"qwen3vl8b_test2_img.npy") @ np.load(QWEN3_8B_CACHE/"qwen3vl8b_test2_text.npy").T)
    c2n = z_norm(np.load(CLIP_CACHE/"test2_image_features.npy") @ np.load(CLIP_CACHE/"test2_text_features.npy").T)
    b2n = z_norm(np.load(QWEN3_2B_CACHE/"test2_image_features.npy") @ np.load(QWEN3_2B_CACHE/"test2_text_features.npy").T)
    combined2 = T2_WEIGHTS[0]*s2n + T2_WEIGHTS[1]*q2n + T2_WEIGHTS[2]*c2n + T2_WEIGHTS[3]*b2n
    boosted = combined2 + JACCARD_BONUS * jaccard

    # Sinkhorn
    log_K = boosted / SINKHORN_TAU
    log_K -= log_K.max()
    K = np.exp(log_K)
    u = np.ones(len(imgs_t2)); v = np.ones(len(captions))
    for _ in range(SINKHORN_ITER):
        u = 1.0 / (K @ v + 1e-12)
        v = 1.0 / (K.T @ u + 1e-12)
    P = np.diag(u) @ K @ np.diag(v)
    assignment = {}; used = set()
    for i in np.argsort(-P.max(axis=1)):
        row = P[i].copy(); row[list(used)] = -1
        best = int(np.argmax(row))
        assignment[int(i)] = best; used.add(best)

    n_multi = len(multi_rows)
    single_rows = [(n_multi + i, str(assignment[i]), "single") for i in range(len(imgs_t2))]

    with open(pred_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "Usage", "class_ids", "Task"])
        for row_id, class_ids, task in multi_rows + single_rows:
            w.writerow([row_id, "Public", class_ids, task])
    print(f"\n✅ {pred_csv}")

    import subprocess
    subprocess.run(["python3.13", str(ROOT_DIR / "score_submission.py"), str(pred_csv)])
    print(f"Time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
