#!/usr/bin/env python3
"""
Generate Qwen3 and Gemma4 description caches in parallel using subprocesses.
Each model runs in its own process to avoid GPU contention from threading.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DESC_CACHE = ROOT_DIR / "description_cache"

MODELS = {
    "qwen3": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "splits": [
            ("test1", ROOT_DIR / "Test1" / "imgs", ROOT_DIR / "Test1" / "images.txt",
             DESC_CACHE / "qwen3_de_cache" / "test1_desc.json"),
            ("test2", ROOT_DIR / "Test2" / "imgs", ROOT_DIR / "Test2" / "images.txt",
             DESC_CACHE / "qwen3_de_cache" / "test2_desc.json"),
        ],
    },
    "gemma4": {
        "model_id": "google/gemma-4-E4B-it",
        "splits": [
            ("test1", ROOT_DIR / "Test1" / "imgs", ROOT_DIR / "Test1" / "images.txt",
             DESC_CACHE / "gemma4_cache" / "test1_desc.json"),
        ],
    },
}

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

SAVE_EVERY = 50


def run_worker(model_name: str):
    """Worker: load one model and generate all its descriptions."""
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor

    config = MODELS[model_name]
    model_id = config["model_id"]

    if model_name == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        print(f"[{model_name}] Loading {model_id}...", flush=True)
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map="auto",
        )
    else:
        from transformers import Gemma4ForConditionalGeneration
        print(f"[{model_name}] Loading {model_id}...", flush=True)
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map="auto",
        )
    model.eval()

    for split_name, image_dir, images_txt, cache_path in config["splits"]:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        image_names = [l.strip() for l in open(images_txt) if l.strip()]
        missing = [n for n in image_names if n not in cached]
        print(f"[{model_name}] {split_name}: cached={len(cached)} missing={len(missing)}", flush=True)
        if not missing:
            continue

        t_start = time.time()
        for idx, image_name in enumerate(tqdm(missing, desc=f"{model_name} {split_name}"), 1):
            try:
                image = Image.open(image_dir / image_name).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DESCRIBE_PROMPT},
                ]}]
                if model_name == "qwen3":
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                    trimmed = out[:, inputs.input_ids.shape[1]:]
                    desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
                else:
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                    ).to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
                    desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
                cached[image_name] = desc
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cached[image_name] = "OOM"
            except Exception as exc:
                cached[image_name] = f"ERROR: {type(exc).__name__}: {exc}"

            if idx % SAVE_EVERY == 0:
                cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True))
        cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True))
        elapsed = time.time() - t_start
        print(f"[{model_name}] {split_name} done: {len(missing)} images in {elapsed/60:.1f} min", flush=True)

    del model, processor
    import gc; gc.collect()
    torch.cuda.empty_cache()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2])
        return

    t_start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Launch both models as separate processes
    procs = {}
    for model_name in ["qwen3", "gemma4"]:
        cmd = [sys.executable, __file__, "--worker", model_name]
        print(f"Launching {model_name} worker...", flush=True)
        procs[model_name] = subprocess.Popen(cmd, env=env)

    # Wait for both
    for model_name, proc in procs.items():
        proc.wait()
        if proc.returncode != 0:
            print(f"{model_name} failed with exit code {proc.returncode}")
        else:
            print(f"{model_name} completed successfully")

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
