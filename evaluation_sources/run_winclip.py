"""One explicit-support WinCLIP+ evaluation; invoked only after BTAD final gate.

The model and CLIP weights are from Anomalib 2.6.2. This adapter changes neither
PromptAD source nor its result files. Each run is one separate OS process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/baselines"
WEIGHT_DIR = ROOT / "pretrained/cache/clip"
WEIGHT = WEIGHT_DIR / "vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
WEIGHT_SHA256 = "8fb26589b9a8bab2e7a683d280e48df89b8e742f3a82132707282620b36facba"
VISA = ROOT / "datasets/raw/VisA"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gate() -> None:
    sys.path.insert(0, str(ROOT / "scripts/ssra"))
    from overnight_status import alive  # type: ignore

    status_path = ROOT / "results/ssra/final_status.txt"
    # overnight_runner writes the experiment terminal marker first; the
    # independently audited finalizer subsequently replaces it with the final
    # research marker. Both are valid only with the audit checks below.
    assert status_path.is_file() and status_path.read_text(encoding="utf-8").strip() in {
        "SSRA-EXPERIMENTS-COMPLETE", "SSRA-FINAL-COMPLETE"
    }, "BTAD final status absent"
    assert not (ROOT / "results/ssra/overnight.lock").exists(), "BTAD controller lock remains"
    progress = json.loads((ROOT / "results/ssra/overnight_progress.json").read_text(encoding="utf-8"))
    assert not alive(progress.get("runner_pid")), "BTAD controller process still alive"
    assert not alive(progress.get("child_pid")), "BTAD GPU child process still alive"
    assert progress["total_complete"] == 45 and progress["failed"] == 0, "BTAD 45-cell result not clean"
    audit_path = ROOT / "results/json/ssra_completion_audit.json"
    assert audit_path.is_file(), "BTAD completion audit missing"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["passed"] == audit["applicable"] and audit["applicable"] > 0, "BTAD completion audit failed"
    with (ROOT / "results/csv/ssra_btad_raw.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 45 and all(r["status"] == "success" for r in rows)


def audit_written_cell(cell: Path, metrics_path: Path, task: dict,
                       test: list[str], official: dict, result: dict) -> None:
    """Technical artifact gate applied to every cell, including the 15 pilot cells."""
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved == result and saved["status"] == "success"
    assert (saved["category"], saved["shot"], saved["support_seed"]) == (
        task["category"], task["shot"], task["support_seed"])
    assert saved["support_paths"] == task["support_paths"]
    assert saved["support_sha256"] == task["support_sha256"]
    assert saved["weight_sha256"] == WEIGHT_SHA256
    assert saved["test_count"] == task["test_count"] == len(test)
    assert math.isfinite(float(saved["elapsed_seconds"])) and float(saved["elapsed_seconds"]) > 0
    assert 0 < int(saved["peak_vram_bytes"]) < 8 * 2**30
    for metric in ("I_AUROC", "P_AUROC", "AUPRO"):
        value = float(saved[metric])
        assert math.isfinite(value) and 0 <= value <= 100
    with (cell / "per_image.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert [record["image"] for record in records] == test
    assert all(int(record["label"]) == int(official[record["image"]]["label"] != "normal")
               and math.isfinite(float(record["pred_score"]))
               and len(record["anomaly_map_sha256"]) == 64 for record in records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--shot", required=True, type=int, choices=(1, 2, 4))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()

    plan = json.loads((OUT / "baseline_plan.json").read_text(encoding="utf-8"))
    matches = [t for t in plan["tasks"] if (t["category"], t["shot"], t["support_seed"]) ==
               (args.category, args.shot, args.seed)]
    assert len(matches) == 1, "configuration absent from audited plan"
    task = matches[0]
    assert WEIGHT.is_file() and sha(WEIGHT) == WEIGHT_SHA256, "official e31 weight missing or hash mismatch"
    split = json.loads(Path(task["split_json"]).read_text(encoding="utf-8"))["categories"][args.category]
    assert split["support"] == task["support_paths"]
    assert [sha(VISA / p) for p in split["support"]] == task["support_sha256"]
    if args.preflight:
        print(json.dumps({"preflight": "PASS", "category": args.category, "shot": args.shot,
                          "seed": args.seed, "test_count": task["test_count"], "weight_sha256": sha(WEIGHT)}))
        return
    gate()

    # Give Anomalib's immutable WinCLIP implementation the already verified local
    # official weight, instead of silently downloading it to an untracked cache.
    os.environ.setdefault("HF_HOME", str(ROOT / "pretrained/cache/hf"))
    import cv2
    import numpy as np
    import open_clip
    import torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
    from anomalib.models.image.winclip.torch_model import WinClipModel

    original_create = open_clip.create_model_and_transforms

    def local_create(*a, **kw):
        kw["cache_dir"] = str(WEIGHT_DIR)
        return original_create(*a, **kw)

    open_clip.create_model_and_transforms = local_create
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    start = time.perf_counter()
    model = WinClipModel(class_name=args.category)
    model.to(device).eval()
    print(f"STAGE model_loaded category={args.category} shot={args.shot} seed={args.seed}", flush=True)
    transform = Compose((ToTensor(), Resize((240, 240), interpolation=InterpolationMode.BICUBIC, antialias=True),
                         Normalize((0.48145466, 0.4578275, 0.40821073),
                                   (0.26862954, 0.26130258, 0.27577711))))

    def tensor(path: Path):
        with Image.open(path) as im:
            return transform(im.convert("RGB"))

    with torch.inference_mode():
        support = torch.stack([tensor(VISA / p) for p in task["support_paths"]]).to(device)
        model.setup(args.category, support)
        assert model.k_shot == args.shot, "few-shot reference count mismatch"
        with (VISA / "split_csv/1cls.csv").open(encoding="utf-8", newline="") as f:
            official = {r["image"]: r for r in csv.DictReader(f)}
        labels, scores, maps, masks, image_rows = [], [], [], [], []
        for index, rel in enumerate(split["test"], 1):
            row = official[rel]
            label = int(row["label"] != "normal")
            pred = model(tensor(VISA / rel).unsqueeze(0).to(device))
            score = float(pred.pred_score.item())
            amap = pred.anomaly_map.squeeze(0).detach().cpu().numpy().astype(np.float32)
            amap = cv2.resize(amap, (400, 400), interpolation=cv2.INTER_CUBIC)
            if label:
                raw_mask = cv2.imread(str(VISA / row["mask"]), cv2.IMREAD_GRAYSCALE)
                assert raw_mask is not None, row["mask"]
                mask = cv2.resize(raw_mask, (400, 400), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask = np.zeros((400, 400), dtype=bool)
            assert np.isfinite(score) and np.isfinite(amap).all()
            labels.append(label)
            scores.append(score)
            maps.append(amap)
            masks.append(mask)
            image_rows.append({"image": rel, "label": label, "pred_score": f"{score:.12g}",
                               "anomaly_map_sha256": hashlib.sha256(amap.tobytes()).hexdigest()})
            if index % 40 == 0 or index == len(split["test"]):
                print(f"STAGE inference {index}/{len(split['test'])}", flush=True)

    assert len(labels) == task["test_count"] and len(set(labels)) == 2
    score_maps = np.stack(maps)
    mask_maps = np.stack(masks)
    image_auc = float(roc_auc_score(labels, scores) * 100)
    print("STAGE pixel_auc", flush=True)
    pixel_auc = float(roc_auc_score(mask_maps.ravel(), score_maps.ravel()) * 100)
    spec = importlib.util.spec_from_file_location("promptad_official_metrics", ROOT / "code/PromptAD/utils/metrics.py")
    assert spec and spec.loader
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    print("STAGE aupro_200", flush=True)
    aupro = float(metrics_module.cal_pro_metric(mask_maps, score_maps, fpr_thresh=0.3, max_steps=200) * 100)
    elapsed = time.perf_counter() - start
    result = {"dataset": "VisA", "detector": "WinCLIP+", "implementation": "anomalib==2.6.2",
              "category": args.category, "shot": args.shot, "support_seed": args.seed,
              "support_paths": task["support_paths"], "support_sha256": task["support_sha256"],
              "weight_sha256": sha(WEIGHT), "score_definition": "Anomalib WinClipModel.pred_score",
              "map_definition": "Anomalib WinClipModel.anomaly_map; 400x400 bicubic for project metric",
              "I_AUROC": image_auc, "P_AUROC": pixel_auc, "AUPRO": aupro,
              "test_count": len(labels), "elapsed_seconds": elapsed,
              "peak_vram_bytes": torch.cuda.max_memory_allocated(), "status": "success"}
    cell = OUT / "winclip" / args.category / f"shot_{args.shot}" / f"support_{args.seed}"
    cell.mkdir(parents=True, exist_ok=True)
    with (cell / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)
    tmp = cell / "metrics.json.tmp"
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    audit_written_cell(cell, tmp, task, split["test"], official, result)
    tmp.replace(cell / "metrics.json")
    assert json.loads((cell / "metrics.json").read_text(encoding="utf-8")) == result
    print(json.dumps(result))
    print("WORKER_RESULT_WRITTEN", flush=True)


if __name__ == "__main__":
    main()
