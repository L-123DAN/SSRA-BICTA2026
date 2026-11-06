"""Evaluate official best SEG checkpoint with PromptAD's dormant AUPRO function."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISA = ROOT / "datasets/raw/VisA"
sys.path.insert(0, str(ROOT / "code/PromptAD"))
os.environ.setdefault("HF_HOME", str(ROOT / "pretrained/cache/hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "pretrained/cache/torch"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from datasets import CLIPDataset  # noqa: E402
from PromptAD import CLIPAD, PromptAD  # noqa: E402
from utils.metrics import cal_pro_metric, metric_cal_pix  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    split = json.loads(Path(config["split_json"]).read_text(encoding="utf-8"))
    rows = {row["image"]: row for row in csv.DictReader((VISA / "split_csv/1cls.csv").open(newline=""))}
    category = config["class_name"]

    def load_indexed(_, shot):
        assert shot == split["shot"]

        def convert(paths):
            out = ([], [], [], [])
            for rel in paths:
                row = rows[rel]
                label = int(row["label"] != "normal")
                out[0].append(str(VISA / rel))
                out[1].append(str(VISA / row["mask"]) if label else 0)
                out[2].append(label)
                out[3].append("bad" if label else "good")
            return out

        c = split["categories"][category]
        return convert(c["support"]), convert(c["test"])

    dataset = CLIPDataset(load_indexed, category, "test", config["k_shot"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    original_create = CLIPAD.create_model_and_transforms

    def cached_create(*a, **kw):
        kw["cache_dir"] = str(ROOT / "pretrained/cache/clip")
        return original_create(*a, **kw)

    CLIPAD.create_model_and_transforms = cached_create
    config.update(device="cuda:0", out_size_h=config["resolution"], out_size_w=config["resolution"])
    start = time.perf_counter()
    model = PromptAD(**config).to("cuda:0")
    checkpoint = run_dir / f"visa/k_{config['k_shot']}/checkpoint/SEG-Seed_{config['seed']}-{category}-check_point.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=False)
    model.eval_mode()
    score_maps, masks = [], []
    for data, mask, label, name, img_type in loader:
        transformed = torch.stack([model.transform(Image.fromarray(f.numpy())) for f in data]).to("cuda:0")
        for score in model(transformed, "seg"):
            score_maps.append(cv2.resize(score, (config["resolution"], config["resolution"]), interpolation=cv2.INTER_CUBIC))
        for m in mask:
            arr = m.numpy()
            arr[arr > 0] = 1
            masks.append(cv2.resize(arr, (config["resolution"], config["resolution"]), interpolation=cv2.INTER_NEAREST))
    scores = np.asarray(score_maps)
    pixel_auc = metric_cal_pix(scores, masks)["p_roc"]
    # PromptAD's supplied function uses the removed np.bool alias; restore it
    # locally at evaluation time without editing the official metric source.
    if "bool" not in np.__dict__:
        np.bool = bool
    aupro = float(cal_pro_metric(masks, scores, fpr_thresh=0.3, max_steps=200) * 100)
    result = {"P_AUROC": pixel_auc, "AUPRO": aupro, "evaluation_seconds": time.perf_counter() - start,
              "checkpoint": str(checkpoint), "official_function": "utils.metrics.cal_pro_metric",
              "steps": 200, "fpr_threshold": 0.3}
    (run_dir / "aupro.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
