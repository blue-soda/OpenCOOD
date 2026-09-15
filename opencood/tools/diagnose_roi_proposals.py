import argparse
import os
import sys
import time

import torch

sys.path.append(os.getcwd())

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize ROI proposal diagnostics for two-stage models.")
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--match_distance", type=float, default=4.0)
    return parser.parse_args()


def scalar(output_dict, key):
    value = output_dict.get(key)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    return 0.0


def main():
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml)
    hypes["train_params"]["batch_size"] = 1
    hypes.setdefault("model", {}).setdefault("args", {}).setdefault(
        "diagnostics", {})
    hypes["model"]["args"]["diagnostics"]["roi_box_stats"] = True
    hypes["model"]["args"]["diagnostics"]["roi_match_distance"] = \
        opt.match_distance

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(hypes, visualize=False, train=opt.split == "train")
    model = train_utils.create_model(hypes)
    state_dict = torch.load(opt.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    stats = {
        "n": 0,
        "gt_nonempty": 0,
        "sample_gt_nonempty_single_roi_empty": 0,
        "sample_gt_nonempty_final_empty": 0,
        "single_roi_zero": 0,
        "single_roi_sum": 0.0,
        "gt_sum": 0.0,
        "single_recall_sum": 0.0,
        "single_precision_sum": 0.0,
        "final_recall_sum": 0.0,
        "final_precision_sum": 0.0,
        "single_raw_anchor_sum": 0.0,
        "single_max_score_sum": 0.0,
        "final_boxes_sum": 0.0,
    }

    prefix = "ectra" if "ectra" in hypes["model"]["core_method"] \
        else "cobevflow"
    start = time.time()
    max_items = min(opt.limit, len(dataset))
    for idx in range(max_items):
        item = dataset[idx]
        if item is None:
            continue
        batch = dataset.collate_batch_train([item])
        batch = train_utils.to_device(batch["ego"], device)
        with torch.no_grad():
            output = model(batch, dataset=dataset)

        gt = scalar(output, f"{prefix}_diag_gt_centered_roi_count_mean")
        single = scalar(output, f"{prefix}_diag_single_roi_boxes_mean")
        final = scalar(output, f"{prefix}_diag_final_fused_boxes_mean")
        stats["n"] += 1
        stats["gt_sum"] += gt
        stats["single_roi_sum"] += single
        stats["single_recall_sum"] += scalar(
            output, f"{prefix}_diag_single_roi_gt_recall")
        stats["single_precision_sum"] += scalar(
            output, f"{prefix}_diag_single_roi_pred_precision")
        stats["final_recall_sum"] += scalar(
            output, f"{prefix}_diag_final_fused_gt_recall")
        stats["final_precision_sum"] += scalar(
            output, f"{prefix}_diag_final_fused_pred_precision")
        stats["single_raw_anchor_sum"] += scalar(
            output, f"{prefix}_diag_single_roi_raw_anchor_mean")
        stats["single_max_score_sum"] += scalar(
            output, f"{prefix}_diag_single_roi_max_score_mean")
        stats["final_boxes_sum"] += final
        if gt > 0:
            stats["gt_nonempty"] += 1
            if single == 0:
                stats["sample_gt_nonempty_single_roi_empty"] += 1
            if final == 0:
                stats["sample_gt_nonempty_final_empty"] += 1
        if single == 0:
            stats["single_roi_zero"] += 1

        if (idx + 1) % 50 == 0:
            print("progress", idx + 1, "elapsed",
                  round(time.time() - start, 1), stats, flush=True)

    print("FINAL", stats)
    if stats["n"] == 0:
        return
    print("sample_empty_given_gt",
          stats["sample_gt_nonempty_single_roi_empty"] /
          max(stats["gt_nonempty"], 1))
    print("sample_single_zero", stats["single_roi_zero"] / stats["n"])
    for key in [
            "gt_sum",
            "single_roi_sum",
            "single_recall_sum",
            "single_precision_sum",
            "final_recall_sum",
            "final_precision_sum",
            "single_raw_anchor_sum",
            "single_max_score_sum",
            "final_boxes_sum"]:
        print("mean_" + key.replace("_sum", ""), stats[key] / stats["n"])


if __name__ == "__main__":
    main()
