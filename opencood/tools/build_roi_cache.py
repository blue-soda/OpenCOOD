# -*- coding: utf-8 -*-
"""Precompute two-stage ROI boxes for frozen single-branch experiments."""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Build ROI proposal cache")
    parser.add_argument("--hypes_yaml", required=True,
                        help="Config file used by the target experiment.")
    parser.add_argument("--model_dir", default="",
                        help="Optional checkpoint folder to load before caching.")
    parser.add_argument("--split", choices=["train", "val", "both"],
                        default="both")
    parser.add_argument("--cache_root", required=True,
                        help="Directory where ROI cache files are written.")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_strict_prediction_hash", action="store_true",
                        help="Use sample/delay keys only. Safe for deterministic eval; "
                             "avoid for randomly augmented training.")
    parser.add_argument("--seed", type=int, default=303)
    return parser.parse_args()


def configure_cache(hypes, opt):
    hypes.setdefault("model", {}).setdefault("args", {})
    hypes["model"]["args"]["roi_cache"] = {
        "enabled": True,
        "mode": "write",
        "root_dir": opt.cache_root,
        "namespace": opt.namespace,
        "strict_prediction_hash": not opt.no_strict_prediction_hash,
        "log_interval": 200,
    }


def set_split_paths(hypes, split):
    if split == "val":
        hypes["validate_dir"] = hypes.get("test_dir", hypes["validate_dir"])


def build_for_split(hypes, model, split, opt, device):
    train = split == "train"
    set_split_paths(hypes, split)
    dataset = build_dataset(hypes, visualize=False, train=train)
    collate_fn = dataset.collate_batch_train if train else dataset.collate_batch_test
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False,
        generator=torch.Generator().manual_seed(opt.seed),
        drop_last=False)
    seen = 0
    skipped = 0
    start = time.time()
    for batch_data in tqdm(loader, desc=f"roi-cache-{split}"):
        if opt.limit > 0 and seen >= opt.limit:
            break
        if batch_data is None:
            skipped += 1
            continue
        batch_data = train_utils.to_device(batch_data["ego"], device)
        with torch.no_grad():
            model(batch_data, dataset=dataset)
        seen += 1
    print("[roi_cache] split={} seen={} skipped={} elapsed_sec={:.1f}".format(
        split, seen, skipped, time.time() - start))


def load_single_pretrain_if_needed(hypes, model):
    pretrain_cfg = hypes.get("is_single_pre_trained", {})
    if not pretrain_cfg.get("pre_train_flag", False):
        return
    pretrain_path = pretrain_cfg["pre_train_path"]
    epoch = int(pretrain_cfg["pre_train_epoch"])
    checkpoint = os.path.join(pretrain_path, f"net_epoch{epoch}.pth")
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print(f"[roi_cache] loaded single pretrain: {checkpoint}")


def main():
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    configure_cache(hypes, opt)
    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes)
    if not opt.model_dir:
        load_single_pretrain_if_needed(hypes, model)
    model.to(device)
    if opt.model_dir:
        _, model = train_utils.load_saved_model_diff(opt.model_dir, model)
    model.eval()
    splits = ["train", "val"] if opt.split == "both" else [opt.split]
    for split in splits:
        build_for_split(hypes, model, split, opt, device)


if __name__ == "__main__":
    main()
