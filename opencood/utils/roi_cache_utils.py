import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Iterable, Optional

import torch


def roi_cache_enabled(cache_args: Optional[Dict[str, Any]]) -> bool:
    return bool(cache_args and cache_args.get("enabled", False))


def roi_cache_mode(cache_args: Optional[Dict[str, Any]]) -> str:
    if not cache_args:
        return "off"
    return str(cache_args.get("mode", "readwrite")).lower()


def tensor_digest(tensor: torch.Tensor, max_elements: int = 4096) -> str:
    """Create a small deterministic digest for cache safety checks."""
    if tensor is None:
        return "none"
    with torch.no_grad():
        flat = tensor.detach().reshape(-1)
        if flat.numel() > max_elements:
            step = max(flat.numel() // max_elements, 1)
            flat = flat[::step][:max_elements]
        flat = flat.to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha1(flat.numpy().tobytes()).hexdigest()[:12]


def value_digest(values: Iterable[Any]) -> str:
    packed = json.dumps(list(values), sort_keys=True, default=str)
    return hashlib.sha1(packed.encode("utf-8")).hexdigest()[:12]


def make_roi_cache_key(
        sample_idx: Any,
        cav_idx: int,
        split: str,
        k: int,
        threshold: Optional[float],
        num_roi_thres: int,
        time_diff: Optional[torch.Tensor] = None,
        prediction_hash: Optional[str] = None) -> str:
    if torch.is_tensor(sample_idx):
        sample_idx = int(sample_idx.detach().cpu().reshape(-1)[0].item())
    if isinstance(sample_idx, (list, tuple)):
        sample_idx = sample_idx[0]
    time_values = []
    if time_diff is not None:
        time_values = [
            round(float(x), 4)
            for x in time_diff.detach().cpu().reshape(-1).tolist()
        ]
    threshold_tag = "none" if threshold is None else f"{float(threshold):.4f}"
    pred_tag = prediction_hash or "nopred"
    key_payload = [
        split, sample_idx, int(cav_idx), int(k), threshold_tag,
        int(num_roi_thres), time_values, pred_tag
    ]
    return value_digest(key_payload)


def _to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    return obj


class RoiBoxCache:
    def __init__(self, cache_args: Optional[Dict[str, Any]]):
        self.enabled = roi_cache_enabled(cache_args)
        self.mode = roi_cache_mode(cache_args)
        self.root_dir = None
        self.strict_prediction_hash = True
        self.namespace = "default"
        self.log_interval = 200
        self._hits = 0
        self._misses = 0
        self._writes = 0
        if self.enabled:
            self.root_dir = cache_args.get("root_dir", None)
            if not self.root_dir:
                raise ValueError("roi_cache.enabled requires roi_cache.root_dir")
            self.namespace = str(cache_args.get("namespace", "default"))
            self.strict_prediction_hash = bool(
                cache_args.get("strict_prediction_hash", True))
            self.log_interval = int(cache_args.get("log_interval", 200))
            os.makedirs(self.root_dir, exist_ok=True)

    def wants_prediction_hash(self) -> bool:
        return self.enabled and self.strict_prediction_hash

    def path_for(self, split: str, key: str) -> str:
        directory = os.path.join(self.root_dir, self.namespace, split)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{key}.pth")

    def load(self, split: str, key: str, device: torch.device) -> Optional[Any]:
        if not self.enabled or self.mode in ("off", "write"):
            return None
        path = self.path_for(split, key)
        if not os.path.exists(path):
            self._misses += 1
            return None
        try:
            payload = torch.load(path, map_location="cpu")
            self._hits += 1
            return _to_device(payload["box_results"], device)
        except Exception as exc:
            self._misses += 1
            print(f"[roi_cache] failed to read {path}: {exc}")
            return None

    def save(self, split: str, key: str, box_results: Any,
             meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self.mode in ("off", "read"):
            return
        path = self.path_for(split, key)
        if os.path.exists(path):
            return
        payload = {
            "meta": meta or {},
            "box_results": _to_cpu(box_results),
        }
        directory = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_roi_", suffix=".pth", dir=directory)
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            self._writes += 1
            if self.log_interval > 0 and self._writes % self.log_interval == 0:
                print("[roi_cache] writes={}, hits={}, misses={}, root={}".format(
                    self._writes, self._hits, self._misses, self.root_dir))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

