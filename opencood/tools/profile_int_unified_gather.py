"""Paired synthetic screen of gathered V2 updates, including index/scatter cost."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_unified_bev import ResidualStateUpdate
from opencood.models.sub_modules.int_unified_gather import gathered_update
from opencood.models.sub_modules.int_cache_repair import expand_tiles, refresh_tiles
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--repeats', type=int, default=30)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    try: run(a, out)
    except Exception as exc:
        write_json(out/'status.json', dict(status='FAILED', error=repr(exc)))
        raise


def run(a, out):
    torch.manual_seed(303)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = ResidualStateUpdate().cuda().eval()
    sha = None
    if a.checkpoint:
        sha = digest(Path(a.checkpoint))
        checkpoint = torch.load(a.checkpoint, map_location='cpu')
        assert checkpoint['contract']['version']=='unified_bev_dense_v2', 'Microbenchmark supports original V2 only'
        saved = checkpoint['model']
        prefix = 'core.update.'
        module.load_state_dict({k[len(prefix):]: v for k, v in saved.items() if k.startswith(prefix)}, strict=True)
    x = torch.randn(1, 513, 100, 252, device='cuda'); x[:, -1:] = 1
    old = x[:, 256:512]; records = []
    with torch.no_grad():
        dense = module(x)
        for tile in [8, 16, 32]:
            h, w = (100 + tile - 1)//tile, (252 + tile - 1)//tile
            for pattern in ['cluster', 'scattered']:
                order = torch.randperm(h*w, device='cuda') if pattern == 'scattered' else torch.arange(h*w, device='cuda')
                for fraction in [.1, .25, .5]:
                    active = torch.zeros(1, 1, h, w, device='cuda', dtype=torch.bool)
                    active.view(-1)[order[:max(1, round(h*w*fraction))]] = True
                    mask = expand_tiles(active, tile, (100, 252))
                    expected = torch.where(mask, dense, old)
                    actual = gathered_update(module, x, mask)
                    error = float((actual-expected).abs().max())
                    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4), error
                    row = dict(tile=tile, pattern=pattern, requested_tile_fraction=fraction,
                               selected_cell_fraction=float(mask.float().mean()), max_abs_error=error, timings={})
                    for mode in ['dense_before', 'masked', 'crop', 'gather', 'dense_after']:
                        def execute():
                            if mode.startswith('dense'): return module(x)
                            if mode == 'masked': return torch.where(mask, module(x), old)
                            if mode == 'crop': return refresh_tiles(module, x, old, mask, 1, 2, tile)[0]
                            return gathered_update(module, x, mask)
                        for _ in range(5): execute()
                        times = []
                        torch.cuda.reset_peak_memory_stats()
                        for _ in range(a.repeats):
                            torch.cuda.synchronize(); tick = time.perf_counter()
                            execute(); torch.cuda.synchronize()
                            times.append(1000*(time.perf_counter()-tick))
                        row['timings'][mode] = dict(mean_ms=float(np.mean(times)), p95_ms=float(np.percentile(times, 95)),
                                                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    records.append(row)
                    write_json(out/'status.json', dict(status='RUNNING', completed_cases=len(records)))
                    print(row, flush=True)
    write_json(out/'report.json', dict(status='COMPLETED', seed=303, repeats=a.repeats,
        checkpoint=a.checkpoint, checkpoint_sha256=sha, device=torch.cuda.get_device_name(), records=records,
        scope='Update only, synthetic packed input/masks, same weights and GPU sequential comparisons.',
        caveats=['Includes dynamic indices, layout conversions, halo dependencies and scatter.',
                 'Excludes observation, context, selection, pose warp and head; no AP evidence.',
                 'Gather is opt-in diagnostic only, no running training or inference policy changed.']))
    write_json(out/'status.json', dict(status='COMPLETED'))


if __name__ == '__main__': main()
