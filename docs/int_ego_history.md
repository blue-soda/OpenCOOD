# INT ego-history feature-memory port

Latest stage: all bounded runs and frozen-history ablations completed. Full
segment, multi-epoch training is now running; see [the frozen epoch protocol](int_epoch_protocol.md)
for launch records, stopping criteria and recovery validation.

## Scope

This implementation ports the public INT Concat and infinite-GRU feature-memory
equations into the main PointPillars processing chain. Source:
https://github.com/ADLab-AutoDrive/INT at
`988157ff131a0c027472bd0f00c0bda0e08cded0`.
The original fusion class is retained under `third_party/INT` for numerical tests.
The upstream LICENSE specifies Apache-2.0, despite its README's MIT statement.

This is **FM-only on DAIR ego LiDAR**, not a reproduction of complete INT or its
Waymo/nuScenes results. Point memory and prediction-map memory are disabled.
There is no SNN yet. A bounded training run establishes executable training,
checkpoint reload, causal state use, and evaluation; it cannot establish
convergence, a temporal accuracy benefit, or an SNN advantage.

## Data and state contract

- Existing DAIR train/val frame IDs assign whole scenes to splits; overlapping
  scenes fail preparation. Other scans from those scenes update state without
  labels. Unassigned scenes are excluded. Only vehicle-side labels are read.
- Dataset reference LiDAR-to-world poses are projected to SE(2), then inverse
  nearest-neighbor sampling aligns the cache in metric coordinates. This handles
  the non-square 201.6 m by 80 m BEV extent. Localization is not estimated here.
- One explicit cache per stream, batch size one; no future-frame access.
  Scene changes, missing/zero-byte PCDs, gaps above 0.25 s and pose jumps above
  `80 m/s * dt + 0.5 m` create independent segments. Shuffled training clips reset
  at their first frame. Repeated/noncausal timestamps without a reset fail.
- Cache is detached each frame, matching the public INT state path. Concat caches
  its output; GRU caches hidden features and age before readout. Age uses actual
  elapsed seconds rather than the upstream fixed 0.05 s.
- Memory operates after 64-channel Pillar scatter and before the spatial BEV
  backbone. At FP32, Concat uses 25,804,800 bytes and GRU 26,208,000 bytes per
  stream (200 x 504 cells). These are cache bytes, not total runtime allocation.
- Concat uses full-width pre-fusion branches and identity initialization to
  preserve the pretrained single-frame detector initially. This differs from
  upstream half-width configurations. GRU has randomly initialized new layers;
  short-run AP comparisons are initialization-dependent.
- Controlled first runs: seed 303, clip length 4, AdamW lr 1e-4 / decay 1e-4,
  BN running statistics frozen, all weights trainable, no augmentation,
  deterministic per-frame point shuffle, TF32 disabled, identical voxel caps.

## Server and reproducible commands

Edit and commit locally in `C:\Workspace\OpenCOOD\OpenCOOD` on `main`, push
`origin main`, then `git pull --ff-only origin main` in
`/data0/chen/gzc/workspace/OpenCOOD` on `chen@mindspore-184`.
Do not deploy source with scp or edit server source independently.
Other tasks share this repository; stage only files belonging to this port.

```bash
cd /data0/chen/gzc/workspace/OpenCOOD
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0
PY=/data0/chen/miniconda3/envs/opencood/bin/python
DATA=/data0/chen/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure
RUN=/data0/chen/gzc/workspace/diagnostics/int_ego_20260929
CKPT=/data0/chen/gzc/workspace/logs/logs/dairv2x/repro_dair_single_pointpillar_wide_for_codyntrust_cobevflow_2026_09_11_02_51_02/net_epoch_bestval_at57.pth

$PY -m unittest discover -s opencood/test -p test_int_feature_memory.py -v
# Preparation requires a new output directory and never overwrites evidence.
$PY -m opencood.tools.prepare_int_ego --data "$DATA" --output "$RUN/manifests_v2"
# For smoke use smoke_manifest.json and --train-steps 4.
# Run each mode on an available GPU and give it a new output directory.
CUDA_VISIBLE_DEVICES=1 $PY -u -m opencood.tools.run_int_ego \
  --data "$DATA" --checkpoint "$CKPT" \
  --manifest "$RUN/manifests_v2/sequence_manifest.json" \
  --mode concat --train-steps 256 --workers 4 --output "$RUN/bounded_concat"
```

Use modes `single`, `concat`, `gru` with the same manifest and training order.
The P1 checkpoint has 206 strictly matched spatial entries and SHA256
`c5fdef2fcc2914a09bb527cee6723730e70850056e3115f3197122e69431efb9`.
Preparation on 2026-09-29 produced train: 8,594 scans / 4,679 labeled frames /
52 scenes / 302 segments; val: 5,794 scans / 1,738 labeled frames /
22 scenes / 214 segments. Original train/val index counts were 4,811/1,789;
zero/missing PCDs are excluded and reset continuity, not replaced with empty
observations. See each frozen manifest's exclusions and boundaries.

## Evidence and acceptance

Seven tests cover upstream forward/backward/cache equivalence, real age updates,
input-state immutability, identity initialization, metric translation/rotation,
scene/gap/clip resets, causality, detachment, and learnable history gradients.
Real-data runs audit pretrained single/Concat identity, finite loss/gradients,
save `trained.pth`, reload it into a fresh model, and evaluate full causal val
streams. The first 16 eligible labels also compare prediction tensors with
history reset and alignment disabled. These diagnostics establish state
influence, not full-dataset accuracy ablations.

`manifest.json` records source commit, input hashes and arguments;
`train_frames.jsonl` and `eval_frames.jsonl` retain per-frame evidence.
Only `result.json` with `status=COMPLETED` confirms an entire run completed.
AP is existing OpenCOOD **BEV polygon IoU AP30/50/70**, not 3D AP.
Old P1 replay used a different GT convention and empty-PCD handling; its AP is
not directly comparable. Timing excludes IO/voxelization/postprocessing and
must not be called end-to-end latency or energy. GPU timings collected during
parallel experiments are diagnostic, not isolated performance benchmarks.

First smoke failures are retained: TF32 identity-output error exceeded 1e-3;
the first 64 val scans contained no labels. Subsequent code disables TF32 and
selects a causal smoke prefix through at least 32 labels and 64 scans.

Next scientific stage after engineering acceptance: freeze an ego-only protocol,
train ANN baselines to convergence with matched budgets and repeated seeds,
perform reset/no-alignment/history-length AP ablations, then introduce a spiking
state cell and compare accuracy, measured costs, and state/reset behavior.

## Execution snapshot, 2026-09-29

All seven tests passed. Both corrected smoke runs completed four supervised
updates, checkpoint reload, and 227 scans / 32 validation labels. Concat identity
head-output maximum error was 4.292e-6. Smoke GRU AP was zero after only four
updates; a successful execution is not evidence of a usable accuracy baseline.

Three full-stream runs started from main commit
`9bcf5f52fbbcbdaffc2511d865a6b89cb376e02e` with the same manifest SHA256
`772a212026870253e610facd3ae7ba9a9f2a72f03b67c34ed490c350e70f7420`.
All completed 256 updates / 311 training scans, with finite gradients.

| Mode | GPU | Launch PID | First/last 32 losses | Validation status at snapshot |
|---|---:|---:|---|---|
| single | 1 | 934744 | 0.532737 / 0.534695 | COMPLETED |
| concat | 2 | 934745 | 0.553243 / 0.549265 | COMPLETED |
| gru | 3 | 934746 | 2.489911 / 0.787703 | COMPLETED |

Outputs: `$RUN/bounded_{single,concat,gru}`; launch commands are recorded in
`$RUN/bounded_*_launch.json`, console logs in `$RUN/bounded_*.log`.
At the 2026-09-29 follow-up, all three results were verified: 5,794 scans,
1,738 labels, 22,971 GT boxes, identical train/eval frame orders and input
manifests, finite training loss/gradients, and matching checkpoint SHA256.
Machine-readable results and provenance are in `docs/int_ego_results/`.

| Mode | BEV AP30 (%) | BEV AP50 (%) | BEV AP70 (%) |
|---|---:|---:|---:|
| single | 83.8002 | 81.7607 | 72.2294 |
| concat | 83.6621 | 81.6135 | 71.8646 |
| gru | 51.8029 | 50.9101 | 43.5495 |

The engineering pipeline passes. These 256-update runs do not show an accuracy
benefit from temporal memory; Concat AP70 is 0.365 percentage points lower than
the matched single-frame run, and GRU needs substantial further training.
The trained Concat and GRU do react to history reset and pose alignment in all
16 diagnostic frames, but tensor differences alone cannot establish AP benefit.

The next check is full-validation, frozen-checkpoint ablation using
`--eval-only --checkpoint "$RUN/bounded_concat/trained.pth" --mode concat`
with `--history-policy reset` or `--history-policy no-align` (also repeat for
GRU). Reset discards history at every scan; no-align retains state without
pose warping. All other inputs and thresholds remain fixed. Give each run a
new output directory; non-default policies are rejected during training.
The thread heartbeat remains at 60 minutes to collect these ablations.

Launched on commit `d5625681d9bf82a140d2af0196d2d9cb3dc8dd7d`:

| Directory under `$RUN` | GPU | PID |
|---|---:|---:|
| ablation_concat_reset | 1 | 1006837 |
| ablation_concat_no_align | 2 | 1006838 |
| ablation_gru_reset | 3 | 1006839 |
| ablation_gru_no_align | 5 | 1006840 |

Exact launch records are in `docs/int_ego_results/ablation_*_launch.json`.
These use the existing trained checkpoints without further optimization.
On the next follow-up, verify all labels/GT, input/weight hashes, and reset/warp
logs before comparing AP. The completed bounded runs should not be restarted.
