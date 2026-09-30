# INT real-frame spike memory: stage 1

## Material Passport

- Task: ego-history INT FM replacement, authorized 2026-09-30.
- Status: 21 semantic tests and real-data engineering/restart gate passed; full
  lif/leaky/gru training launched on 2026-09-30. No SNN accuracy claim yet.
- Data: existing DAIR vehicle-only decoded v3 manifest; no infrastructure scans, artificial latency or asynchronous training.
- Scope: mixed ANN spatial detector + SNN FM. Not a full spiking detector or a reproduction of full INT PC/FM/PM.

## Fixed comparison

Compare `lif`, `leaky`, and a newly trained `gru`, seed 303, identical original P1
initialization, complete segment sampling, geometry, loss, preprocessing and
stopping criteria. Existing per-frame-detach ANN results remain historical
baselines; the new GRU receives the same TBPTT and warmup as the two new models.

`lif`: input 3x3 convolution, aligned membrane U, real-delta-t exponential decay,
binary threshold event, differentiable subtractive reset, aligned exponentially
filtered spike trace R, linear 1x1 readout into the original BEV backbone.
The detector reads R only; no raw X or membrane bypass. Each real scan is exactly
one update. Surrogate derivative is `1/(1+(pi*x)^2)` for normalized voltage minus
one; forward events are exactly binary. Negative currents can inhibit membrane.

`leaky`: identical trainable layers, time constants, gain initialization and state
dimensions, but emission is ReLU(V/threshold), U=V, without binary threshold or
reset. Here the positive threshold parameter is a continuous gain denominator.
This control jointly removes threshold quantization/reset; further ablation is
needed to separate their contributions if a benefit is found.

Both use C=32 for U and R: 64 continuous cached channels, 25,804,800 FP32 bytes
at 200x504. GRU caches 65 channels (including age), 26,208,000 bytes. Metadata and
transient training activations are additional. This approximately matches cache
budget, not parameters/FLOPs; also report trainable parameters and latency.

Time constants start at tau_u=.2s, tau_r=.1s and threshold=1, positive via softplus
with .001 floor; input convolution starts at .25 times a partial identity,
readout repeats channel identity to fill 64 output channels. They are learnable.
No firing-rate regularization, teacher distillation or extra losses in stage 1.
Inspect all-grid and observed-cell emission rates, membrane/trace magnitude and
temporal gradients before full training. Geometry validity does not imply a
LiDAR free-space measurement. Dynamic object motion is not compensated.

## Training

Dedicated config: `opencood/hypes_yaml/dair-v2x/snn/int_ego_memory.yaml`.
Dedicated runner: `python -m opencood.tools.run_int_epochs`.
Flags: `--mode {lif,leaky,gru} --tbptt-steps 4 --fm-only-epochs 1 --seed 303
--lr 0.0001 --workers 4 --min-epochs 20 --max-epochs 40 --patience 10 --min-delta 0.001`.

Epoch 1 freezes spatial parameters to adapt each FM; epoch 2 onward trains all
parameters. BN running statistics remain frozen. Each TBPTT window includes up
to four real scans within one segment. Unlabelled context retains its graph;
labelled losses are averaged per window, then one optimizer step occurs. No
optimizer update is allowed while its window is still being built. A context-only
window causes no optimizer update. Detach at window boundaries retains numerical
state; only physical segment/gap boundaries reset it. Frame coverage and optimizer
updates are counted separately. Four frames limit gradient horizon, not memory.

Full validation uses complete causal streams, without gradients. Same validation
plateau rule as the ANN phase: >=20 epochs, 10 epochs without >.001 AP70 gain, LR
<=initial/8; maximum40. A maximum-epoch stop is not convergence. Single seed and
validation-selected results are exploratory, not held-out/multi-seed evidence.
Promising results require further seeds/independent evaluation and frozen-checkpoint
reset/no-align ablations. No dense-GPU efficiency claim from firing rate alone.

## Engineering gate

1. Preserve the existing upstream INT parity/geometry tests.
2. Verify binary forward, soft-reset residual, trace decay without new events,
   real-time constants, both-state warping, missing-region reset and causality.
3. Verify a future labelled loss reaches an unlabelled earlier input, stops across
   detach, and ANN controls have the same cross-frame capability.
4. On real data, run two engineering epochs on two complete train/val segments;
   include both FM-only and joint learning. No pilot AP enters the final comparison.
5. Compare uninterrupted two-epoch training with epoch-1 restart for exact model,
   optimizer and scheduler equality. Preserve failed trials; fix locally and push.
6. Only then launch the full frozen-manifest comparison in fresh directories.

All code changes: local main commit/push, server pull. Preserve concurrent research
changes. Runtime snapshots include source/config/data hashes, effective arguments,
checkpoint hashes, full frame logs and emission/state diagnostics.

## Execution evidence

The engineering pilot covered32 train scans/20 labels per epoch, seven optimizer
updates per epoch, and two complete independent validation segments. Both epochs
completed for all three modes. LIF had1.44%-1.51% whole-grid emissions; temporal
gradients were finite/nonzero. Spatial parameters stayed exactly frozen in epoch1
and changed in epoch2. Restarting LIF after epoch1 reproduced epoch2 model,
optimizer, scheduler and AP exactly. See `int_spike_results/engineering_gate.json`.

Full training was launched from code commit `e85e20d` using the original P1
checkpoint (not the pilot weights), on the frozen decoded full manifest. Complete
commands/PIDs/GPUs are recorded in `int_spike_results/launch_full.json`. Server
root: `/data0/chen/gzc/workspace/diagnostics/int_spike_20260930`. All three runners
write their own `full_{mode}/curve.json`, `best.json`, `result.json` and per-frame
logs. Inspect those and process commands before any restart; do not duplicate jobs.

Current model-step timings include diagnostic reductions/synchronizations. A
formal latency comparison must disable diagnostics uniformly and profile anew;
these initial timings and emission rates do not establish energy efficiency.
