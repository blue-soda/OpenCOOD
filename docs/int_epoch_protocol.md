# INT full-epoch baseline protocol (2026-09-29)

2026-09-30 06:18 CST: all three repaired full runs completed 23 epochs and met
the predefined validation plateau stopping condition. Each selected epoch 13.
Best BEV AP70: single 74.2956%, Concat 75.5047%, GRU 75.9669%. This is one seed
with validation selection, not a held-out/multi-seed result. See
`int_ego_results/full_training_summary.json`. Full-history ablations and temporal
checkpoint replays are now active under `full_best_*` (launch JSONs in the same
results directory). Training runs should not be relaunched.

Latest recovery: the first full runs failed their coverage guard because
`010585.pcd` is nonempty but corrupt. All train/val scans were subsequently
decoded; only that training frame was removed, with a new segment boundary.
Active runs are now `full_segments_v3_{single,concat,gru}`. See recovery below.

The 256-update run was only 256/4679 = 0.0547 supervised epochs. It was an
integration test, not a convergence baseline. Full-val AP70 (%) was:

| Model | Aligned history | Reset every scan | History without alignment |
|---|---:|---:|---:|
| Concat | 71.8646 | 71.8919 | 71.8761 |
| GRU | 43.5495 | 55.4158 | 38.6903 |

The matched single-frame short run scored 72.2294. These results do not show
temporal benefit. GRU's reset advantage motivates inspecting state training,
not claiming a result about converged GRUs or SNNs.

## Corrected state protocol

Old training reset every four scans, while validation kept state across full
valid segments. Training segments have length median 20, p95 86.95, max 210;
validation segments median 20, p95 71, max 110. The new sampler shuffles whole
segments per epoch, preserves chronological order within each segment, includes
every unlabeled warmup scan, and resets only at genuine segment boundaries.
This removes artificial four-scan truncation. State remains detached per scan,
as in the upstream INT path; this is not full backpropagation through time.
The previous weights are not reused: all three new runs start from the same
original P1 spatial checkpoint, with the existing Concat/GRU initialization.

After the full decode audit, an epoch is exactly 8,593 valid train scans,
including 4,678 supervised updates (the previous counts were 8,594 / 4,679).
Each epoch is followed by the same full validation stream: 5,794 scans and
1,738 labeled frames. Missing decoded samples fail the coverage check instead
of silently changing the experimental population.

## Training and stopping rule

- Modes single/concat/gru, seed 303; whole-segment ordering seed 303 + epoch.
- Batch one, all parameters trained, BN running statistics frozen; no data
  augmentation in this controlled phase; same point ordering and voxel caps.
- AdamW lr 1e-4, decay 1e-4, gradient norm cap 10; TF32 off, deterministic cuDNN.
- Validation AP70 controls ReduceLROnPlateau: absolute threshold 0.001 AP
  (0.1 percentage point), patience 2, factor 0.5, minimum lr 1e-6.
- Initial allocation: at most 40 epochs. Operational plateau requires at least
  20 epochs, 10 epochs without a >0.001 improvement over the significant best,
  and lr <= 1.25e-5. This heuristic is recorded before running; it is not a
  mathematical guarantee of convergence.
- `PLATEAU_REACHED` and `EPOCH_LIMIT_REACHED` are distinct terminal states.
  If curves still improve at the limit, review and extend the allocation with
  `--resume --max-epochs N`; do not relabel the limit as convergence.
- Because stopping/LR depend on validation, this is model development data,
  not an untouched test set. No multi-seed or paper-quality conclusion yet.

## Reproducible execution

Use `opencood.tools.run_int_epochs` in the same environment documented in
`int_ego_history.md`, with `--mode single|concat|gru`, original P1 `--checkpoint`,
`--manifest .../manifests_v3_decoded/sequence_manifest.json`, and a new `--output`.
Use `--max-epochs 40 --min-epochs 20 --patience 10 --workers 4`.

Each `epoch_NNN` retains frame logs, train summary, full eval summary and a
model-only `trained.pth`. `curve.json` records losses, LR, AP and cost per epoch;
`best.json` identifies an immutable epoch checkpoint. `latest.pth` is an atomic
epoch-boundary recovery snapshot including optimizer, scheduler, counters and
contract. It resumes from the next epoch; interrupted partial-epoch evidence is
renamed and retained. Dataset/config/initialization hashes and protocol options
are checked before resume. The seed resets deterministically at each epoch.

Three 40-epoch runs retain approximately 15 GB of checkpoints plus logs; each
epoch checks a 5 GiB free-space reserve. No shared environment packages change.
All source changes are local main commits, pushed and then pulled on the server.
The existing 60-minute heartbeat must inspect curve/latest/result and process
logs. It should check the first full epoch for 8,593 scans / 4,678 updates,
validation GT equality and finite gradients, then track plateau/LR behavior.
After training, repeat the frozen-checkpoint history ablations on selected
epochs before introducing any SNN component.

## Launch and recovery validation

Nine unit tests passed. The GRU smoke comparison of two continuous epochs versus
one epoch plus process restart produced identical model tensors (maximum absolute
difference 0), optimizer state, scheduler state and validation AP. See
`int_ego_results/epoch_resume_audit.json`. Scope: 81 training scans / 32 supervised
updates per epoch and 227 validation scans / 32 labels. This checks recovery,
not full-dataset convergence.

Full runs launched from main `0ea0287dcd10f92ef92fe3ecefc80cd0f0043b1d`:

| Output below diagnostics/int_ego_20260929 | GPU | PID |
|---|---:|---:|
| full_segments_single | 1 | 1051951 |
| full_segments_concat | 2 | 1051952 |
| full_segments_gru | 3 | 1051953 |

Exact commands are in `int_ego_results/full_segments_*_launch.json`.
These original full runs failed at epoch 1 before saving a checkpoint. Bounded runs and their
four frozen-history ablations have already completed and must not be relaunched.

## Decode failure and recovery

All three original full runs encountered the same corrupt frame `010585`,
whose 627,243-byte compressed PCD decodes empty (SHA256
`5cf410e75d7bfbc46f06e2a69c4d6ae193e737a82412f9013272f485693657d0`).
The nonzero-size check in the original manifest was insufficient. The coverage
guard rejected the 4,678/4,679 updates; no checkpoint was saved. Failure logs and
machine-readable FAILED records remain in the original directories.

`audit_int_stream` decoded all 14,388 candidate scans using the training reader,
excluding only this frame and splitting memory continuity at its position.
Validation rows are exactly unchanged. The new manifest SHA256 is
`18886ad3a55df76621b61cea67e1d352203e8cdf4cd8c36fec36ff72fe27951a`.
Training has 303 segments, 8,593 scans, 4,678 labels. Decode JSONL records are
under `manifests_v3_decoded`; summary evidence is `int_ego_results/decode_repair_audit.json`.
Eleven unit tests pass, including bad-frame state separation and unchanged valid
streams. Any newly invalid frozen training frame now raises immediately.

Restarted from the original P1 initialization on main
`917cdeac211ed88a0d68e1bb22578672ded4c30c`:

| Current directory | GPU | PID |
|---|---:|---:|
| full_segments_v3_single | 1 | 1077525 |
| full_segments_v3_concat | 2 | 1077526 |
| full_segments_v3_gru | 3 | 1077527 |

All other training/stopping options are unchanged. Use these v3 runs for the next
heartbeat; inspect `failure.json` as well as logs and process liveness. Runtime
exceptions now write failure metadata. Each run also saves `effective_runtime.json`
to distinguish executed settings from unused legacy YAML fields: batch 1,
AdamW, no artificial delay or asynchronous fusion, no augmentation, full segments.

## Full training acceptance, 2026-09-30

All 69 completed epochs have the required 8,593 scans / 4,678 updates and
5,794 validation scans / 1,738 labels / 22,971 GT. Losses and AP are finite.
Each mode completed 107,594 optimizer updates; selected immutable checkpoint
hashes match the saved best records. Later train loss decreased while validation
did not improve; report validation-plateau stopping rather than claiming loss
convergence. Single-frame checkpoint replay matched AP30/50/70 within 2.8e-8
absolute AP. New checks hold each temporal checkpoint fixed and compare aligned,
reset-every-scan and no-alignment policies. These comparisons remain pending.

| Active evaluation | GPU | PID |
|---|---:|---:|
| full_best_concat_aligned | 1 | 1675126 |
| full_best_concat_reset | 2 | 1675127 |
| full_best_concat_no_align | 3 | 1675128 |
| full_best_gru_aligned | 5 | 1675129 |
| full_best_gru_reset | 6 | 1675130 |
| full_best_gru_no_align | 7 | 1675131 |

Next heartbeat: verify these six results, fixed-weight/manifest hashes, frame
counts, reset/warp behavior and aligned replay agreement. Then summarize actual
history benefit separately from the difference between separately trained models.
