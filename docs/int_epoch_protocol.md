# INT full-epoch baseline protocol (2026-09-29)

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

An epoch is exactly 8,594 valid train scans, including 4,679 supervised updates.
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
`--manifest .../manifests_v2/sequence_manifest.json`, and a new `--output`.
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
logs. It should check the first full epoch for 8,594 scans / 4,679 updates,
validation GT equality and finite gradients, then track plateau/LR behavior.
After training, repeat the frozen-checkpoint history ablations on selected
epochs before introducing any SNN component.
