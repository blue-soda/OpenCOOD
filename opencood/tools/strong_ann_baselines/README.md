# Strong ANN reproduction harness (2026-10-25)

Research protocol: `docs/cp_snn/STRONG_ANN_BASELINES_20261025.md`.

`evaluate_cosdh.py` and `evaluate_infocom.py` accept explicit official checkout,
official model directory, data root, ordered JSON split, fresh output, GPU and
frame limit. Default limit=2 is a smoke; limit=0 evaluates the requested split.
They import the selected checkout, not the surrounding OpenCOOD package.
No training or checkpoint writes are performed. Official code is not modified.

InfoCom `--checkpoint-compat` explicitly permits ONLY the documented published
sigma/std rename and identity AdaptiveScale parameters. Raw mismatch and effective
strict coverage are saved. This is an inference compatibility hypothesis; it does
not prove the original training implementation. Omitting the flag fails on the
downloaded checkpoint, intentionally. Never replace this with strict=False.

The remaining scripts capture the specific 146 deployment paths. `prepare_remote`
restores missing original backup labels from the existing archive and reuses a
CUDA extension ONLY after checking all official source files byte-for-byte.
`audit_common_gt` exports the exact current paired-data GT and verifies two frames
through the actual dataset; `replay_common_gt` rescores unchanged predictions and
records GT differences. `audit_pairs` records participant range differences.

`launch_eval` checks GPU0 (CoSDH) / GPU2 (InfoCom) five times, requires >=8GB free
and <=30% utilization, starts a bounded 4h process, and rejects reused outputs.
`start_replays`/`finish_replays` wait for those two full evaluations and run CPU
replay once; inference failure is preserved, not retried. `collect_evidence`
captures metadata/status JSON, excluding point clouds and predictions.

Current runs: `/data1/zhh/gzc/diagnostics/ann_baselines_20261025/`:
`cosdh_common_full01`, `infocom_common_full01_compat`.
Current Python: `/data1/zhh/gzc/miniconda3/envs/ann_runtime/bin/python`.
Our GT replay uses original `opencood` environment in that envs directory.

Full split means all IDs supplied to that invocation; a custom two-frame split
is still only a smoke. Do not report its AP as validation AP. Official payload
estimates are NOT real wire bytes; codec measurement remains pending.

Update: `infocom_wire.py` implements exact latent+mask transport (bitmap/value
dictionary or dense FP32, zlib; 28-byte header plus separate 80-byte metadata).
`evaluate_infocom.py --wire-codec` substitutes decoded messages into the real
network, records per-frame packets and CPU cost. `launch_wire_stage.py` snapshots
the scripts on GPU3; `verify_wire_full.py` independently checks every file and
compares to the matched-participant no-wire run. All 1738 predictions matched
bitwise. CoSDH wire accounting is still pending; `audit_cosdh_support.py` checks
the latent dependency closure under its decoder convolutions and warp, without
claiming a complete packet or final detection parity.
