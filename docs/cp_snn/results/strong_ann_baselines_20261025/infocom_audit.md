# InfoCom reproduction audit (2026-10-25)

Scope: read-only inspection of local official repository and supplied PDF. No remote jobs started, no official code modified. Local repository commit: `a3140d67405ac13948ad11801e1f0dd2c97ee126`, origin `https://github.com/fengxueguiren/InfoCom.git`.

## Primary evidence

- Supplied PDF: `C:/Workspace/InfoCom/InfoCom Kilobyte-Scale Communication-Efficient Collaborative Perception with Information Bottleneck.pdf`. Extracted text beside this file in `paper.txt`.
- Paper: https://arxiv.org/abs/2512.10305 ; project: https://weiquanmin.github.io/infocom/
- Official README checkpoint folder: https://drive.google.com/drive/folders/1CoZ5rN4hXlBg78G1NNVZ6BrbUpJFXh3j
- Supplemented annotations: https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/

## Paper Table 1: DAIR-V2X

All communication methods are based on CoAlign/PointPillars. These are paper-reported, not reproduced scores.

| Method | Reported volume | AP30 | AP50 | AP70 |
|---|---:|---:|---:|---:|
| No collaboration | 0 | .7278 | .6844 | .5665 |
| Late collaboration | 6.250 KB | .7993 | .6709 | .4708 |
| Standard CoAlign | 24.609 MB | .8294 | .7843 | .6353 |
| Where2comm | 2.462 MB | .8048 | .7539 | .6070 |
| ERMVP | .531 MB | .8217 | .7791 | .6324 |
| InfoCom | 5.922 KB | .8228 | .7789 | .6385 |

Paper calls task 3D detection. Official `opencood/utils/eval_utils.py` evaluates BEV polygons and globally sorts predictions by confidence; do not label reproduced result 3D-IoU AP.

## Data, annotation, and evaluator compatibility

Official DAIR YAML: `opencood/hypes_yaml/back/dairv2x/pointpillar_coalign.yaml`.

- Range: [-100.8,-40,-3.5,100.8,40,1.5], same nominal range as our paired SNN config.
- Voxel size [.4,.4,5], max points 32, max voxels train 32000/test 70000; resulting initial BEV [64,200,504]. Our SNN uses [.2,.2,.1], so these are distinct backbone families.
- Official root_dir=train.json, validate_dir=val.json, test_dir=val.json. Repository does not include those files. Exact ID identity must be checked on server. Our paired audit has 4678 train / 1738 val; paper does not enumerate exact IDs or sample counts.
- Dataset loads cooperative world annotations; camera/front paths use label_world_backup, lidar uses cooperative_label_path. It unconditionally also reads vehicle-side/label/lidar_backup, vehicle-side/label/lidar and infrastructure-side/label/virtuallidar, even for lidar-only inference.
- Official world-label projection does NOT filter object type; our PairedDAIRFusion filters Car/Van/Bus/Truck before projection. Inspect actual complemented label type distributions and compare per-frame GT boxes/counts (our 28622 val GT cannot be assumed identical).
- Official intermediate reader shuffles points also during eval and applies mask_ego_points to both agents. Our paired eval does not shuffle and removes ego points only on vehicle; additionally we shift road z datum before voxelization.
- Official communication range 100m may omit infrastructure for some pairs. Record effective CAV count vs our paired data.
- Both use score .20, NMS .15; official anchors feature_stride=2 vs our 8. Preserve official path first, then clearly label a harmonized-evaluation run.
- Inference overrides validate_dir with test_dir; training validation uses config validate_dir. Avoid accidentally evaluating a different file.

## Mechanism and wire accounting

Pipeline: PointPillars scatter -> [64,200,504] -> IB encoder -> latent mu [256] -> spatial mask [1,200,504], top 10%, 4-bit -> multiscale decoder -> reconstructed [64,200,504] -> CoAlign multiscale alignment/attention -> detector. At eval only mu is used, no stochastic latent noise. At training sample mu + std*eps and add KL after begin_epoch.

Paper volume = (D*32 + alpha*H*W*b)/8. DAIR gives 1024 + 5040 = 6064 bytes = 5.921875 KiB (paper labels KB). 24.609 MB similarly corresponds to MiB.

This formula accounts latent values and selected mask values but omits the selected positions' coding cost. Official code only returns dense dequantized float masks, not a serialized packet. Need design a lossless wire roundtrip and report both paper formula and real wire size. Count mask support bitmap/indices, dimensions/quantization metadata where needed, frame/pose metadata and framing. Do not assume missing index cost makes method useless; use a competitive codec and charge all methods consistently.

An illustrative bitmap+packed-4bit-selected-values format costs 12600+5040+1024=18664 bytes before headers/compression; this is NOT a measured optimum or paper result. A full 4-bit mask is 50400 bytes before compression. Actual zeros after quantization can reduce nonzero support.

`topk_mask` resolves threshold ties with torch.randperm; inference seeds NumPy only. Set/record Torch seed as well for reproducibility and inspect whether ties occur, rather than silently attributing score fluctuations to model quality.

## Computation and parameters

Paper has runtime plot on OPV2V/RTX3090, 20 trials, not DAIR end-to-end FLOPs. Appendix states ~.02s additional inference vs ERMVP ~.16s; no tabulated DAIR FLOPs or total parameter count. Cannot compare directly to our GPU latency or theoretical AC cost.

Locally instantiated official IBCommunication(channels=64,encoding_dim=256,decoding_dim=256,output_size=(200,504)) under existing local opencood Python. Parameter counts:

| Component | Parameters |
|---|---:|
| Communication module total | 106,952,427 |
| Decoder total | 104,540,904 |
| Decoder FC alone | 103,622,400 |
| Encoder | 2,161,136 |
| mu_net / std_net | 66,304 each |
| Mask adapt_layer_in / out | 117,760 / 19 |

Decoder FC expands 256 to 256*25*63=403200. Its weight+bias alone is ~414.5 MB FP32. These counts are source-instantiation measurements for the default config, not loaded checkpoint evidence or full detector counts. Full measurement still required after checkpoint/config verification. Potential SNN advantage: parameters/model memory and communication-module compute, but no speed/energy claim yet.

Official IBCommunication computes reconstruction for ego too and replaces ego with original feature afterward. Profiling should distinguish official execution from an optional semantically equivalent skip-ego optimization; do not win solely by leaving obvious opponent overhead unoptimized.

## Reproduction commands (after download and data-path configuration)

Run from dedicated InfoCom checkout, isolated environment, PYTHONPATH pointing only there. Do not install it editable into the currently running OpenCOOD environment: both use package name opencood.

```bash
cd /data1/zhh/gzc/InfoCom
export PYTHONPATH="$PWD"
PY=/path/to/isolated/infocom/bin/python
# Build Cython extension for the selected Python/NumPy ABI:
$PY opencood/utils/setup.py build_ext --inplace
# Verify strict coverage independently before official non-strict loader.
CUDA_VISIBLE_DEVICES=<gpu> $PY opencood/tools/inference.py --model_dir <copied-full-infocom-dair-dir> --fusion_method intermediate --save_npy --save_vis_interval 1000000
# Communication finetune from copied base model/config:
CUDA_VISIBLE_DEVICES=<gpu> $PY opencood/tools/train_infocom.py --model_dir <copied-base-dair-dir>
# Base training fallback:
CUDA_VISIBLE_DEVICES=<gpu> $PY opencood/tools/train.py --hypes_yaml <adapted-base-dair.yaml>
```

For base->InfoCom config use model.core_method `point_pillar_baseline_multiscale_ib`; keep rest of official model dimensions. Downloaded config takes precedence over generic README. README example ib_params.beta=.001, begin_epoch=15, while paper appendix beta=.01. Record this discrepancy; do not silently pick the best one. Generic base has batch4,30epochs,Adam .002,weight_decay1e-4,scheduler decay .1 at10,15. Paper describes 30 base +30 comm finetune. Actual train_infocom sets all parameters trainable (no frozen-base code), starts finetune epoch0, optimizer fresh, and uses task-only until epoch>begin_epoch.

Train validation uses shuffle=True/drop_last=True, checkpoint selection by validation loss, unlike our full AP curve selection. Final full eval drop_last=False. Do not describe these as matched training unless changed in a separately declared control.

## Environment and runtime risks

- README says provided weights depend on spconv1.2.1; paper appendix says spconv2.3.6, Torch1.10.1, Python3.7.16 and CUDA12.2. Treat as inconsistent metadata.
- This PointPillar model has no sparse-convolution weights. Preprocessor explicitly supports spconv1 VoxelGeneratorV2 and spconv2 Point2VoxelCPU3d. Existing spconv2 env likely reusable in an isolated clone; check voxel equivalence and loaded key shapes rather than doing unnecessary sparse-kernel transposition.
- Minimum runtime dependencies include Torch, NumPy, spconv/cumm, scipy, shapely, einops, timm (indirect fusion imports), icecream, h5py, open3d, opencv, pillow, pyquaternion, tqdm, pypcd, tensorboardX, yaml, Cython extension. Use installed-compatible packages and freeze resolved environment rather than blanket requirements upgrade.
- Repo ships cp37/cp38 Linux box_overlaps.so; rebuild for selected ABI. Standard PointPillar NMS is CPU polygon implementation; no need to build all pcdet CUDA extensions unless import smoke identifies an actual dependency.
- Official load_saved_model uses strict=False and ignores missing/unexpected keys. A missing communication module would still produce a deceptively successful load; record exact key coverage externally.
- train_infocom resumes model weights only, not optimizer/RNG; finetune resets epoch0. Do not use it for interruption recovery expecting exact resume.
- Training helper copies scripts into run dir, and deletes older best files. Copy original official weight+config into a new experiment directory before any train command. Preserve hash of original download.
- Latent decoder is large; full training needs memory admission and monitored first batches, not just free GPU inference memory.

## Immediate priorities

1. Download full DAIR CoAlign InfoCom weight+config; strict key/shape audit; preserve SHA256.
2. Run one-frame geometry/GT/finite-output smoke using dedicated checkout; compare effective IDs and GT to our paired manifest.
3. Full official evaluation and exported prediction replay, then harmonized evaluator.
4. Add real packet codec and measure mean/P95/max bytes, full model and communication-only latency/parameters/MACs under same hardware and batch1.
5. Only then choose bounded SNN improvements on observed Pareto gaps. Preserve scheme2 SNN core; do not resume scheme3 merely because InfoCom AP is higher.
