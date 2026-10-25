# CoSDH reproduction audit — 2026-10-25

Read-only audit of official repository `C:/Workspace/CoSDH`, commit `b270b69fe133c0c1ca2044ad072cd77fc21863fb`. No shared OpenCOOD edits, training, or server mutations performed by this audit.

## Sources and exact paper results

- Official repository: https://github.com/Xu2729/CoSDH
- Official paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_CoSDH_Communication-Efficient_Collaborative_Perception_via_Supply-Demand_Awareness_and_Intermediate-Late_Hybridization_CVPR_2025_paper.pdf
- Local PDF text extracted to `paper.txt` with pdftotext -layout. Web PDF returned 403; paper numbers below read from supplied local PDF.
- Table 1, printed page 6839, PDF page 6:

| DAIR-V2X row | AP50 % | AP70 % | BD, paper label Mbps |
|---|---:|---:|---:|
| No fusion |66.51|55.46|0|
| Early fusion |74.52|59.22|50.2|
| Late fusion |67.86|50.47|0.2|
| CoAlign unrestricted |78.06|63.09|1968.8|
| Where2comm unrestricted |76.13|60.16|172.3|
| CoSDH unrestricted |76.75|63.85|7.1|
| CoAlign <=6.75Mbps |75.26|60.19|3.8|
| Where2comm <=6.75Mbps |74.98|59.51|5.3|
| CoSDH <=6.75Mbps |76.47|63.76|1.4|

Paper BD is mean bandwidth per collaborating agent at 10Hz. 1.4/7.1 decimal Mbps would imply 17.5/88.75 kB per frame. **Do not treat this conversion as exact:** repository CommRecorder divides by 1024^2 while labeling Mb/s; under that implementation values imply 18.35008/93.06112 kB per frame. No published executable accounting path for this model establishes which convention generated Table1. Paper values are rounded. Paper has no measured compute/FLOP/latency main-table result located in the supplied text.

Paper calls task 3D object detection, but official evaluator computes **BEV polygon IoU** using xy of first four corners (`common_utils.convert_format:234`), then globally score-sorts TP/FP in `eval_utils.calculate_ap`. It is not volumetric 3D AP. This matches the user's requested BEV AP concept, subject to labels/splits.

## Mechanism

PointPillars → multiscale BEV features (64/128/256 channels) → ego demand mask from sparse point density (<4 points; paper density 4/32) and collaborator supply mask from smoothed foreground confidence → each scale compressed by 16x autoencoder → FP16 intermediate feature → geometry-aligned max fusion → detection. Hybrid late branch additionally sends confidence-filtered roadside predictions; discard confidence<0.3, multiply roadside score by0.8 before joint NMS. This branch can retain recall when intermediate communication is very limited.

For configured DAIR grid x504,y200, multiscale maps are approximately [64,100,252], [128,50,126], [256,25,63], with latent channels4/8/16. Dense FP16 latent values alone total352800 bytes before spatial selection, metadata, demand mask, and late predictions. This is derived shape arithmetic, not measured payload.

## Official configuration / commands

`opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_cosdh.yaml`:
- train.json / val.json / val.json. Paper text does not give exact split IDs or counts.
- range[-100.8,-40,-3.5,100.8,40,1.5], pillar[.4,.4,5],32 points; max voxels train32000/test70000.
- max_cav5 (DAIR actual pair2), no injected noise by default, comm_range100, proj_first false.
- 30epochs in YAML vs **40 in paper**, Adam .002, decay1e-4, LR milestones10/20, batch4 per process. Do not silently change or conflate.
- class score threshold.2; anchor4.5x2x1.56, feature stride2; NMS.15; supply threshold.01.
- inference seed303. Loader pretrained is strict=False; independently require empty missing/unexpected keys.

Run from official repo with PYTHONPATH explicitly set to that repo, avoiding collision with our package also named opencood:
```
PYTHONPATH=$PWD python opencood/tools/inference.py --model_dir /path/to/dairv2x_cosdh --fusion_method intermediatelate --save_vis_interval 1000000 --save_npy
PYTHONPATH=$PWD python opencood/tools/inference.py --model_dir /path/to/dairv2x_cosdh --fusion_method intermediate --save_vis_interval 1000000 --save_npy
PYTHONPATH=$PWD torchrun --master_port 22334 --nproc_per_node=4 opencood/tools/train_ddp.py -y opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_cosdh.yaml
```
Inference saves boxes/GT but not scores with stock save_npy; unified offline evaluator requires add score/frame-ID export. Minimal one-GPU `python .../train_ddp.py -y ...` supported, but changes effective global batch relative4GPU official training; record adaptation.

## Weights and dependencies

Official Google Drive root https://drive.google.com/drive/folders/1T3LLCn257Gynoqmm_HeXJu3Q8PBYGHfL . Read-only HTTP access succeeded. DAIR child folder observed ID `1smPONv2qCjlBCfOwco_eNNokoIDw04Ww`. Parent agent handles downloads; no download attempted here. Exact checkpoint/config compatibility pending.

README explicitly requires CoAlign complemented annotations: https://drive.google.com/file/d/13g3APNeHBVjPcF-nTuUoNOSGyTzdfnUK/view . Dataset loader unconditionally reads cooperative/label_world_backup and vehicle-side/label/lidar_backup as well as normal labels; missing backups block loading even if only lidar_all later used. Do not create empty labels to bypass.

Environment YAML Python3.7.11, no torch pin; requirements include shapely2.0.0, h5py, pyquaternion, efficientnet_pytorch. Existing isolated OpenCOOD-compatible CUDA env is preferable to upgrading running research env. spconv2 supported by SpVoxelPreprocessor; needs cumm. Need Cython box_overlaps and CUDA iou3d_nms; repo pcdet_utils/setup.py builds extra extensions too. Build commands from repo root:
```
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
```
Use read-only import preflight for torch/spconv/open3d/shapely/timm/einops/efficientnet_pytorch/h5py/pyquaternion; do not pip-editable-install over the main OpenCOOD package. Own PYTHONPATH plus copied compatible extensions or isolated builds avoids package collision.

## Important fidelity / communication / computation issues

1. Official `inference.py` has no byte statistics. `inference_d.py` references undefined `comm_recorder` at end; not a usable bandwidth baseline as-is. `point_pillar_comm_multiscale` discards returned communication rates.
2. No actual packet serializer or decoder boundary in official main model. Paper says sparse latent coordinates+FP16 values, but coordinate width, headers, demand/request bytes and late-prediction byte format are unspecified. CommRecorder pose default6bytes is a heuristic, not actual6FP32 values. Measure total bidirectional traffic, not only selected features; include request mask, geometry metadata and late results.
3. **Order discrepancy:** paper Fig/eq describes selection→encoder→sparse latent transmission→decoder→warp. Official model calls compressor encoder→FP16 cast→full decoder **before** Where2comm, then warps full decoded features and masks them in ego grid. A sparse latent transmission implementation may alter output because convolutional decoder needs neighboring latent cells; halo dependencies must be preserved. Do not claim exact packet equivalence by simply counting selected post-decoder locations divided by compression16. First reproduce AP of official graph, then report explicit accounting estimate separately from a verified wire implementation.
4. Official forward computes full backbone once for confidence, then `get_multiscale_feature` recomputes its blocks. Hybrid roadside late call also runs another forward. For compute comparison report stock end-to-end and any verified cached implementation separately. Do not win latency by leaving unnecessary duplicate computation only in ANN baseline.
5. `Communication.gaussian_filter.requires_grad=False` sets attribute on module, not parameters; training optimizer may include filter parameters. Preserve official graph for reproduction; check actual checkpoint gaussian weights before changing.
6. `backbone_fix` references nonexistent `self.naive_compressor` rather than ModuleList; don't enable this path blindly.

## Dataset parity with our SNN

Confirmed same nominal detection range and score/NMS thresholds as our e3dsnn_receiver_sa_181.yaml. Both reference cooperative_label_path projected from world into vehicle, no injected noise; shared files on146 may allow same split.

**Not yet established actual parity:** official repo carries no data IDs; compare train/val JSON hashes, ordered IDs, label hashes and GT arrays remotely before comparing AP. Our manifest selects4678train/1738val and filters `type` to Car/Van/Bus/Truck. Official CoSDH `project_world_objects_dairv2x` accepts every passed object without class filtering, expands z-range by1m before projection filtering, and final GT filter ignores z. Our code uses existing custom postprocessor; GT counts/coordinates may differ. Our main result has28622GT. Official complemented label_world plus backup scheme may differ from raw labels. Need obtain official graph's GT count and per-frame ordered/canonical hashes; if mismatch evaluate common-label branch separately, retaining paper-fidelity branch.

Our model additionally corrects roadside z shift and has sparse voxel .2/.2/.1 instead of pillars; preserve respective backbones for method comparison, but document architecture/compute differences rather than asserting activation-only fairness.

Follow-up source audit: with `label_type=lidar`, `generate_object_center_lidar` reads only `vehicles_all` (complemented cooperative world labels). `intermediate_late_fusion_dataset.get_item_single_car` uses this method twice, not `generate_object_center_single`; final hybrid postprocess constructs GT from ego only. Thus backup/front and single-side annotation fields are loaded unconditionally but unused in this particular LiDAR hybrid model/GT path. Prefer restoring genuine originals from archives. If unavailable, an explicitly documented LiDAR-only compatibility adapter may omit unused reads, with source diff and parity checks; never copy complemented labels into a falsely named original backup. Delivered evaluator currently performs no fallback and will fail on missing required loader files.

Parent deployment audit reports146 raw split counts4811/1789 versus our valid manifest4678/1738, and complemented labels contain onlyCar/Bus/Van/Truck. Accordingly class filtering likely does not change this dataset, but shared ordered valid split and actual projected GT equality still require runtime comparison.

## Recommended immediate sequence

1. Official checkpoint+config → strict load, 2-frame forward+GT dump, then1738-frame official AP reproduction on existing data only after annotation audit.
2. Export predictions+scores, match common GT/ordered IDs; rerun both with shared BEV evaluator and retain native/paper results distinctly.
3. Compare intermediate-only and full hybrid across a preregistered small confidence threshold set using held-out tuning discipline. Record actual measured bytes if serializer verified; otherwise label estimates explicitly.
4. Profile sender, receiver, voxelization, model and NMS independently; report complete end-to-end latency on sameGPU/batch1 incl sync/warmup; parameters and dense/sparse operation estimate conventions separately.
5. Use CoSDH to test whether baseline2's performance gap is missing high-value remote objects or receiver use. Its supply/demand idea and very small late branch suggest bounded SNN modifications, but must retain baseline2 primary and avoid resurrecting failed scheme3 without mechanism evidence.

## Delivered evaluator

`evaluate_cosdh.py` is a standalone official-hybrid evaluator. Python3.7 syntax compilation and CLI help passed locally; no GPU execution performed in this audit. Example (paths supplied by deployment owner):
```
python evaluate_cosdh.py --repo /path/CoSDH --model-dir /path/dairv2x_cosdh --data-root /path/cooperative-vehicle-infrastructure --split val.json --output /path/new_smoke_output --limit 2 --gpu 0
```
Use `--limit 0` and a new output directory for full evaluation. The evaluator refuses an existing output path, enforces exact checkpoint key/shape agreement, checks finite weights/model outputs, uses official intermediatelate postprocessing, exports pred/score/GT/frameID NPZ with SHA manifest, and reports official BEV AP. It sets seed303, workers0, batch1 and25% allocator limit; three warmup forwards on the first batch precede synchronized hybrid-inference timing. Latency includes official model calls and postprocessing, excludes data load/H2D/export/AP (load/H2D are separately recorded). Communication remains null until a real protocol is implemented; GT parity remains explicitly unestablished. Failed execution leaves failure.json; completed 2-frame smoke is not a main result.
