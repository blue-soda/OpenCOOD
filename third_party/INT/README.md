# INT source provenance

- Repository: https://github.com/ADLab-AutoDrive/INT
- Commit: `988157ff131a0c027472bd0f00c0bda0e08cded0`
- Original source: `det3d/models/detectors/voxelnet.py`, retained verbatim as `voxelnet.py.txt` for parity tests.
- The actual upstream LICENSE is Apache-2.0 (retained here), despite the upstream README describing MIT.

`opencood/models/sub_modules/int_feature_memory.py` ports the Concat and
`infinite_GRU` paths and their layer construction. Tests instantiate the actual
upstream class from the snapshot and compare outputs, cached state and gradients.

Adaptations: replace det3d norm/Sequential helpers with PyTorch equivalents;
make the 0.05-second age increment an explicit argument; avoid mutating caller
state; use metric SE(2) inverse sampling on a non-square BEV; externalize sequence
state. GRU stores hidden + age before readout projection, whereas Concat stores
the post-fusion output, as in the original detector.

The DAIR configuration uses 64-channel Pillar scatter features, full-width
Concat branches and one convolution per pre/post block. Concat starts as an
identity on nonnegative current features. This differs from the published Waymo
VoxelNet configuration and is an explicitly controlled FM-only port. PC, PM,
CenterPoint, the full official training recipe and leaderboard reproduction are
not part of this implementation.
