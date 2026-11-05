"""Run the existing stage profiler with the archived CoDynTrust fusion class.

The current OpenCOOD checkout has the CoDynTrust model but its copied
``raindrop_flow.py`` misses the local ``AttenFusion`` symbol.  The archived
CoDynTrust implementation uses the same attention class as where2comm, so we
inject that class at runtime without modifying either source file.
"""

import runpy
import importlib.util
import sys

import opencood.data_utils.post_processor.voxel_postprocessor as current_voxel

# Use the complete CoDynTrust fusion implementation.  The current checkout's
# raindrop_flow.py is a reduced variant and does not define HybridFusion,
# although the selected checkpoint/configuration requires it.
_att_name = 'opencood.models.fuse_modules.att_fuse'
_att_spec = importlib.util.spec_from_file_location(
    _att_name,
    '/data0/chen/gzc/workspace/CoDynTrust/opencood/models/fuse_modules/att_fuse.py')
_archived_att = importlib.util.module_from_spec(_att_spec)
sys.modules[_att_name] = _archived_att
_att_spec.loader.exec_module(_archived_att)

_flow_name = 'opencood.models.fuse_modules.raindrop_flow'
_flow_spec = importlib.util.spec_from_file_location(
    _flow_name,
    '/data0/chen/gzc/workspace/CoDynTrust/opencood/models/fuse_modules/raindrop_flow.py')
_archived_flow = importlib.util.module_from_spec(_flow_spec)
sys.modules[_flow_name] = _archived_flow
_flow_spec.loader.exec_module(_archived_flow)

# The archived CoDynTrust checkpoint expects this uncertainty-aware post
# processing method.  The current checkout contains the dataset/model call but
# not the method, so attach the archived implementation at runtime for this
# read-only benchmark.  No repository source file is modified.
_spec = importlib.util.spec_from_file_location(
    'codyntrust_archived_voxel_postprocessor',
    '/data0/chen/gzc/workspace/CoDynTrust/opencood/data_utils/post_processor/voxel_postprocessor.py')
_archived_voxel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_archived_voxel)
current_voxel.VoxelPostprocessor.single_post_process_w_uncertainty = (
    _archived_voxel.VoxelPostprocessor.single_post_process_w_uncertainty)

runpy.run_path('/data0/chen/gzc/workspace/diagnostics/profile_latency_stages.py',
               run_name='__main__')
