# -*- coding: utf-8 -*-
"""DAIR-V2X-C reader for CoBEVFlow-style irregular asynchronous training.

This class keeps the CoBEVFlow multi-frame output contract used by
``IntermediateFusionDatasetIrregularFlowNew`` while making DAIR metadata
handling explicit and portable. It follows the DAIR directory conventions used
by OpenCOOD/CoST:

    cooperative-vehicle-infrastructure/
      cooperative/data_info.json
      vehicle-side/data_info.json
      infrastructure-side/data_info.json
      train.json / val.json
      ...
"""

import copy
import os
import os.path as osp
from collections import OrderedDict

from scipy import stats

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_irregular_multi import (
    IntermediateFusionDatasetDAIRIrregularMulti,
    id_to_str,
    load_json,
)


class CoBEVFlowDAIRIrregularDataset(IntermediateFusionDatasetDAIRIrregularMulti):
    """CoBEVFlow-compatible DAIR-V2X-C irregular asynchronous dataset.

    Vehicle-side lidar is treated as ego. Infrastructure-side lidar is treated
    as the collaborating agent. For the collaborating agent, historical frames
    are sampled with the same Bernoulli frame-delay convention as CoBEVFlow's
    irregular OPV2V reader: ``sum(Binomial(n, p))`` frames per recurrent step.
    """

    def __init__(self, params, visualize, train=True):
        params = copy.deepcopy(params)
        data_dir = params.get("data_dir") or params.get("dair_data_dir")
        if not data_dir:
            raise ValueError(
                "CoBEVFlowDAIRIrregularDataset requires `data_dir` "
                "or `dair_data_dir` pointing to cooperative-vehicle-infrastructure."
            )

        params["data_dir"] = data_dir
        params.setdefault("dair_data_dir", data_dir)
        params["root_dir"] = self._resolve_split(params.get("root_dir"), data_dir, "train.json")
        params["validate_dir"] = self._resolve_split(
            params.get("validate_dir"), data_dir, "val.json"
        )
        params["test_dir"] = self._resolve_split(params.get("test_dir"), data_dir, "val.json")
        super().__init__(params=params, visualize=visualize, train=train)

    @staticmethod
    def _resolve_split(path, data_dir, default_name):
        if not path:
            path = osp.join(data_dir, default_name)
        if osp.isdir(path):
            path = osp.join(path, default_name)
        if osp.exists(path):
            return path
        anno_split = osp.join(
            osp.dirname(data_dir), "DAIR-V2X-C_Complemented_Anno", default_name
        )
        if osp.exists(anno_split):
            return anno_split
        return path

    @staticmethod
    def _frame_id_from_path(path):
        if not path:
            return None
        filename = path.replace("\\", "/").split("/")[-1]
        return osp.splitext(filename)[0]

    @classmethod
    def _vehicle_frame_id(cls, frame_info):
        return cls._frame_id_from_path(frame_info.get("vehicle_pointcloud_path"))

    @classmethod
    def _infrastructure_frame_id(cls, frame_info):
        return cls._frame_id_from_path(
            frame_info.get("infrastructure_pointcloud_path")
            or frame_info.get("infrastructure_image_path")
        )

    def _resolve_lidar_path(self, relative_path):
        candidate = osp.join(self.root_dir, relative_path)
        if osp.exists(candidate):
            return candidate

        normalized = relative_path.replace("\\", "/")
        filename = normalized.split("/")[-1]
        if normalized.startswith("vehicle-side/velodyne/"):
            alt_dir = osp.join(
                osp.dirname(self.root_dir),
                "cooperative-vehicle-infrastructure-vehicle-side-velodyne",
            )
            alt_path = osp.join(alt_dir, filename)
            if osp.exists(alt_path):
                return alt_path
        if normalized.startswith("infrastructure-side/velodyne/"):
            alt_dir = osp.join(
                osp.dirname(self.root_dir),
                "cooperative-vehicle-infrastructure-infrastructure-side-velodyne",
            )
            alt_path = osp.join(alt_dir, filename)
            if osp.exists(alt_path):
                return alt_path

        return candidate

    def is_valid_id(self, veh_frame_id):
        """Check whether the current DAIR pair has enough historical inf frames."""
        if veh_frame_id not in self.co_idx2info:
            return False

        frame_info = self.co_idx2info[veh_frame_id]
        inf_frame_id = self._infrastructure_frame_id(frame_info)
        if inf_frame_id is None or inf_frame_id not in self.inf_idx2info:
            return False

        cur_inf_info = self.inf_idx2info[inf_frame_id]
        batch_start_id = int(cur_inf_info.get("batch_start_id", 0))
        if int(inf_frame_id) - self.binomial_n * self.k < batch_start_id:
            return False

        for i in range(self.binomial_n * self.k):
            delay_id = id_to_str(int(inf_frame_id) - i)
            if delay_id not in self.inf_fid2veh_fid:
                return False
        return True

    def retrieve_base_data(self, idx):
        """Return current ego frame plus delayed infrastructure history."""
        final_data = OrderedDict()
        curr_veh_frame_id = self.data[idx]
        frame_info = self.co_idx2info[curr_veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        curr_inf_frame_id = self._infrastructure_frame_id(frame_info)
        bernoulli_dist = stats.bernoulli(self.binomial_p)

        for cav_idx in range(2):
            final_data[cav_idx] = OrderedDict()
            final_data[cav_idx]["ego"] = cav_idx == 0
            final_data[cav_idx]["debug"] = {
                "scene": "dair-v2x-c",
                "scene_name": "dair-v2x-c",
                "cav_id": str(cav_idx),
            }
            final_data[cav_idx]["curr"] = {}
            final_data[cav_idx]["curr"]["frame_id"] = (
                curr_veh_frame_id if cav_idx == 0 else curr_inf_frame_id
            )
            final_data[cav_idx]["curr"]["timestamp"] = (
                curr_veh_frame_id if cav_idx == 0 else curr_inf_frame_id
            )
            final_data[cav_idx]["curr"]["time_diff"] = 0
            final_data[cav_idx]["curr"]["sample_interval"] = 0

            if cav_idx == 0:
                lidar_path = frame_info["vehicle_pointcloud_path"]
                curr_pose = self.get_vehicle_trans(curr_veh_frame_id)
                vehicles_single = load_json(
                    osp.join(
                        self.root_dir,
                        "vehicle-side/label/lidar/{}.json".format(curr_veh_frame_id),
                    )
                )
            else:
                lidar_path = frame_info["infrastructure_pointcloud_path"]
                curr_pose = self.get_inf_trans(curr_inf_frame_id, system_error_offset)
                vehicles_single = load_json(
                    osp.join(
                        self.root_dir,
                        "infrastructure-side/label/virtuallidar/{}.json".format(
                            curr_inf_frame_id
                        ),
                    )
                )

            final_data[cav_idx]["curr"]["lidar_np"] = pcd_utils.read_pcd(
                self._resolve_lidar_path(lidar_path)
            )[0]
            final_data[cav_idx]["curr"]["params"] = OrderedDict()
            final_data[cav_idx]["curr"]["params"]["vehicles"] = (
                load_json(osp.join(self.root_dir, frame_info["cooperative_label_path"]))
                if cav_idx == 0
                else []
            )
            final_data[cav_idx]["curr"]["params"]["vehicles_single"] = vehicles_single
            final_data[cav_idx]["curr"]["params"]["lidar_pose"] = curr_pose

            final_data[cav_idx]["past_k"] = OrderedDict()
            if cav_idx == 0:
                for hist_idx in range(self.k):
                    final_data[cav_idx]["past_k"][hist_idx] = final_data[0]["curr"]
                continue

            latest_frame_id = curr_inf_frame_id
            for hist_idx in range(self.k):
                sample_interval = int(sum(bernoulli_dist.rvs(self.binomial_n)))
                latest_frame_id = id_to_str(int(latest_frame_id) - sample_interval)
                veh_frame_id_of_inf = self.inf_fid2veh_fid[latest_frame_id]
                hist_frame_info = self.co_idx2info[veh_frame_id_of_inf]
                hist_system_offset = hist_frame_info["system_error_offset"]

                final_data[cav_idx]["past_k"][hist_idx] = {}
                final_data[cav_idx]["past_k"][hist_idx]["frame_id"] = latest_frame_id
                final_data[cav_idx]["past_k"][hist_idx]["timestamp"] = latest_frame_id
                final_data[cav_idx]["past_k"][hist_idx]["time_diff"] = (
                    int(curr_inf_frame_id) - int(latest_frame_id)
                )
                final_data[cav_idx]["past_k"][hist_idx]["sample_interval"] = -sample_interval
                final_data[cav_idx]["past_k"][hist_idx]["lidar_np"] = pcd_utils.read_pcd(
                    self._resolve_lidar_path(hist_frame_info["infrastructure_pointcloud_path"])
                )[0]
                final_data[cav_idx]["past_k"][hist_idx]["params"] = OrderedDict()
                final_data[cav_idx]["past_k"][hist_idx]["params"]["vehicles"] = []
                final_data[cav_idx]["past_k"][hist_idx]["params"]["lidar_pose"] = (
                    self.get_inf_trans(latest_frame_id, hist_system_offset)
                )

        return final_data
