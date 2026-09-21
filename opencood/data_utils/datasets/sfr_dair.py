"""Metadata-only SFR extension of the unchanged DAIR binomial sampler."""
import numpy as np
from opencood.data_utils.datasets.codyntrust_dair_irregular_flow_dataset import CoDynTrustDAIRIrregularFlowDataset
from opencood.utils.transformation_utils import x1_to_x2


class SfrDAIRDataset(CoDynTrustDAIRIrregularFlowDataset):
    def retrieve_base_data(self, idx):
        base = super().retrieve_base_data(idx)
        vehicle = base[0]['curr']
        infrastructure = base[1]['curr']
        vi = self.ectra_history_index.by_id[vehicle['frame_id']]
        ii = self.inf_idx2info[infrastructure['frame_id']]
        now_us, paired_us = int(vi['pointcloud_timestamp']), int(ii['pointcloud_timestamp'])
        observations = []
        for agent in (0, 1):
            row = []
            for frame in base[agent]['past_k'].values():
                info = self.ectra_history_index.by_id[frame['frame_id']] if agent == 0 else self.inf_idx2info[frame['frame_id']]
                stamp = int(info['pointcloud_timestamp'])
                row.append(dict(agent_id=str(agent), frame_id=frame['frame_id'], observation_us=stamp,
                                native_sequence_id=str(info['batch_id']), sequence_id=str(vi['batch_id']),
                                paired_time_s=(stamp-(now_us if agent == 0 else paired_us))/1e6,
                                physical_time_s=(stamp-now_us)/1e6,
                                nominal_pose=np.asarray(frame['params']['lidar_pose']).tolist()))
            observations.append(row)
        self._sfr_metadata = dict(vehicle_frame=vehicle['frame_id'], infrastructure_frame=infrastructure['frame_id'],
                                 ego_timestamp_us=now_us, paired_timestamp_us=paired_us,
                                 paired_offset_ms=(paired_us-now_us)/1000., sequence_id=str(vi['batch_id']),
                                 observations=observations,
                                 current_infrastructure_to_ego=x1_to_x2(infrastructure['params']['lidar_pose'], vehicle['params']['lidar_pose']).tolist())
        return base

    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        if output is not None:
            output['ego']['sfr_metadata'] = self._sfr_metadata
        return output

    def collate_batch_train(self, batch):
        output = super().collate_batch_train(batch)
        if output is not None:
            output['ego']['sfr_metadata'] = [item['ego']['sfr_metadata'] for item in batch]
        return output
