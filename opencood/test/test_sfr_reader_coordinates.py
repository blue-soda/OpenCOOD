from collections import OrderedDict
import numpy as np
import torch
from opencood.data_utils.datasets.sfr_dair import SfrDAIRDataset
from opencood.models.sfr.geometry import reader_to_reference, transform_points
from opencood.utils.transformation_utils import x1_to_x2


def test_actual_reader_direction_against_independent_pose_conversion():
    reader = object.__new__(SfrDAIRDataset)
    reader.proj_first, reader.k = False, 2
    ego = [10., -3., 1., 0., 25., 0.]
    infra = [40., 12., 5., 0., -37., 0.]
    base = OrderedDict((agent, dict(past_k={i:dict(params=dict(lidar_pose=pose)) for i in range(2)}))
                       for agent, pose in ((0, ego), (1, infra)))
    matrices = reader.get_past_k_pairwise_transformation2ego(base, ego, 2)
    expected = torch.from_numpy(x1_to_x2(infra, ego))
    inverse_sampling = torch.from_numpy(matrices[1, 0])
    corrected = reader_to_reference(inverse_sampling)
    torch.testing.assert_close(corrected, expected)
    assert not np.allclose(matrices[1, 0], expected.numpy())
    point = torch.tensor([[2., 1., -1.]], dtype=torch.float64)
    torch.testing.assert_close(transform_points(point, corrected), transform_points(point, expected))
    torch.testing.assert_close(reader_to_reference(torch.from_numpy(matrices[0, 0])), torch.eye(4, dtype=torch.float64))
