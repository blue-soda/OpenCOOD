from types import SimpleNamespace
import torch
from torch import nn
from opencood.models.sfr.sender import geometry_map, foreground_cells
from opencood.models.sfr.fusion import SfrFusion


def test_unobserved_is_not_free_and_padding_not_counted():
    voxels = torch.zeros(1, 4, 4)
    voxels[0, 0, :3] = torch.tensor([1.5, 2.5, 1.])
    voxels[0, 1, :3] = torch.tensor([1.5, 2.5, 3.])
    maps, valid = geometry_map(voxels, torch.tensor([2]), (4, 4), [0, 0, -1, 4, 4, 4])
    assert valid.sum() == 1 and valid[2, 1]
    assert maps[2, 1, 2] == 2 and maps[2, 1, 3] == 1
    assert not valid[0, 0]


def test_foreground_ownership_budget_and_empty():
    boxes = torch.tensor([[2., 2., 0., 1., 2., 2., 0.], [2., 2., 0., 1., 2., 2., 0.]])
    cfg = dict(foreground_margin_m=.1, cells_per_instance=4, max_cells=4)
    feature = torch.ones(3, 4, 4)
    result = foreground_cells(boxes, torch.tensor([.9, .8]), feature, [0, 0, -1, 4, 4, 1], cfg)
    assert len(result['points']) == 4 and result['instance_ids'].eq(0).all() and result['ambiguity'] == 4
    empty = foreground_cells(boxes[:0], torch.zeros(0), feature, [0, 0, -1, 4, 4, 1], cfg)
    assert empty['features'].shape == (0, 3)


def test_empty_collaboration_exact_single_and_frozen_head_input_gradient():
    sender = SimpleNamespace(out_channel=3, bounds=[0, 0, -1, 4, 4, 1],
                             cls_head=nn.Conv2d(3, 2, 1), reg_head=nn.Conv2d(3, 14, 1), use_dir=False)
    model = SfrFusion(sender, dict(hidden_dim=4, train_heads=False))
    ego = torch.randn(1, 3, 4, 4)
    output, stats = model(ego, None)
    torch.testing.assert_close(output['psm'], sender.cls_head(ego), rtol=0, atol=0)
    assert stats['coverage_fraction'] == 0
    features = torch.full((1, 3), 5., requires_grad=True)
    output, _ = model(ego, (torch.tensor([[1.7, 1.9, 0.]]), features, torch.tensor([.3]), torch.tensor([.9])))
    output['psm'].square().sum().backward()
    assert features.grad.abs().sum() > 0
    assert model.adapter.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.cls_head.parameters())
