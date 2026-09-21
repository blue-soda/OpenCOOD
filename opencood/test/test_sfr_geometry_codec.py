import math
import pytest
import torch
from opencood.models.sfr.geometry import se2, transform_points, transform_boxes, transport, splat, warp_map, associate
from opencood.models.sfr.codec import quantize, fake_quantize, encode, decode


def test_transport_rigid_turn_and_center_gradient():
    old = torch.tensor([[1., 2., 0., 1., 2., 4., 0.]])
    new = torch.tensor([[4., 5., 0., 1., 2., 4., math.pi/2]], requires_grad=True)
    point = torch.tensor([[3., 2., 0.]])
    moved = transport(point, old, new)
    torch.testing.assert_close(moved, torch.tensor([[4., 7., 0.]]))
    moved.sum().backward()
    assert new.grad[:, :3].abs().sum() > 0 and new.grad[:, 6].abs().sum() > 0


def test_warp_inverse_and_forward_translation_agree():
    source = torch.zeros(1, 1, 4, 5)
    source[0, 0, 1, 1] = 1
    bounds = [0, 0, -1, 5, 4, 1]
    transform = se2(torch.tensor([1., 0., 0.]))
    warped = warp_map(source, transform, bounds)
    assert warped[0, 0, 1, 2] == pytest.approx(1.)
    moved = transform_points(torch.tensor([[1.5, 1.5, 0.]]), transform)
    dense, coverage = splat(moved, torch.ones(1, 1), bounds, (4, 5))
    torch.testing.assert_close(warped, dense)
    assert coverage.sum() == 1


def test_subcell_splat_empty_conflict_and_coordinate_gradient():
    xyz = torch.tensor([[1.2, 1.3, 0.], [1.2, 1.3, 0.]], requires_grad=True)
    dense, mass = splat(xyz, torch.tensor([[2.], [4.]]), [0, 0, -1, 4, 4, 1], (4, 4))
    assert mass.sum().item() == pytest.approx(2.)
    torch.testing.assert_close(dense[mass > 0], torch.full_like(dense[mass > 0], 3.))
    (mass*torch.arange(16).reshape(1, 1, 4, 4)).sum().backward()
    assert xyz.grad[:, :2].abs().min() > 0
    empty, coverage = splat(xyz[:0], torch.zeros(0, 3), [0, 0, -1, 4, 4, 1], (4, 4))
    assert empty.shape == (1, 3, 4, 4) and empty.sum() == coverage.sum() == 0


def test_codec_real_roundtrip_accounts_every_byte():
    values = torch.randn(13, 5)
    q, scale = quantize(values)
    packet, costs = encode({'agent_id': 'infra', 'observation_us': 12345}, {'features': q, 'scale': scale, 'valid': torch.ones(3, 4, dtype=torch.bool)})
    meta, tensors = decode(packet)
    assert sum(v for k, v in costs.items() if k != 'total') == len(packet) == costs['total']
    assert meta['observation_us'] == 12345
    assert ((tensors['features'].float()*tensors['scale']-values).abs() <= scale/2+1e-6).all()
    x = values.requires_grad_()
    fake_quantize(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    with pytest.raises(ValueError):
        decode(packet[:-1])


def test_static_association_and_transform_composition():
    box = torch.tensor([[1., 2., 0., 1., 2., 4., 0.]])
    a, b = se2(torch.tensor([1., 0., .3])), se2(torch.tensor([0., 2., -.1]))
    torch.testing.assert_close(transform_boxes(transform_boxes(box, a), b), transform_boxes(box, b@a))
    rows, cols = associate(box, box)
    assert rows.tolist() == cols.tolist() == [0]
