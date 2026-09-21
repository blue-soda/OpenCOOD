import torch
from opencood.models.sfr.calibration import BackgroundCalibration, weighted_fit, background_points, pose_loss
from opencood.models.sfr.geometry import se2, transform_points


def settings():
    return dict(hidden_dim=16, radius_m=2., temperature_m=.2, max_height_difference_m=2., min_matches=4,
                min_overlap=.08, min_spread_m2=.1, max_translation_m=3., max_yaw_rad=.1, max_residual_m=.9,
                translation_loss_scale_m=1., yaw_loss_scale_rad=.03)


def test_planar_fit_left_correction_and_turn():
    source = torch.tensor([[0.,0.],[1.,2.],[4.,1.],[-2.,3.]])
    delta = torch.tensor([.3, -.2, -.03])
    target = transform_points(torch.cat((source,torch.zeros(4,1)),1),se2(delta))[:, :2]
    fitted = weighted_fit(source,target,torch.ones(4))
    torch.testing.assert_close(fitted,delta,atol=1e-6,rtol=1e-5)


def test_no_overlap_and_degenerate_fall_back():
    model=BackgroundCalibration(settings())
    source=dict(xyz=torch.tensor([[0.,0.,0.],[2.,0.,0.],[4.,0.,0.],[6.,0.,0.]]),descriptor=torch.ones(4,2))
    target=dict(xyz=source['xyz']+torch.tensor([100.,0.,0.]),descriptor=torch.ones(4,2))
    correction,_,stats=model(source,target)
    assert not stats['accepted'] and stats['reason']=='no_overlap'
    torch.testing.assert_close(correction,torch.eye(4))
    correction,_,stats=model(source,source)
    assert not stats['accepted'] and stats['reason']=='degenerate'


def test_known_small_perturbation_recovers_and_has_matcher_gradient():
    cfg=settings(); cfg['temperature_m']=.7
    model=BackgroundCalibration(cfg)
    xy=torch.tensor([[0.,0.],[1.,1.],[2.,0.],[0.,3.],[4.,4.],[5.,1.]])
    xyz=torch.cat((xy,torch.tensor([[.1],[.5],[.3],[1.],[.7],[.8]])),1)
    perturb=se2(torch.tensor([.2,-.1,.01]))
    source=dict(xyz=transform_points(xyz,perturb),descriptor=torch.ones(6,2))
    target=dict(xyz=xyz,descriptor=torch.ones(6,2))
    correction,delta,stats=model(source,target)
    assert stats['accepted']
    before=(source['xyz'][:,:2]-xyz[:,:2]).norm(dim=-1).mean()
    after=(transform_points(source['xyz'],correction)[:,:2]-xyz[:,:2]).norm(dim=-1).mean()
    assert after < before*.5
    loss=pose_loss(delta,torch.inverse(perturb),cfg);loss.backward()
    assert sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)>0


def test_background_excludes_foreground_and_unobserved():
    geometry=torch.zeros(6,6,4);geometry[:,:,2]=2
    observed=torch.ones(6,6,dtype=torch.bool);observed[0,0]=False
    foreground=torch.zeros_like(observed);foreground[2,2]=True
    points=background_points(dict(geometry=geometry,observed=observed,foreground=foreground),torch.eye(4),[0,0,-1,6,6,3],100,1)
    assert len(points['xyz'])==36-9-1
    assert (points['xyz'][:,2]==2).all()
