import torch
from opencood.models.sfr.temporal import Temporal
from opencood.tools.sfr_run import motion_objective


def test_loss_replay_supervises_middle_and_current_without_target_update():
    model = Temporal(dict(hidden_dim=16, max_speed=35., max_acceleration=6., max_yaw_rate=1., ttl_s=2., match_radius_m=5., gate_growth_mps=1.))
    def message(time, x):
        box = torch.tensor([[x, 0., 0., 1., 2., 4., 0.]])
        return dict(boxes=box, scores=torch.tensor([.9]), points=box[:, :3].clone(), features=torch.ones(1, 4),
                    instance_ids=torch.tensor([0]), metadata=dict(time_s=time, arrival_s=0., agent_id='1', frame_id=str(time), sequence_id='a'))
    history = [message(-.4, 0.), message(-.2, .4)]
    target = message(0., .8)
    settings = dict(motion_min_dt_s=.05, motion_pseudo_score=.5, motion_yaw_weight=.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    original = [m['boxes'].clone() for m in history]
    for step in range(2):
        optimizer.zero_grad()
        loss, diagnostics = motion_objective(model, history, target, settings)
        assert diagnostics['motion_valid_count'] == 2
        loss.backward()
        assert sum(p.grad.abs().sum() for p in model.motion.parameters()) > 0
        if step:
            assert sum(p.grad.abs().sum() for p in model.encoder.parameters()) > 0
            assert sum(p.grad.abs().sum() for p in model.recurrent.parameters()) > 0
        optimizer.step()
    for a, b in zip(original, history):
        torch.testing.assert_close(a, b['boxes'])


def test_fixed_cv_assignment_cannot_hide_large_learned_error():
    model=Temporal(dict(hidden_dim=16,max_speed=35.,max_acceleration=6.,max_yaw_rate=1.,ttl_s=2.,match_radius_m=5.,gate_growth_mps=1.,minimum_motion_observations=2))
    def message(time,x):
        box=torch.tensor([[x,0.,0.,1.,2.,4.,0.]])
        return dict(boxes=box,scores=torch.tensor([.9]),points=box[:,:3],features=torch.ones(1,4),instance_ids=torch.tensor([0]),
                    metadata=dict(time_s=time,arrival_s=0.,agent_id='1',frame_id=str(time),sequence_id='a'))
    with torch.no_grad():model.motion[-1].bias.fill_(100.)
    settings=dict(motion_min_dt_s=.05,motion_pseudo_score=.5,motion_yaw_weight=.2,motion_min_observations=2,motion_fixed_cv_targets=True)
    history=[message(-1.8,0.),message(-1.3,1.)];target=message(0.,3.6)
    cv,cv_stats=motion_objective(model,history,target,settings,'cv')
    learned,learned_stats=motion_objective(model,history,target,settings,'learned')
    assert cv_stats['motion_valid_count']==learned_stats['motion_valid_count']==1
    assert learned_stats['motion_center_error_m']>4 and float(learned)>float(cv)
    assert learned_stats['motion_excluded_history_count']>0
