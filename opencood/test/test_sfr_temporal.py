import torch
import pytest
from opencood.models.sfr.temporal import Temporal, ReplayWindow


def temporal():
    return Temporal(dict(hidden_dim=16, max_speed=35., max_acceleration=6., max_yaw_rate=1., ttl_s=2., match_radius_m=5., gate_growth_mps=1.))


def msg(time, x, frame=None, source='1', scene='a'):
    boxes = torch.tensor([[x, 0., 0., 1., 2., 4., 0.]])
    return dict(boxes=boxes, scores=torch.tensor([.9]), points=boxes[:, :3].clone(), features=torch.ones(1, 4),
                instance_ids=torch.tensor([0]), metadata=dict(time_s=time, agent_id=source, frame_id=str(time) if frame is None else frame, sequence_id=scene))


def test_cv_real_last_observation_after_missing_update():
    model = temporal()
    state, _ = model.sequence([msg(-1., 0.), msg(-.5, 1.)], -.25, 'cv')
    observed = model.observe(msg(0., 2.), 0, 0., model.predict(state[0], 0., 'cv'), 'cv')
    assert observed.velocity[0] == pytest.approx(2.)
    assert observed.caches['1']['time'] == 0.


def test_prediction_identity_semigroup_and_cache_unchanged():
    model = temporal()
    track = model.observe(msg(-1., 0.), 0, -1.)
    track.acceleration = torch.tensor([2., 1.])
    a = model.predict(track, 0.)
    b = model.predict(model.predict(track, -.5), 0.)
    torch.testing.assert_close(a.box, b.box)
    torch.testing.assert_close(a.velocity, b.velocity)
    assert a.caches is track.caches
    torch.testing.assert_close(model.predict(track, -1.).box, track.box)
    with pytest.raises(ValueError):
        model.predict(track, -2.)


def test_source_permutation_dedup_and_unseen_collaborator_birth():
    model = temporal()
    messages = [msg(-.5, 8.), msg(0., 0., source='0'), msg(-1., 7.)]
    first, _ = model.sequence(messages, mode='cv')
    second, stats = model.sequence(list(reversed(messages))+[messages[0]], mode='cv')
    torch.testing.assert_close(torch.stack([t.box for t in first]), torch.stack([t.box for t in second]))
    assert len(first) == 2 and stats['deduplicated'] == 1
    data = model.collect(first)
    assert data is not None and data[0][0, 0] == pytest.approx(9.)


def test_late_message_replay_ttl_and_scene_reset():
    model = temporal()
    replay = ReplayWindow(model, 2.)
    replay.receive(msg(-.5, 1.), 0., 'cv')
    state, _ = replay.receive(msg(-1., 0.), 0., 'cv')
    assert state[0].box[0] == pytest.approx(2.)
    state, _ = replay.receive(msg(-3., -4.), 0., 'cv')
    assert replay.rejected == 1
    state, _ = replay.receive(msg(0., 20., scene='b'), 0., 'cv')
    assert len(state) == 1 and state[0].box[0] == 20.
    empty, _ = model.sequence([msg(-3., 0.)])
    assert empty == []


def test_future_observation_excluded_and_scene_mixture_rejected():
    model = temporal()
    assert model.sequence([msg(1., 0.)])[0] == []
    with pytest.raises(ValueError):
        model.sequence([msg(-1., 0.), msg(0., 0., scene='b')])


def test_collected_orientation_composes_sensor_and_final_transport_turn():
    from opencood.models.sfr.geometry import se2
    model=temporal();message=msg(-.5,0.)
    message['nominal_to_reference']=se2(torch.tensor([1.,2.,.3]))
    track=model.observe(message,0,-.5,mode='cv')
    track.box=track.box.clone();track.box[6]=.2
    data=model.collect([track],include_orientation=True)
    torch.testing.assert_close(data[4],torch.tensor([[torch.sin(torch.tensor(.5)),torch.cos(torch.tensor(.5))]]))
    assert torch.equal(data[1],message['features'])
