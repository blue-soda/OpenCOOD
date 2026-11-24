"""Train the fixed-window backbone and its per-scan control on every eligible DAIR training label."""
import argparse
from pathlib import Path

import torch

from opencood.data_utils.int_v2v_ego_stream import v2v_rows
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid, periodic


def full_rows(manifest, datasets, split):
    if fit.digest(manifest) != '184798f833dd52b6eee2cf2f3b04ab7684221be43b9302edccd4889f51d88738':
        raise ValueError('Unexpected frozen full-stream manifest')
    rows = fit.read(manifest)[split]
    for row in rows:
        for key in ('pcd','label'):
            if row[key]:
                path = datasets / Path(row[key]).relative_to('/data1/tzm/dataset')
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[key] = str(path)
    return rows


def train(a):
    if a.qat_int8 and not a.continuous_backbone:
        raise ValueError('INT8 training requires the matched continuous backbone.')
    if a.train_drop_silent and (a.independent or a.pillar_up or (a.continuous_backbone and not a.qat_int8)):
        raise ValueError('Train the silent-remainder control on the native or INT8 temporal backbone.')
    a.out.mkdir(parents=True, exist_ok=a.resume)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    manifest = a.v2v or a.root / 'sequence_manifest.json'
    rows = v2v_rows(manifest,'train',getattr(a,'v2v_dataset_root',None)) if a.v2v else full_rows(manifest,a.datasets,'train')
    labels = sum(r['supervised'] for r in rows)
    scenes = len({r['scene'] for r in rows})
    assert (len(rows),labels,scenes) == ((6958,6958,31) if a.v2v else (8593,4678,52))
    cfg = fit.load_yaml(str(a.root/'inputs/config.yaml'))
    cfg['data_augment'] = []
    initial = a.root/'initial.pt'
    assert fit.digest(initial) == '57f9d146f2a4dd1344ad36dd670bc7f9bcdfd85ae04aaa8c4855cc635b148c33'
    src = fit.build_source(a.root/'inputs',cfg)
    variant = 'fine_independent' if a.independent else 'fine_temporal'
    net = RecoveryPyramid(src.core,initial,variant).cuda()
    weak_policy = getattr(a, 'deep_weak_policy', None)
    if weak_policy:
        if a.independent or a.pillar_up or a.frame_ann or a.shift_codes or a.train_drop_silent or a.adjacent_target or a.firing_only_gradient:
            raise ValueError('Deep weak-event training needs an unmodified temporal sender.')
        net.set_deep_weak_policy(weak_policy)
    firing_only_gradient = getattr(a, 'firing_only_gradient', False)
    if firing_only_gradient:
        if a.independent or a.continuous_backbone or a.shift_codes or a.adjacent_target or a.train_drop_silent or a.pillar_up or a.frame_ann:
            raise ValueError('Use the firing-only gradient control with the native temporal backbone.')
        for sender in net.senders:
            sender.firing_only_gradient = True
    if a.adjacent_target:
        if a.independent or (a.continuous_backbone and not a.qat_int8) or a.train_drop_silent or a.pillar_up:
            raise ValueError('Use adjacent-target differences on the native or quantization-trained INT8 temporal sender.')
        net.train_adjacent_target = True
        for sender in net.senders:
            sender.adjacent_target = True
    if a.shift_codes:
        if a.independent or a.continuous_backbone or a.train_drop_silent or a.pillar_up:
            raise ValueError('Single-shift messages require the native temporal backbone.')
        for sender in net.senders:
            sender.use_shift_codes(a.shift_codes)
    if a.train_drop_silent:
        net.train_drop_silent = True
        for sender in net.senders:
            sender.drop_silent_remainder = True
    if a.frame_ann:
        if not a.independent or a.pillar_up or a.continuous_backbone:
            raise ValueError('The spatial ANN control requires --independent and unmodified interfaces.')
        for sender in net.senders:
            sender.full_amplitude = True
            sender.log_step.requires_grad_(False)
    if a.continuous_backbone:
        if a.pillar_up or a.independent:
            raise ValueError('The continuous temporal control uses the unmodified Pillar/up interfaces.')
        for sender in net.senders:
            sender.continuous_threshold = .5
            if a.qat_int8:
                sender.payload_dtype = 'int8'
                sender.int8_ste = True
    criterion = fit.PointPillarDirLoss(cfg['loss']['args']).cuda()
    groups = [dict(params=[p for k,p in net.named_parameters() if not k.startswith('heads.')], lr=3e-5),
              dict(params=net.heads.parameters(),lr=1.5e-5)]
    if a.pillar_up:
        from opencood.tools.select_int_stream_variant import install_pillar_up
        install_pillar_up(net, src.core)
        calibration = periodic(rows[:20])
        _, samples = fit.loader(calibration, cfg, 0, 303)
        state = None
        with torch.no_grad():
            for row, sample in zip(calibration, samples):
                assert row['frame'] == sample['meta']['frame']
                if row['reset_before']:
                    state = None
                ego = fit.to_device(sample['ego'], torch.device('cuda'))
                spatial = fit.spatial_input(src, ego)
                _, state, _ = net(spatial, src.core, state, ordinary=not row['reset_before'])
        for interface in net.pillar_up:
            interface.calibrated()
        groups.append(dict(params=net.pillar_up.parameters(), lr=1e-3))
    opt = torch.optim.AdamW(groups,weight_decay=1e-4)
    initial_lrs = [group['lr'] for group in opt.param_groups]
    contract = dict(seed=a.seed, variant=variant, epochs=a.epochs, manifest=fit.digest(manifest),
                    initial=fit.digest(initial), labels_per_epoch=labels, scenes=scenes,
                    dataset='v2v_single' if a.v2v else 'dair_single',
                    training='all supervised five-scan clips; same shuffle, optimizer boundaries and fixed final epoch')
    if a.pillar_up:
        contract.update(pillar_up=True, interfaces=['pillar_output', 'fine_up', 'middle_up', 'deep_up'],
                        coding='E-3DSNN Multispike 0..4; floating point PFN and projection weights fixed',
                        calibration_frames=[row['frame'] for row in rows[:20]], learning_rates=initial_lrs)
    if a.continuous_backbone:
        contract.update(continuous_threshold=.5, coding='bounded continuous residuals at every backbone sender',
                        threshold_source='R-505 four training scenes; fixed before target validation')
    if a.qat_int8:
        contract.update(training_payload='int8', coding='signed INT8 feedback with a straight-through rounding gradient')
    if a.train_drop_silent:
        contract['train_drop_silent'] = True
    if a.shift_codes:
        contract['shift_codes'] = a.shift_codes
    if a.adjacent_target:
        contract['adjacent_target'] = True
    if firing_only_gradient:
        contract['firing_only_gradient'] = True
    if weak_policy:
        contract['deep_weak_policy'] = weak_policy
    if a.frame_ann:
        contract.update(frame_ann=True, coding='continuous spatial activations, no quantization or history')
    restore_path = None
    if a.cross_from_first:
        previous = fit.read(a.cross_from_first/'contract.json')
        if (a.epochs != 2 or previous['seed'] == a.seed or
                previous != dict(contract, seed=previous['seed'], epochs=1)):
            raise ValueError('Cross the second-epoch order from a matched first-epoch checkpoint.')
        restore_path = a.cross_from_first/'last.pt'
        contract.update(first_epoch_seed=previous['seed'], first_checkpoint_sha=fit.digest(restore_path),
                        first_endpoint_sha=fit.digest(a.cross_from_first/'final.pt'))
    cp = a.out/'contract.json'
    if cp.exists():
        assert fit.read(cp) == contract
    else:
        fit.write_json(cp,contract)
    start = 1
    if a.resume and (a.out/'last.pt').exists():
        restore_path = a.out/'last.pt'
    if restore_path is not None:
        checkpoint = torch.load(restore_path,map_location='cpu')
        if a.cross_from_first and restore_path == a.cross_from_first/'last.pt':
            endpoint = torch.load(a.cross_from_first/'final.pt',map_location='cpu')
            assert checkpoint['epoch'] == 1 and endpoint['contract'] == previous
            assert endpoint['net'].keys() == checkpoint['net'].keys()
            assert all(torch.equal(v,checkpoint['net'][k]) for k,v in endpoint['net'].items())
            del endpoint
        net.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
    for epoch in range(start,a.epochs+1):
        # Epoch-local traces make a restarted, incomplete epoch distinguishable.
        trace = a.out/('epoch_%02d' % epoch)
        attempt = 0
        while trace.exists():
            attempt += 1
            trace = a.out/('epoch_%02d_retry_%d' % (epoch,attempt))
        trace.mkdir()
        for group, lr in zip(opt.param_groups, initial_lrs):
            group['lr'] = lr if epoch == 1 else lr * .5
        if a.pillar_up:
            before = [s.log_step.detach().clone() for s in net.pillar_up]
        result = fit.replay(rows,cfg,src,net,criterion,opt,epoch,trace,{},memoryless=a.independent,
                            order_seed=a.seed,window_mode='fixed')
        assert result['labels'] == labels and result['updates'] > 0
        if a.pillar_up:
            result['step_parameter_l2_change'] = [
                (s.log_step.detach() - start_value).norm().item()
                for s, start_value in zip(net.pillar_up, before)]
        fit.write_json(trace/'result.json',result)
        fit.checkpoint(a.out/'last.pt',net,opt,epoch,0.,0)
    temp = a.out/'final.tmp'
    torch.save(dict(net=net.state_dict(), contract=contract),temp)
    temp.replace(a.out/'final.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','datasets','out'):
        p.add_argument(name,type=Path)
    p.add_argument('--seed',type=int,default=303)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--independent',action='store_true')
    p.add_argument('--deep-weak-policy', choices=('repeat','strong'),
                   help='Suppress isolated weak changes at deep internal convolution inputs.')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--cross-from-first',type=Path,help='Keep first-epoch model and optimizer; run the other order in epoch two.')
    p.add_argument('--pillar-up',action='store_true',help='Jointly fit four multispike interfaces.')
    p.add_argument('--continuous-backbone',action='store_true',help='Fit the matched real-valued temporal control.')
    p.add_argument('--qat-int8',action='store_true',help='Use the deployed INT8 payload at every training step.')
    p.add_argument('--train-drop-silent',action='store_true',help='Train with silent sender remainders discarded from the first step.')
    p.add_argument('--firing-only-gradient',action='store_true',
                   help='Zero the native sender surrogate gradient on silent outputs, preserving its forward computation.')
    p.add_argument('--shift-codes', choices=('ordinary', 'all'),
                   help='Train with signed power-of-two updates on the designated scans.')
    p.add_argument('--adjacent-target', action='store_true',
                   help='Encode changes relative to the prior true target, retaining the same receiver accumulation.')
    p.add_argument('--frame-ann',action='store_true',help='Fit the unquantized independently reset spatial network.')
    p.add_argument('--v2v',type=Path,help='Use the audited single-car, duplicate-free training manifest.')
    p.add_argument('--v2v-dataset-root',type=Path,help='Location of the V2V4Real dataset replica.')
    train(p.parse_args())
