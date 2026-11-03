"""Verified fixed event-training terminals for fresh-stream inference only."""
from opencood.tools.int_joint_adaptation_setup import *


EVENT_ARGUMENTS=ARGUMENTS+('event-learning-verification','event-learning-source')


def prepare_event(a,out):
    base,cfg,rows,source,old_exports,cal,rc,identity=prepare(a,out)
    vp=Path(a.event_learning_verification);v=json.loads(vp.read_text());lp=Path(a.event_learning_source)
    assert v['status']=='VERIFIED' and v['run']=='R-234'
    for n in ('contract.json','checkpoint_manifest.json'):
        assert digest(lp/n)==v['source_sha256'][n]
    c=json.loads((lp/'contract.json').read_text())
    for k in ('checkpoint_sha256','protected_checkpoint_sha256','initialization_sha256','calibration_sha256','readout_calibration_sha256','learning_contract_sha256','trained_checkpoint_sha256'):
        assert c[k]==identity[k],k
    assert not c['preflight_only'] and c['event_weight']==c['unit_energy_ratio']==.1 and c['dense_readout_mac_normalizer']==20230963200
    assert len(c['train_frames'])==2048 and c['training_labels']==1396 and len(c['expected_update_frames'])==389 and c['tbptt']==4
    assert c['preflight_verification_sha256']==v['preflight_verification_sha256']
    manifest={z['name']:z for z in json.loads((lp/'checkpoint_manifest.json').read_text())['files']}
    exports={};checksums={};frozen=frozen_digest(source)
    for name in ('control','event'):
        path=lp/(name+'_final.pth');checksums[name]=digest(path)
        assert checksums[name]==manifest[path.name]['sha256']==v['training'][name]['checkpoint_sha256']
        assert path.stat().st_size==manifest[path.name]['bytes']
        saved=torch.load(path,map_location='cpu');contract=dict(saved['contract']);contract['trainable_prefixes']=list(contract['trainable_prefixes'])
        assert contract==c and saved['mode']==name and saved['next_scan']==2048 and saved['updates']==c['expected_update_frames']
        model=copy.deepcopy(source);assert load_state_update_targets(model,saved['targets'])==2449664
        assert frozen_digest(model)==frozen
        exports[name]=model.eval()
    # The zero-coefficient terminal must reconstruct the original detection-only model.
    assert model_digest(exports['control'])==model_digest(old_exports['qat'])
    identity.update(event_training_verification_sha256=digest(vp),event_training_contract_sha256=digest(lp/'contract.json'),event_checkpoint_sha256=checksums,control_original_model_exact=True)
    del old_exports;torch.cuda.empty_cache()
    return base,cfg,rows,source,exports,cal,rc,identity


def require_event_export(a,identity):
    proof=json.loads(Path(a.event_export_verification).read_text())
    assert proof['status']=='VERIFIED' and proof['run']=='R-235'
    for k in ('event_checkpoint_sha256','calibration_sha256','readout_calibration_sha256','event_training_verification_sha256'):
        assert proof[k]==identity[k],k
    return digest(Path(a.event_export_verification))
