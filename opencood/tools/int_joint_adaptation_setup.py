"""Frozen joint quantized interface for ordinary-state adaptation; no real training."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_fine_integer import FINE_POLICY
from opencood.models.sub_modules.int_fine_readout_integer import FineReadoutIntegerDetector
from opencood.models.sub_modules.int_joint_state_qat import JointStateQATDetector,STATE_QAT_PREFIXES
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.int_artifact_paths import artifact_path
from opencood.models.sub_modules.int_joint_state_qat import load_state_update_targets
from opencood.tools.pilot_int_joint_state_qat import frozen_digest
import copy,numpy as np
from opencood.utils import eval_utils
def prepare(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    torch.cuda.set_per_process_memory_fraction(.65)
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert (audit.get('status')=='VERIFIED' if not a.validation_prefix else audit['frames']==5794 and len(audit['files'])==7532);assert gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for item in audit['files']:assert digest(Path(artifact_path(item['path'])) if a.validation_prefix else Path(item['relocated_path']))==item['sha256']
    a.mode='gru';a.stage='full' if a.validation_prefix else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    for p in source.parameters():p.requires_grad_(False)
    
    fv=json.loads(Path(a.fine_verification).read_text());assert fv['status']=='VERIFIED' and fv['run']=='R-200'
    assert digest(Path(a.calibration))==fv['source_sha256']['calibration.json'] and fv['checkpoint_sha256']==gate['checkpoint_sha256']
    pv=json.loads(Path(a.primitive_verification).read_text());assert pv['status']=='VERIFIED' and pv['run']=='R-204' and pv['tests']==446
    calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    proof=json.loads(Path(a.fine_stream_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-205' and proof['calibration_sha256']==digest(Path(a.calibration))
    iv=json.loads(Path(a.implicit_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-201' and iv['checkpoint_sha256']==gate['checkpoint_sha256'] and iv['calibration_sha256']==digest(Path(a.readout_calibration))
    rc=json.loads(Path(a.readout_calibration).read_text());assert rc['relative_quantum']==.125 and rc['floor']==1e-4
    lv=json.loads(Path(a.learning_verification).read_text());assert lv['status']=='VERIFIED' and lv['run']=='R-221' and lv['attempt']=='v2'
    lp=Path(a.learning_source);assert digest(lp/'contract.json')==lv['source_sha256']['contract.json'];lc=json.loads((lp/'contract.json').read_text())
    assert lc['checkpoint_sha256']==gate['checkpoint_sha256'] and lc['protected_checkpoint_sha256']==gate['training']['protected_light']['checkpoint_sha256'] and lc['initialization_sha256']==cv['rank128_checkpoint_sha256']
    assert lc['calibration_sha256']==digest(Path(a.calibration)) and lc['readout_calibration_sha256']==digest(Path(a.readout_calibration)) and len(lc['expected_update_frames'])==389
    quality=json.loads(Path(a.quality_verification).read_text());assert quality['status']=='VERIFIED' and quality['run']=='R-212' and quality['checkpoint_sha256']==gate['checkpoint_sha256']
    quality_contract=Path(a.quality_source)/'contract.json';assert digest(quality_contract)==quality['source_sha256']['contract.json']
    assert quality['calibration_sha256']==digest(Path(a.calibration)) and json.loads(quality_contract.read_text())['readout_calibration_sha256']==digest(Path(a.readout_calibration))
    exports={};trained_hashes={};frozen=frozen_digest(source)
    for name in ('ann','qat'):
        path=lp/(name+'_final.pth');assert digest(path)==lv['training'][name]['checkpoint_sha256'];saved=torch.load(path,map_location='cpu')
        saved_contract=dict(saved['contract']);saved_contract['trainable_prefixes']=list(saved_contract['trainable_prefixes'])
        assert saved_contract==lc and saved['mode']==name and saved['updates']==lc['expected_update_frames']
        model=copy.deepcopy(source);assert load_state_update_targets(model,saved['targets'])==2449664
        assert frozen_digest(model)==frozen and model_digest(model)!=model_digest(source)
        exports[name]=model.eval();trained_hashes[name]=digest(path)
    identity=dict(checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=lc['protected_checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),calibration_sha256=digest(Path(a.calibration)),readout_calibration_sha256=digest(Path(a.readout_calibration)),learning_verification_sha256=digest(Path(a.learning_verification)),quality_verification_sha256=digest(Path(a.quality_verification)),trained_checkpoint_sha256=trained_hashes,learning_contract_sha256=digest(lp/'contract.json'),audit_sha256=digest(Path(a.audit)),trainable_prefixes=STATE_QAT_PREFIXES,adapted_parameters=2449664,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),precision=dict(matmul_tf32=False,cudnn_tf32=False),cuda_memory_fraction=.65)
    return base,cfg,rows,source,exports,calibration,rc,identity


ARGUMENTS=('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','fine-stream-verification','implicit-verification','readout-calibration','learning-verification','learning-source','quality-verification','quality-source','output')
LEDGER_KEYS=('core_fp32_conv_flops','fine_fp64_reference_flops','fine_integer_macs','readout_fp64_reference_flops','readout_integer_macs','logical_core_conv_flops','cache_bytes','audit_scratch_bytes')


def ledger(model,info,fp32):
    fine=info.get('fine_integer',{})
    if isinstance(model,JointStateQATDetector) and model.enabled:fine={k:dict(v.statistics) for k,v in model.core.observation.leaves().items()}
    readout=info.get('readout_events',{});ff64=sum(z['actual_fp64_flops'] for z in fine.values());rf64=sum(z['actual_fp64_conv_flops'] for z in readout.get('layers',[]))
    if isinstance(model,JointStateQATDetector):rf64=model.core.readout.last_reference_flops
    return dict(core_fp32_conv_flops=fp32,fine_fp64_reference_flops=ff64,fine_integer_macs=sum(z['actual_integer_macs'] for z in fine.values()),readout_fp64_reference_flops=rf64,readout_integer_macs=sum(z['actual_integer_macs'] for z in readout.get('layers',[])),logical_core_conv_flops=fp32+(81749606400 if fine else 0),cache_bytes=info['cache_bytes'],audit_scratch_bytes=info.get('audit_scratch_bytes',0),reset=info['reset'],refresh=info['full_refresh'],layers=fine,readout_events=readout)


def freeze(models):
    for m in models:
        for p in m.parameters():p.requires_grad_(False)
        m.eval()


def metrics(models,records):
    return {n:dict(parameters=sum(p.numel() for p in m.parameters()),**{k+'_mean':sum(r['arms'][n][k] for r in records)/len(records) for k in LEDGER_KEYS},model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
