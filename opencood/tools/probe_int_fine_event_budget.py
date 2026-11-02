"""Shadow fine-encoder event budgets; ANN activations are never replaced."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason
from torch.nn import functional as F


def input_fanout(conv,x,output):
    h,w=output.shape[-2:];kh,kw=conv.kernel_size;sh,sw=conv.stride;ph,pw=conv.padding
    oh=x.shape[-2]-((h-1)*sh-2*ph+kh);ow=x.shape[-1]-((w-1)*sw-2*pw+kw)
    assert 0<=oh<sh and 0<=ow<sw
    fan=F.conv_transpose2d(x.new_ones(1,1,h,w),x.new_ones(1,1,kh,kw),stride=conv.stride,padding=conv.padding,output_padding=(oh,ow)).to(torch.int64)
    assert fan.shape[-2:]==x.shape[-2:]
    return fan


def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert audit['status']==gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for item in audit['files']:assert digest(Path(item['relocated_path']))==item['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    model=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(model,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    model.core.readout,_=compact_readout(base.core.readout,128);model.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    before=model_digest(model);calibration_rows=rows[:512];evaluation_rows=rows[512:768];assert len(calibration_rows)==512 and len(evaluation_rows)==256
    fine={name:m for name,m in model.core.observation.named_modules() if isinstance(m,nn.Conv2d)};assert fine
    sums={};elements={};quanta={};previous={};fanouts={};handles=[];context=dict(phase='calibrate',reset=True,records={},core=0,fine=0)
    def observe(name,conv,inputs,output):
        x=inputs[0].detach();assert x.shape[0]==1 and conv.groups==1 and conv.dilation==(1,1) and conv.padding_mode=='zeros'
        if context['phase']=='calibrate':
            value=x.double().square().sum((0,2,3));sums[name]=sums.get(name,0)+value;elements[name]=elements.get(name,0)+x.shape[0]*x.shape[2]*x.shape[3];return
        q=quanta[name];integer=torch.round(x.double()/q).to(torch.int64)
        assert bool(torch.isfinite(x).all()) and int(integer.abs().max())<2**31
        old=None if context['reset'] else previous.get(name);delta=integer if old is None else integer-old
        shape=(x.shape[-2:],output.shape[-2:])
        if name not in fanouts:
            fanouts[name]=(shape,input_fanout(conv,x,output))
        assert fanouts[name][0]==shape;fan=fanouts[name][1]
        dense=output.numel()*conv.in_channels*conv.kernel_size[0]*conv.kernel_size[1]
        record=dict(input_channels=conv.in_channels,output_channels=conv.out_channels,input_hw=list(x.shape[-2:]),output_hw=list(output.shape[-2:]),kernel=list(conv.kernel_size),stride=list(conv.stride),padding=list(conv.padding),dense_mac=dense,
            static_nonzeros=int((integer!=0).sum()),delta_nonzeros=int((delta!=0).sum()),input_elements=x.numel(),output_elements=output.numel(),max_input_integer=int(integer.abs().max()),max_delta_integer=int(delta.abs().max()),
            shadow_static_weighted_mac=int(((integer!=0).to(torch.int64)*fan).sum())*conv.out_channels,shadow_delta_weighted_mac=int(((delta!=0).to(torch.int64)*fan).sum())*conv.out_channels,
            shadow_static_unit_ac=int((integer.abs()*fan).sum())*conv.out_channels,shadow_delta_unit_ac=int((delta.abs()*fan).sum())*conv.out_channels,
            quantization_error_in_quanta=float((x.double()/q-integer.double()).abs().max()))
        assert record['shadow_static_weighted_mac']<=dense and record['shadow_delta_weighted_mac']<=dense and record['quantization_error_in_quanta']<=.500001
        context['records'][name]=record;previous[name]=integer
    for name,m in fine.items():handles.append(m.register_forward_hook(lambda mod,inp,out,name=name:observe(name,mod,inp,out)))
    def count(name,mod,inp,out):
        if context['phase']!='evaluate':return
        mac=inp[0].numel()*mod.out_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1] if isinstance(mod,nn.ConvTranspose2d) else out.numel()*mod.in_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1]
        context['core']+=2*mac
        if name.startswith('observation.'):context['fine']+=2*mac
    for name,m in model.core.named_modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):handles.append(m.register_forward_hook(lambda mod,inp,out,name=name:count(name,mod,inp,out)))
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        calibration_frames=[r['frame'] for r in calibration_rows],evaluation_frames=[r['frame'] for r in evaluation_rows],relative_quantum=.125,floor=1e-4,precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary='Fine Conv input RMS calibrated on fixed train512; shadow current integer and delta budgets on disjoint train256 using unchanged ANN features, reset history only at scene/gap/start. No integer output replaces ANN; not quantized own-state execution and not a quality prediction. Compare sparse current-input ANN with delta propagation before designing a new backend. Full dense core actually executes; hooks/diagnostics invalidate latency. No AP/optimizer. Unit AC hardware prices and accumulator storage are unmeasured.')
    write_json(out/'contract.json',contract)
    with torch.no_grad():
        state=None;_,stream=loader(calibration_rows,cfg,2,303)
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));pred,state,info=model.step(ego,state,sample['meta'])
            assert all(bool(torch.isfinite(v).all()) for v in pred.values())
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='calibrate',scans=i+1,total=512))
        assert set(sums)==set(fine)
        for name in fine:quanta[name]=(.125*torch.sqrt(sums[name]/elements[name])).clamp_min(1e-4).reshape(1,-1,1,1)
        write_json(out/'calibration.json',dict(relative_quantum=.125,floor=1e-4,layers={n:dict(sum_squares=sums[n].cpu().tolist(),elements_per_channel=elements[n],quanta=quanta[n].flatten().cpu().tolist()) for n in fine}))
        state=None;context['phase']='evaluate';records=[];_,stream=loader(evaluation_rows,cfg,2,303)
        with (out/'frames.jsonl').open('x',buffering=1) as log:
            for i,sample in enumerate(stream):
                meta=sample['meta'];reason,_=reset_reason(state,meta);context.update(reset=bool(reason),records={},core=0,fine=0);ego=to_device(sample['ego'],torch.device('cuda'))
                pred,state,info=model.step(ego,state,meta);assert all(bool(torch.isfinite(v).all()) for v in pred.values());assert set(context['records'])==set(fine)
                row=dict(frame=meta['frame'],reset=info['reset'],refresh=info['full_refresh'],core_conv_flops=context['core'],fine_conv_flops=context['fine'],cache_bytes=info['cache_bytes'],layers=context['records'])
                assert row['fine_conv_flops']==2*sum(z['dense_mac'] for z in row['layers'].values())
                records.append(row);log.write(json.dumps(row)+'\n')
                if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='evaluate',scans=i+1,total=256))
    for h in handles:h.remove()
    assert model_digest(model)==before
    keys=('dense_mac','shadow_static_weighted_mac','shadow_delta_weighted_mac','shadow_static_unit_ac','shadow_delta_unit_ac')
    budget={n:{k:sum(r['layers'][n][k] for r in records)/256 for k in keys} for n in fine}
    metrics=dict(parameters=sum(p.numel() for p in model.parameters()),core_conv_flops_mean=sum(r['core_conv_flops'] for r in records)/256,fine_conv_flops_mean=sum(r['fine_conv_flops'] for r in records)/256,cache_bytes=22579200,model_step_ms_mean=None,ap50=None,ap70=None)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,layer_budgets=budget,total_shadow_budget={k:sum(v[k] for v in budget.values()) for k in keys},weights_unchanged=True,trained=False,ap_evaluated=False,timing_valid=False,actual_sparse_execution=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min frozen fine event budget')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
