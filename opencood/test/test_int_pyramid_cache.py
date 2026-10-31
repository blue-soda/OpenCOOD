import unittest
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace
from opencood.models.sub_modules.codyntrust_base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.codyntrust_downsample_conv import DownsampleConv
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_cache import CachedPyramidDetector, PackedPyramidStage
from opencood.models.sub_modules.int_cache_repair import refresh_tiles


def fixture():
    torch.manual_seed(303)
    source=SimpleNamespace(pillar_vfe=nn.Identity(),scatter=nn.Identity(),use_dir=True,shrink_flag=True,
        backbone=ResNetBEVBackbone(dict(layer_nums=[3,4,5],layer_strides=[2,2,2],num_filters=[128,256,512],
            upsample_strides=[1,2,4],num_upsample_filter=[128,128,128])),
        shrink_conv=DownsampleConv(dict(input_dim=384,kernal_size=[3],dim=[256],stride=[1],padding=[1])),
        cls_head=nn.Conv2d(256,2,1),reg_head=nn.Conv2d(256,14,1),dir_head=nn.Conv2d(256,4,1))
    model=PyramidStateDetector(source,{'model':{'args':{'lidar_range':[-4,-4,-1,4,4,1]}}})
    for name in ('fine_update','coarse_update','deep_update'):
        nn.init.normal_(getattr(model.core,name).history.weight,std=.001)
        nn.init.normal_(getattr(model.core,name).residual[-1].weight,std=.001)
    for module in model.modules():
        if isinstance(module,nn.BatchNorm2d):module.running_mean.fill_(.1);module.bias.data.fill_(.03)
    return model


class PyramidCacheTests(unittest.TestCase):
    def test_anchor_period_one_matches_original(self):
        from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
        base=fixture();cached=AnchoredLatePyramidDetector(base,period=1);a=b=None
        with torch.no_grad():
            for i in range(4):
                x=torch.randn(1,64,24,32);pose=np.eye(4);pose[0,3]=i*.1
                meta=dict(timestamp_us=i*100000,segment='a',pose=pose)
                expected,a,_=base.step_spatial(x,a,meta);actual,b,info=cached.step_spatial(x,b,meta)
                for k in expected:self.assertTrue(torch.equal(actual[k],expected[k]))
                self.assertTrue(torch.equal(a['deep'],b['anchor']['value']))
                self.assertTrue(info['late_refresh'])

    def test_anchor_reuse_skips_deep_compute_and_keeps_original_pose(self):
        from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
        from opencood.models.sub_modules.int_feature_memory import warp_bev
        from opencood.models.sub_modules.int_block_cache import block_lidar_range
        base=fixture();cached=AnchoredLatePyramidDetector(base,period=3,coverage='zero');calls=[]
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=cached.step_spatial(x,None,meta);initial=state['anchor'];outputs=[]
            handles=[base.core.deep_encoder.register_forward_hook(lambda *_:calls.append(1)),
                     base.core.deep_update.register_forward_hook(lambda *_:calls.append(1)),
                     base.core.deblocks[2].register_forward_pre_hook(lambda m,args:outputs.append(args[0]))]
            try:
                for i in (1,2):
                    pose=np.eye(4);pose[0,3]=i*.4
                    _,state,info=cached.step_spatial(x,state,dict(meta,timestamp_us=i*100000,pose=pose))
                    self.assertIsNone(info['late_refresh']);self.assertEqual(calls,[])
                    self.assertIs(state['anchor']['value'],initial['value'])
                    self.assertTrue(np.array_equal(state['anchor']['pose'],initial['pose']))
                    bounds=block_lidar_range(cached.bounds,x.shape[-2:],initial['value'].shape[-2:],8)
                    expected,_=warp_bev(initial['value'],initial['pose'],pose,bounds)
                    self.assertTrue(torch.equal(outputs[-1],expected))
                _,_,info=cached.step_spatial(x,state,dict(meta,timestamp_us=250000))
                self.assertEqual(info['late_refresh'],'period');self.assertEqual(len(calls),2)
            finally:
                for handle in handles:handle.remove()

    def test_anchor_coverage_and_age_guards(self):
        from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
        base=fixture();strict=AnchoredLatePyramidDetector(base,period=3,coverage='strict')
        age=AnchoredLatePyramidDetector(base,period=3,coverage='zero',max_age=.15)
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=strict.step_spatial(x,None,meta);pose=np.eye(4);pose[0,3]=2
            _,_,info=strict.step_spatial(x,state,dict(meta,timestamp_us=100000,pose=pose))
            self.assertEqual(info['late_refresh'],'coverage')
            _,_,info=age.step_spatial(x,state,dict(meta,timestamp_us=160000))
            self.assertEqual(info['late_refresh'],'age')
            _,_,info=age.step_spatial(x,state,dict(meta,timestamp_us=100000,segment='b'))
            self.assertEqual(info['late_refresh'],'scene')
            with self.assertRaises(ValueError):age.step_spatial(x,state,meta)

    def test_same_plan_dense_control_preserves_writes_and_planned_work(self):
        from opencood.models.sub_modules.int_pyramid_regions import coalesced_refresh
        from opencood.models.sub_modules.int_pyramid_block_cache import LocalBlock
        model=fixture();block=LocalBlock(model.core.observation[1]).eval()
        x=torch.randn(1,128,48,64);old=torch.randn_like(x);request=torch.zeros_like(x[:,:1],dtype=torch.bool)
        request[...,20:24,20:24]=True
        with torch.no_grad():
            actual,written,stats=coalesced_refresh(block,x,old,request,stride=1)
            expected,mask,control=coalesced_refresh(block,x,old,request,stride=1,dense_reference=True)
        self.assertTrue(torch.equal(mask,written))
        self.assertTrue(torch.allclose(actual,expected,atol=1e-4,rtol=1e-4))
        self.assertEqual(stats['execution_tile'],control['execution_tile'])
        self.assertEqual(stats['stage_conv_ratio'],control['planned_stage_conv_ratio'])
        self.assertLess(stats['stage_conv_ratio'],control['stage_conv_ratio'])
        self.assertTrue(torch.equal(expected[~mask.expand_as(expected)],old[~mask.expand_as(old)]))

    def test_compact_late_cache_preserves_outputs_and_reduces_live_state(self):
        from opencood.models.sub_modules.int_pyramid_block_cache import BlockCachedPyramidDetector
        model=fixture();options=dict(force_full_blocks=list(range(9)),fraction=.1,tile=4,execution='coalesced')
        old=BlockCachedPyramidDetector(model,**options)
        compact=BlockCachedPyramidDetector(model,compact_dense=True,**options)
        a=b=None
        with torch.no_grad():
            for i in range(8):
                pose=np.eye(4);pose[0,3]=i*.12
                meta=dict(timestamp_us=i*100000,segment='a' if i<6 else 'b',pose=pose)
                x=torch.randn(1,64,64,96)
                expected,a,ai=old.step_spatial(x,a,meta)
                actual,b,bi=compact.step_spatial(x,b,meta)
                for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]))
                for key in old.state_keys:self.assertTrue(torch.equal(a[key],b[key]))
                self.assertTrue(all(c is None for c in b['blocks'][:9]))
                for c,d in zip(a['blocks'][9:],b['blocks'][9:]):
                    for key in c:self.assertTrue(torch.equal(c[key],d[key]))
                self.assertLess(bi['cache_bytes'],ai['cache_bytes'])
                tensors=[v for c in b['blocks'][9:] for v in c.values()]+[b['value'],b['coarse']]
                self.assertEqual(bi['cache_bytes'],sum(v.numel()*v.element_size() for v in tensors))
        with self.assertRaises(ValueError):BlockCachedPyramidDetector(model,compact_dense=True)

    def test_forced_blocks_only_bypass_their_own_selection(self):
        from opencood.models.sub_modules.int_pyramid_block_cache import BlockCachedPyramidDetector
        model=fixture();indices=[i for i in range(15) if i not in (3,8,14)]
        cached=BlockCachedPyramidDetector(model,fraction=0,force_full_blocks=indices,execution='coalesced')
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=cached.step_spatial(x,None,meta)
            _,_,info=cached.step_spatial(x,state,dict(meta,timestamp_us=100000))
        self.assertEqual([s['actual_write_fraction'] for s in info['cache_stages']],
                         [1. if i in indices else 0. for i in range(15)])
        with self.assertRaises(ValueError):BlockCachedPyramidDetector(model,force_full_blocks=[15])

    def test_block_cache_full_refresh_matches_original(self):
        from opencood.models.sub_modules.int_pyramid_block_cache import BlockCachedPyramidDetector
        model=fixture();cached=BlockCachedPyramidDetector(model,fraction=1,execution='coalesced')
        a=b=None
        with torch.no_grad():
            for i in range(3):
                pose=np.eye(4);pose[0,3]=i*.2
                meta=dict(timestamp_us=i*100000,segment='a',pose=pose)
                x=torch.randn(1,64,24,32)
                pred,a,_=model.step_spatial(x,a,meta)
                actual,b,info=cached.step_spatial(x,b,meta)
                self.assertEqual(len(b['blocks']),15)
                self.assertEqual([m.radius for m,_,_ in cached.blocks],[3,2,2,2,3,2,2,2,2,3,2,2,2,2,2])
                for key in pred:self.assertTrue(torch.equal(pred[key],actual[key]))
                for key in model.state_keys:self.assertTrue(torch.equal(a[key],b[key]))
                self.assertGreater(info['cache_bytes'],info['state_bytes'])

    def test_block_cache_same_mask_matches_dense_on_moving_stream(self):
        from opencood.models.sub_modules.int_pyramid_block_cache import BlockCachedPyramidDetector
        model=fixture()
        sparse=BlockCachedPyramidDetector(model,fraction=.1,tile=2,max_cache_age=.2,execution='coalesced')
        dense=BlockCachedPyramidDetector(model,fraction=.1,tile=2,max_cache_age=.2,
            execution='masked_dense',allow_fallback=False)
        state=None
        with torch.no_grad():
            for i in range(4):
                pose=np.eye(4);theta=i*.01
                pose[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
                pose[0,3]=i*.15
                meta=dict(timestamp_us=i*100000,segment='a',pose=pose)
                x=torch.randn(1,64,64,96)
                actual,new,_=sparse.step_spatial(x,state,meta)
                expected,reference,_=dense.step_spatial(x,state,meta,requests=sparse.last_requests)
                for k in actual:self.assertTrue(torch.allclose(actual[k],expected[k],atol=1e-4,rtol=1e-4))
                for c,d in zip(new['blocks'],reference['blocks']):
                    for k in c:self.assertTrue(torch.allclose(c[k],d[k],atol=1e-4,rtol=1e-4))
                    self.assertLess(float(c['age'].max()),.2)
                state=new
            with self.assertRaises(ValueError):sparse.step_spatial(x,state,meta)

    def test_block_cache_skips_all_blocks_when_no_requested_work(self):
        from opencood.models.sub_modules.int_pyramid_block_cache import BlockCachedPyramidDetector
        model=fixture();cached=BlockCachedPyramidDetector(model,fraction=0,execution='coalesced')
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=cached.step_spatial(x,None,meta)
            calls=[];hooks=[m.register_forward_hook(lambda *_:calls.append(1)) for m,_,_ in cached.blocks]
            try:_,new,info=cached.step_spatial(x,state,dict(meta,timestamp_us=100000))
            finally:
                for h in hooks:h.remove()
            self.assertEqual(calls,[])
            for c,d in zip(state['blocks'],new['blocks']):self.assertTrue(torch.equal(c['value'],d['value']))
            self.assertTrue(all(s['executed_input_area_ratio']==0 for s in info['cache_stages']))

    def test_coalescing_preserves_sparse_write_mask_and_bounds_work(self):
        from opencood.models.sub_modules.int_pyramid_regions import coalesced_refresh
        model=fixture();stage=PackedPyramidStage(model.core.observation,model.core.fine_update).eval()
        x=torch.randn(1,64,129,161);old=torch.randn(1,128,65,81);valid=torch.ones_like(old[:,:1],dtype=torch.bool)
        with torch.no_grad():
            packed=stage.pack(x,old,valid)
            for mode in ('empty','cluster','fragmented','full'):
                request=torch.zeros_like(valid)
                if mode=='cluster':request[...,20:25,20:25]=True;request[...,28:33,20:25]=True
                if mode=='fragmented':request[...,::8,::8]=True;request[...,-1,-1]=True
                if mode=='full':request.fill_(True)
                actual,written,stats=coalesced_refresh(stage,packed,old,request,tile=4)
                expected,mask,_=refresh_tiles(stage,packed,old,request,2,stage.radius,4,
                    allow_fallback=False,execution='masked_dense')
                self.assertTrue(torch.equal(written,mask))
                self.assertTrue(torch.allclose(actual,expected,atol=1e-4,rtol=1e-4))
                self.assertTrue(torch.equal(actual[~mask.expand_as(actual)],old[~mask.expand_as(old)]))
                self.assertLessEqual(stats['stage_conv_ratio'],1.)
                self.assertLessEqual(stats['stage_conv_ratio'],stats['original_stage_conv_ratio'])
                if mode=='empty':self.assertEqual(stats['tiles'],0)

    def test_stage_sparse_matches_same_mask_dense_at_edges_and_odd_sizes(self):
        model=fixture();stage=PackedPyramidStage(model.core.observation,model.core.fine_update).eval()
        self.assertEqual(stage.radius,15)
        x=torch.randn(1,64,65,81);old=torch.randn(1,128,33,41);valid=torch.ones_like(old[:,:1],dtype=torch.bool)
        request=torch.zeros_like(valid);request[...,0,0]=True;request[...,16,20]=True;request[...,-1,-1]=True
        with torch.no_grad():
            packed=stage.pack(x,old,valid)
            actual,written,_=refresh_tiles(stage,packed,old,request,2,stage.radius,4,allow_fallback=False)
            expected,mask,_=refresh_tiles(stage,packed,old,request,2,stage.radius,4,allow_fallback=False,execution='masked_dense')
        self.assertTrue(torch.equal(written,mask))
        self.assertTrue(torch.allclose(actual,expected,atol=1e-4,rtol=1e-4))
        self.assertTrue(torch.equal(actual[~written.expand_as(actual)],old[~written.expand_as(old)]))

    def test_full_refresh_matches_dense_recurrent_model(self):
        base=fixture();cached=CachedPyramidDetector(base,fraction=1)
        x=torch.randn(1,64,24,32);a=b=None;pose=np.eye(4)
        with torch.no_grad():
            for stamp in (100000,200000,300000):
                meta=dict(segment='a',timestamp_us=stamp,pose=pose)
                expected,a,_=base.step_spatial(x,a,meta);actual,b,_=cached.step_spatial(x,b,meta)
                self.assertTrue(all(torch.equal(actual[k],expected[k]) for k in actual))
                self.assertTrue(all(torch.equal(a[k],b[k]) for k in base.state_keys))

    def test_whole_reuse_skips_encoders_and_updates_until_age_guard(self):
        base=fixture();cached=CachedPyramidDetector(base,fraction=0,max_cache_age=.25)
        x=torch.randn(1,64,24,32);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        calls=[]
        with torch.no_grad():
            _,state,_=cached.step_spatial(x,None,meta)
            hooks=[stage.register_forward_pre_hook(lambda m,args:calls.append(tuple(args[0].shape))) for stage in cached.stages]
            _,later,info=cached.step_spatial(x*2,state,dict(meta,timestamp_us=200000))
            self.assertFalse(calls)
            self.assertTrue(all(torch.equal(state[k],later[k]) for k in base.state_keys))
            self.assertTrue(all(s['executed_input_area_ratio']==0 for s in info['cache_stages']))
            cached.step_spatial(x,later,dict(meta,timestamp_us=400000))
            self.assertEqual(len(calls),3)
            for hook in hooks:hook.remove()

    def test_partial_closed_loop_reference_and_reset_guards(self):
        base=fixture();sparse=CachedPyramidDetector(base,fraction=.1,tile=2,allow_fallback=False)
        dense=CachedPyramidDetector(base,fraction=.1,tile=2,allow_fallback=False,execution='masked_dense')
        x=torch.randn(1,64,48,64);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        with torch.no_grad():
            _,state,_=sparse.step_spatial(x,None,meta)
            current=dict(meta,timestamp_us=200000)
            # Use the exact same current input and prior cache for both executors.
            y=x.clone();y[...,16:20,16:20]+=1
            a,an,_=sparse.step_spatial(y,state,current);b,bn,_=dense.step_spatial(y,state,current,requests=sparse.last_requests)
            self.assertTrue(all(torch.allclose(a[k],b[k],atol=1e-4,rtol=1e-4) for k in a))
            self.assertTrue(all(torch.allclose(an[k],bn[k],atol=1e-4,rtol=1e-4) for k in base.state_keys))
            for changed,reason in [(dict(current,segment='b'),'scene'),(dict(current,reset_before=True),'explicit'),(dict(current,timestamp_us=500000),'gap')]:
                _,_,info=sparse.step_spatial(y,state,changed);self.assertEqual(info['reset'],reason)
            with self.assertRaises(ValueError):sparse.step_spatial(y,state,meta)


if __name__=='__main__':unittest.main()
