import unittest
import numpy as np
import torch
from opencood.utils import residual_spike_packet as wire
from opencood.models.fuse_modules.e3dsnn_residual_stream import signed_if,ResidualSender,ResidualReceiver


class ResidualTests(unittest.TestCase):
    def meta(self):return dict(channels=2,shape_zyx=[2,3,4],steps=16,frame='a',basis='fixed',pose=[1,0,0])

    def test_wire_signed_steps_budget_identity(self):
        m=self.meta();a=[(0,3,1),(0,7,-1),(1,3,-1),(2,8,3)]
        p=wire.encode(a,m);np.testing.assert_array_equal(wire.decode(p,m),a)
        with self.assertRaises(ValueError):wire.decode(p,dict(m,frame='b'))
        with self.assertRaises(ValueError):wire.decode(p[:-1]+bytes([p[-1]^1]),m)
        with self.assertRaises(ValueError):wire.encode([(0,1,1),(0,1,-1)],m)
        events,packet,n,stopped=wire.append_prefix([],a,m,len(p)-1)
        self.assertLessEqual(len(packet),len(p)-1)
        self.assertEqual(events,a[:n]);self.assertTrue(stopped)
        self.assertEqual(wire.append_prefix([],a,m,0)[1],b'')

    def test_subtractive_conservation(self):
        v=torch.tensor([.4,-.6,1.7,-2.1],requires_grad=True)
        original=v.clone();total=torch.zeros_like(v)
        for _ in range(3):
            s=signed_if(v);v=v-s;total=total+s
        torch.testing.assert_close(v+total,original)
        (v+total).sum().backward();self.assertTrue(torch.isfinite(original.grad_fn.next_functions[0][0].variable.grad).all())

    def test_selected_large_events_have_task_surrogate(self):
        from opencood.models.fuse_modules.e3dsnn_residual_stream import ann_quant
        for function in (signed_if,ann_quant):
            x=torch.tensor([-20.,20.],requires_grad=True)
            function(x).sum().backward()
            self.assertTrue(torch.equal(x.grad,torch.ones_like(x)))

    def test_sender_feedback_and_gradients(self):
        torch.manual_seed(1);sender=ResidualSender(4,2)
        x=torch.randn(1,4,2,3,4,requires_grad=True)
        reconstruction=torch.zeros_like(x);potential=torch.zeros(1,2,2,3,4)
        s,v=sender.step(x,reconstruction,potential,'residual_snn')
        reconstruction=sender.decode(s);s2,_=sender.step(x,reconstruction,v-s,'residual_snn')
        reconstruction= reconstruction+sender.decode(s2)
        reconstruction.square().mean().backward()
        for p in sender.parameters():self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
        self.assertFalse(torch.equal(sender.drive(x),sender.drive(x-reconstruction)))

    def test_exact_hard_symbols_at_extreme_finite_values(self):
        from opencood.models.fuse_modules.e3dsnn_residual_stream import ann_quant,select_symbols
        for dtype in (torch.float32,torch.float64):
            for function in (signed_if,ann_quant):
                x=torch.tensor([-1e30,-2**25,-1.1,-.5,0.,.5,1.1,2**25,1e30],dtype=dtype,requires_grad=True)
                expected=torch.sign(x)*(x.abs()>=1) if function is signed_if else x.clamp(-3,3).round()
                y=function(x)
                self.assertTrue(torch.equal(y,expected))
                y.sum().backward();self.assertTrue(torch.equal(x.grad,torch.ones_like(x)))
                events=select_symbols(y,x.abs(),len(x),0)
                meta=dict(channels=1,shape_zyx=[1,1,len(x)],steps=16)
                np.testing.assert_array_equal(wire.decode(wire.encode(events,meta),meta),events)

    def test_reject_invalid_symbols_before_integer_cast(self):
        from opencood.models.fuse_modules.e3dsnn_residual_stream import select_symbols
        for invalid in (.1,3.5,4.):
            with self.assertRaises(ValueError):select_symbols(torch.tensor([invalid]),torch.ones(1),1,0)
        with self.assertRaises(FloatingPointError):select_symbols(torch.tensor([float('nan')]),torch.ones(1),1,0)

    def test_receiver_zero_and_frame_reset(self):
        torch.manual_seed(2);receiver=ResidualReceiver(4,2)
        local=torch.randn(1,4,2,3,4);zero=torch.zeros_like(local);mask=torch.zeros_like(local[:,:1])
        for family in ('residual_snn','residual_ann'):
            state=receiver.begin(local);out=receiver.receive(state,local,zero,mask,family)
            self.assertTrue(torch.equal(out,zero))
            road=torch.randn_like(local);mask=torch.ones_like(mask)
            a=receiver.receive(receiver.begin(local),local,road,mask,family)
            b=receiver.receive(receiver.begin(local),local,road,mask,family)
            self.assertTrue(torch.equal(a,b));mask.zero_()

    def test_local_context_does_not_cancel(self):
        torch.manual_seed(3);r=ResidualReceiver(4,2)
        zero=torch.zeros(1,4,2,3,4);road=torch.randn_like(zero);support=torch.ones_like(zero[:,:1])
        a=r.context(zero,road,support)-r.context(zero,zero,support*0)
        local=torch.ones_like(zero)*2
        b=r.context(local,road,support)-r.context(local,zero,support*0)
        self.assertFalse(torch.allclose(a,b))


if __name__=='__main__':unittest.main()
