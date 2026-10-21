"""CPU checks for the byte/gradient bridge, not detector training evidence."""
import unittest
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_content_message import CountContentBottleneck
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire
from opencood.models.fuse_modules.e3dsnn_snapshot_selection import select_training_snapshot
from opencood.tools.test_e3dsnn_task_snapshot import meta
from opencood.utils import task_snapshot_packet as wire


class SnapshotTrainingTest(unittest.TestCase):
    def test_paid_forward_and_project_decode_gradients(self):
        for family in ('if8', 'ann2bit'):
            torch.manual_seed(21)
            model = CountContentBottleneck(family='count3' if family == 'if8' else family)
            features = torch.rand(1, 128, 1, 1, 4) * 4 + 1
            if family == 'if8':
                stream = integrate_and_fire(torch.relu(model.project(features)), 8)
                full = stream.prefix(len(stream.addresses));scales = torch.ones(1,32,1,1,1)
                kwargs = dict(stream=stream)
            else:
                message = model.encode(features);full,scales = message.counts,message.scales
                kwargs = dict(row_order=np.arange(4))
            m = meta(family)
            result = select_training_snapshot(full,scales,m,None,**kwargs)
            recovered,paid = wire.decode(result.packet,m)
            self.assertTrue(torch.equal(result.counts.detach(),full.detach()))
            np.testing.assert_array_equal(recovered,result.counts.detach().numpy())
            np.testing.assert_array_equal(paid,scales.detach().numpy().reshape(32))
            if family == 'ann2bit':
                events = result.ann_events()
                self.assertTrue(all(torch.equal(e.detach(),(full.detach()>=k).float()) for k,e in enumerate(events,1)))
                # Keep the original three-level readout order and gradients.
                latent = sum(e*result.scales for e in events)
            else:
                latent = result.counts
            model.decode(latent).square().mean().backward()
            for p in (model.project.weight,model.expand.weight):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(float(p.grad.abs().sum()),0)

    def test_partial_budget_and_silence(self):
        for family in ('if8','ann2bit'):
            full = torch.ones(1,32,1,1,4,requires_grad=True)
            scales = torch.ones(1,32,1,1,1)
            stream = integrate_and_fire(full,8)
            kwargs = dict(stream=stream) if family=='if8' else dict(row_order=np.arange(4))
            m = meta(family)
            first = np.zeros((1,32,1,1,4),dtype=np.uint8)
            if family=='if8':first.ravel()[int(stream.addresses[0])]=1
            else:first[:,:,:,:,0]=1
            cap = len(wire.encode(first,np.ones(32,dtype=np.float32),m))
            selected = select_training_snapshot(full,scales,m,cap,**kwargs)
            self.assertGreater(len(selected.packet),0)
            self.assertLessEqual(len(selected.packet),cap)
            self.assertLess(float(selected.counts.sum()),float(full.sum()))
            silence = select_training_snapshot(full,scales,m,0,**kwargs)
            self.assertEqual(silence.packet,b'')
            self.assertEqual(float(silence.counts.sum()),0)
            self.assertTrue(torch.equal(silence.scales,scales))
            silence.counts.sum().backward()
            self.assertEqual(float(full.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
