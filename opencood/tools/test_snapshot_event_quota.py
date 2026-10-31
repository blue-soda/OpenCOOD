"""Codec-free IF prefix values and gradients; byte measurements are never fabricated."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import torch
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire
from opencood.models.fuse_modules.e3dsnn_snapshot_selection import select_event_quota


class EventQuotaTest(unittest.TestCase):
    def test_values_gradients_and_no_codec(self):
        for quota in (0,1,30,10000,None):
            torch.manual_seed(19)
            drive=(torch.rand(1,32,1,1,4)*8).requires_grad_()
            stream=integrate_and_fire(drive,8)
            n=len(stream.addresses) if quota is None else min(quota,len(stream.addresses))
            expected=stream.prefix(n)
            with ExitStack() as stack:
                for name in ('select','encode','decode'):
                    stack.enter_context(patch('opencood.utils.task_snapshot_packet.'+name,side_effect=AssertionError('codec called')))
                result=select_event_quota(stream,quota)
            self.assertTrue(torch.equal(expected,result.counts));self.assertIsNone(result.packet)
            self.assertFalse(result.record['communication_measured'])
            self.assertNotIn('actual_bytes',result.record)
            self.assertEqual(result.record['selected_events'],n)
            wanted=torch.autograd.grad(expected.square().sum(),drive,retain_graph=True)[0]
            actual=torch.autograd.grad(result.counts.square().sum(),drive)[0]
            self.assertTrue(torch.equal(actual,wanted))

    def test_empty_invalid_and_nested_prefixes(self):
        stream=integrate_and_fire(torch.zeros(1,32,1,1,4),8)
        self.assertEqual(select_event_quota(stream,19000).record['selected_events'],0)
        for bad in (-1,1.5,True):
            with self.assertRaises(ValueError):select_event_quota(stream,bad)
        stream=integrate_and_fire(torch.ones(1,32,1,1,4)*3,8)
        self.assertTrue(torch.all(select_event_quota(stream,50).counts<=select_event_quota(stream,100).counts))


if __name__=='__main__':unittest.main()
