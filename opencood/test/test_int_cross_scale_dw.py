import unittest

import torch
from torch import nn

from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleDepthwise


def route(h, w, shift):
    source = torch.arange(w).expand(h,w)-shift
    invalid = (source<0)|(source>=w)
    ids = source.clamp(0,w-1)+w*torch.arange(h)[:,None]
    return ids.flatten(), invalid.view(1,1,h,w)


class CrossScaleTest(unittest.TestCase):
    def test_geometry_and_signed_updates(self):
        torch.manual_seed(303)
        conv = nn.Conv2d(2,2,3,2,1,groups=2)
        edges = [CrossScaleDepthwise(conv,m) for m in ('dense','amplitude','unit')]
        caches = [None]*3
        previous = torch.zeros(1,2,7,9,dtype=torch.long)
        full = (0,0,7,9)
        for step,(shift,box) in enumerate([(0,full),(0,full),(2,full),(1,full),(0,(2,2,7,9)),(0,full),(0,full)]):
            ir, out = route(7,9,shift), route(4,5,shift//2)
            aligned = previous.flatten(2).index_select(2,ir[0]).reshape_as(previous).masked_fill(ir[1],0)
            delta = torch.randint(-3,4,previous.shape) if step<6 else torch.zeros_like(previous)
            current = aligned+delta
            results = []
            for i,e in enumerate(edges):
                e.begin(current,delta,ir,out,caches[i])
                results.append(e.region(box))
                caches[i] = e.end()
                self.assertTrue(torch.equal(results[0],results[-1]))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(caches[0],caches[i])))
            counts = edges[2].counts
            if step in (0,3):
                self.assertEqual(counts['reuse'],0)
            if step in (1,2):
                self.assertGreater(counts['positive_units'],0)
                self.assertGreater(counts['negative_units'],0)
            if step == 4:
                self.assertLess(counts['complete'],counts['sites'])
            if step == 5:
                self.assertTrue(0<counts['reuse']<counts['sites'])
            if step == 6:
                self.assertEqual(counts['reuse'],counts['sites'])
                self.assertEqual(counts['delta_nonzero_taps'],0)
            previous = current

