import unittest
import torch
from opencood.models.sub_modules.int_cached_neighborhoods import CachedNeighborhoods
from opencood.models.sub_modules.int_selected_conv import prepare_selected_neighborhoods


class TestCachedNeighborhoods(unittest.TestCase):
    def test_grid_change_boundaries_and_mask_reuse_exact(self):
        lookup=CachedNeighborhoods()
        for height,width in ((1,1),(1,7),(8,1),(5,9),(9,5)):
            masks=[torch.zeros(1,1,height,width,dtype=torch.bool),torch.ones(1,1,height,width,dtype=torch.bool)]
            scattered=masks[0].clone();scattered[:,:,::2,::3]=True;masks.append(scattered)
            previous=None
            for mask in masks:
                selected,neighbors,shape=lookup(mask)
                reference=prepare_selected_neighborhoods(mask)
                self.assertTrue(torch.equal(selected,reference[0]))
                self.assertTrue(torch.equal(neighbors,reference[1]))
                self.assertEqual(shape,reference[2])
                if previous is not None:self.assertIs(lookup.table,previous)
                previous=lookup.table
                self.assertEqual(lookup.geometry_bytes,height*width*9*8)
            self.assertEqual(lookup.table.shape,(height*width,9))
