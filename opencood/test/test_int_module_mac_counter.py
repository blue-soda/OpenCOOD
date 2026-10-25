import unittest
import torch
from opencood.tools.profile_int_basis_regions import ModuleMacCounter
from opencood.tools.train_int_anchor_residual import ConvCounter


class ModuleMacTests(unittest.TestCase):
    def test_unique_leaf_attribution_matches_executed_cost(self):
        model=torch.nn.Sequential(torch.nn.Conv2d(4,6,3,padding=1,groups=2),
            torch.nn.ReLU(),torch.nn.ConvTranspose2d(6,2,2,stride=2))
        x=torch.randn(1,4,7,9)
        counter=ModuleMacCounter(dict(model=model,alias=model[0]))
        reference=ConvCounter(model,torch.nn.ModuleList())
        try:
            model(x)
            expected=7*9*6*2*9+7*9*6*2*4
            self.assertEqual(sum(counter.macs.values()),expected)
            self.assertEqual(sum(counter.macs.values()),reference.total)
            self.assertEqual(set(counter.macs),{'model.0','model.2'})
        finally:counter.close();reference.close()
        model(x)
        self.assertEqual(sum(counter.macs.values()),expected)


if __name__=='__main__':unittest.main()
