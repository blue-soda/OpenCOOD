import copy
import tempfile
import unittest
from pathlib import Path
import torch
from opencood.tools.int_readout_warm_start import load_warm_start


class TestReadoutWarmStart(unittest.TestCase):
    def test_weights_only_period_transfer_and_identity_rejection(self):
        source=torch.nn.Linear(3,2);target=torch.nn.Linear(3,2)
        contract=dict(kind='current',source={'data':'frozen'},frozen_deep_checkpoint='D',
                      width=3,deep_period=4,readout_period=3,trainable_parameters=8,architecture='fixture')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'repair.pth'
            torch.save(dict(repair=source.state_dict(),contract=contract,epoch=9,optimizer={'must_not_restore':True}),str(path))
            destination=dict(contract,readout_period=5)
            info=load_warm_start(target,path,destination)
            self.assertEqual((info['epoch'],info['source_period'],info['target_period']),(9,3,5))
            self.assertEqual(len(info['sha256']),64)
            for a,b in zip(source.parameters(),target.parameters()):self.assertTrue(torch.equal(a,b))
            for key,value in [('source',{'data':'different'}),('frozen_deep_checkpoint','wrong'),('width',4),('deep_period',6),('architecture','other'),('kind','current_gated')]:
                wrong=copy.deepcopy(destination);wrong[key]=value
                with self.assertRaises(AssertionError):load_warm_start(target,path,wrong)
