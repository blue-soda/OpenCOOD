import unittest

import numpy as np
import torch

from opencood.models.sub_modules.int_silent_drive import DriveFit, SilentDrive


class SilentDriveTest(unittest.TestCase):
    def test_fold_coefficients_and_errors_agree_with_explicit_design(self):
        rng=np.random.RandomState(303)
        model=DriveFit()
        samples=[]
        for scene in range(4):
            h,w=8,32
            old=torch.from_numpy(rng.normal(size=(1,256,h,w)).astype('f'))
            fine=torch.from_numpy(rng.normal(size=(1,128,2*h,2*w)).astype('f'))
            coarse=fine[0,::16,::2,::2]
            full=old + .2*old + .3 + .1*coarse[:1]
            mask=torch.ones(1,1,h,w,dtype=torch.bool)
            model.observe(str(scene),fine,old,mask,full)
            ids=np.arange(h*w)[::16]
            # Check a single channel independently against a conventional row-major design matrix.
            x=np.column_stack((old[0,17].flatten().numpy()[ids],np.ones(len(ids)),coarse.flatten(1).numpy().T[ids]))
            y=(full[0,17].flatten().double()-old[0,17].flatten().double()).numpy()[ids]
            samples.append((x,y))
        fit=model.finish()
        for mode,size in [('old',2),('drive',10)]:
            x=np.concatenate([v[0][:,:size] for v in samples])
            y=np.concatenate([v[1] for v in samples])
            gram=x.T@x
            expected=np.linalg.solve(gram+.001*np.diag(gram.diagonal()),x.T@y)
            np.testing.assert_allclose(expected,fit['coefficients'][mode][17],atol=1e-8)
        self.assertTrue(all(v['predicted_sse']<v['hold_sse'] for v in fit['folds']['drive'].values()))

    @unittest.skipUnless(torch.cuda.is_available(),'the deployment kernel runs on CUDA')
    def test_only_newly_held_cells_change_and_cost_counts_match(self):
        old=torch.ones(1,256,2,3,device='cuda')
        fine=torch.ones(1,128,4,6,device='cuda')
        weight=torch.zeros(256,10)
        weight[:,0]=-.5
        weight[:,2:]=.125
        mask=torch.zeros(1,1,2,3,device='cuda',dtype=torch.bool)
        mask[:,:,0,1]=True
        output=old.clone()
        model=SilentDrive({'drive':weight},'drive')
        ops=model.apply(fine,old,mask,output)
        self.assertEqual(ops,19*256)
        self.assertTrue(torch.equal(output,old+.5*mask))
        self.assertEqual(model.apply(fine,old,mask&False,output),0)
