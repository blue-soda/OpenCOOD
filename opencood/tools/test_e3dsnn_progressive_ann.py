"""ANN control prerequisites: fixed-scale messages and continuous readout."""
import unittest
import numpy as np
import torch
from opencood.utils import quantized_feature_packet as codec
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract,ProgressiveMembraneFusion


class AnnPrerequisiteTest(unittest.TestCase):
    def test_fragment_scale_is_preserved_not_requantized(self):
        coords=np.array([[0,0,i] for i in range(3)])
        features=np.tile(np.array([.4,.8,1.2],dtype=np.float32)[:,None],(1,128))
        options=dict(bits=2,keep_fraction=1.)
        selected,codes,scales=codec.prepare(coords,features,'relu',options)
        meta={'shape_zyx':[1,1,3]}
        decoded=[]
        for i in range(3):
            packet=codec.encode_prepared(selected[i:i+1],codes[i:i+1],scales,meta,'relu',options)
            c,v,_=codec.decode(packet,'relu',options)
            np.testing.assert_array_equal(c,selected[i:i+1]);np.testing.assert_array_equal(v,codes[i:i+1]*scales)
            decoded.append(v)
        np.testing.assert_array_equal(np.concatenate(decoded),codes.astype(np.float32)*scales)
        # Requantizing a prefix with a different maximum really changes values.
        bad=codec.encode(selected[:2],codes[:2]*scales,meta,'relu',options)
        _,bad_values,_=codec.decode(bad,'relu',options)
        self.assertFalse(np.array_equal(bad_values,codes[:2]*scales))
        self.assertEqual(codec.encode(coords,features,meta,'relu',options),
            codec.encode_prepared(selected,codes,scales,meta,'relu',options))

    def test_empty_and_invalid_prepared_codes(self):
        options=dict(bits=2,keep_fraction=1.);coords=np.empty((0,3),dtype=np.uint16)
        codes=np.empty((0,128),dtype=np.uint8);scales=np.ones(128,dtype=np.float32)
        packet=codec.encode_prepared(coords,codes,scales,{'shape_zyx':[1,1,3]},'relu',options)
        c,v,_=codec.decode(packet,'relu',options);self.assertEqual(v.shape,(0,128))
        for bad in (np.ones((1,128))*.1,np.ones((1,128))*4,np.zeros((1,128))):
            with self.assertRaises(ValueError):codec.encode_prepared(np.array([[0,0,0]]),bad,scales,{},'relu',options)

    def test_ann_continuous_currents_and_inhibition(self):
        c=MessageContract('v','r','infra','a'*64,'ann-output-v1',(1,1,1),(0.,0.,0.),(1.,1.,1.),(1,2))
        pose=torch.eye(4,dtype=torch.double)[None];ego=torch.tensor([[[[[-.4]]]]],dtype=torch.double)
        m=ProgressiveMembraneFusion(1,readout='relu').double()
        state=m.begin(ego,c,pose,c.encoder_sha256,c.channel_basis)
        m.receive(state,torch.full_like(ego,.25),c,pose,0)
        self.assertEqual(float(state.output),0.);self.assertAlmostEqual(float(state.residual),-.15)
        m.receive(state,torch.full_like(ego,.35),c,pose,1)
        self.assertAlmostEqual(float(m.finish(state)),.2)
        # Count4's integer wire constraint remains enforced by the default path.
        snn=ProgressiveMembraneFusion(1).double();state=snn.begin(ego,c,pose,c.encoder_sha256,c.channel_basis)
        with self.assertRaises(ValueError):snn.receive(state,torch.full_like(ego,.25),c,pose,0)


if __name__=='__main__':unittest.main()
