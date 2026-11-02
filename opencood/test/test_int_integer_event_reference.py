import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_integer_event_reference import folded_integer_weights,integer_input,dense_integer_reference,sparse_integer_delta


class IntegerEventTests(unittest.TestCase):
    def test_signed_stream_accumulation_is_bit_exact(self):
        torch.manual_seed(303);conv=nn.Conv2d(3,4,3,padding=1);quantum=torch.tensor([.1,.2,.3]);weight,scale=folded_integer_weights(conv,quantum);previous=None;acc=None
        with torch.no_grad():
            for i in range(8):
                values=integer_input(torch.randn(1,3,4,5),quantum);reference,bound=dense_integer_reference(conv,values,weight)
                if previous is None:acc=reference
                else:
                    delta,stats=sparse_integer_delta(conv,values-previous,weight);acc=acc+delta
                self.assertTrue(torch.equal(acc,reference));self.assertLess(bound,2**53);previous=values
        self.assertEqual(weight.dtype,torch.int16);self.assertTrue(bool((scale>0).all()))

    def test_nonlinear_two_layer_stream_and_overflow_guard(self):
        torch.manual_seed(303);convs=[nn.Conv2d(2,3,3,padding=1),nn.Conv2d(3,2,1)];quanta=[torch.tensor([.125]*2),torch.tensor([.125]*3)];weights=[folded_integer_weights(c,q) for c,q in zip(convs,quanta)];old=[None,None];acc=[None,None]
        with torch.no_grad():
            for step in range(5):
                a=b=torch.randn(1,2,3,4)
                for i,(conv,q,(w,scale)) in enumerate(zip(convs,quanta,weights)):
                    na=integer_input(a,q);nb=integer_input(b,q);self.assertTrue(torch.equal(na,nb));reference,_=dense_integer_reference(conv,na,w)
                    if old[i] is None:acc[i]=reference
                    else:delta,_=sparse_integer_delta(conv,nb-old[i],w);acc[i]=acc[i]+delta
                    self.assertTrue(torch.equal(reference,acc[i]));old[i]=nb
                    a=torch.relu((reference.double()*scale.reshape(1,-1,1,1)).float()+conv.bias.reshape(1,-1,1,1));b=torch.relu((acc[i].double()*scale.reshape(1,-1,1,1)).float()+conv.bias.reshape(1,-1,1,1))
            with self.assertRaises(ValueError):dense_integer_reference(convs[0],torch.full((1,2,3,4),2**52,dtype=torch.int64),weights[0][0])


if __name__=='__main__':unittest.main()
