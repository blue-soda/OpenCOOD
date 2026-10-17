import unittest
import torch
from opencood.tools.int_cache_gate_calibration import cache_descriptor, descriptor_change, replay_gate


class CacheGateTests(unittest.TestCase):
    def test_descriptor_shape_identity_scale_and_change(self):
        x=torch.ones(1,256,50,126)
        a=cache_descriptor(x)
        self.assertEqual(tuple(a.shape),(1,16,25,63))
        self.assertEqual(descriptor_change(a,a),0.)
        self.assertAlmostEqual(descriptor_change(a,2*a),.5,places=6)
        self.assertAlmostEqual(descriptor_change(3*a,6*a),.5,places=6)

    def test_replay_keeps_refresh_reference_and_obeys_guards(self):
        pair=lambda score,valid=True,age=.1:dict(score=score,valid=valid,age=age,target_relative_error=.2)
        rows=[dict(reset=True,lags={}),dict(reset=False,lags={'1':pair(.1)}),
              dict(reset=False,lags={'1':pair(.1),'2':pair(.9)}),
              dict(reset=False,lags={'1':pair(.1,False)}),
              dict(reset=True,lags={}),dict(reset=False,lags={'1':pair(.1,age=.3)})]
        result=replay_gate(rows,.5)
        self.assertEqual(result['counts'],dict(initial=2,period=0,age=1,coverage=1,change=1,reuse=1))
        self.assertEqual(result['reuse_fraction'],1/6)
        rows=[dict(reset=(i==0),lags={str(lag):pair(0) for lag in [1,2]}) for i in range(4)]
        self.assertEqual(replay_gate(rows,.5)['counts']['period'],1)


if __name__=='__main__':unittest.main()
