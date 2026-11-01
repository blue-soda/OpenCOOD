import copy
import math
import unittest
from opencood.models.sub_modules.int_event_refresh import TemporalRefreshController


class EventRefreshTests(unittest.TestCase):
    def test_exact_decay_and_subdivision_without_spikes(self):
        c=TemporalRefreshController(mode='continuous',threshold=100.)
        state=dict(membrane=.7,timestamp_us=0,scene='a')
        _,whole,_=c.step(.4,200000,'a',state)
        _,half,_=c.step(.4,100000,'a',state)
        _,split,_=c.step(.4,200000,'a',half)
        self.assertAlmostEqual(whole['membrane'],split['membrane'],places=14)
        _,decayed,_=c.step(0.,100000,'a',state)
        self.assertAlmostEqual(decayed['membrane'],.7*math.exp(-.5),places=14)

    def test_event_reset_is_distinct_from_continuous_control(self):
        event=TemporalRefreshController();continuous=TemporalRefreshController(mode='continuous')
        state=dict(membrane=.7,timestamp_us=0,scene='a');saved=copy.deepcopy(state)
        fired,e,ei=event.step(1.,100000,'a',state)
        cfired,c,ci=continuous.step(1.,100000,'a',state)
        self.assertTrue(fired and cfired);self.assertTrue(ei['spike']);self.assertFalse(ci['spike'])
        self.assertAlmostEqual(c['membrane']-e['membrane'],1.)
        self.assertEqual(ei['pre_reset_membrane'],ci['pre_reset_membrane'])
        self.assertEqual(saved,state)

    def test_threshold_has_no_memory_and_events_do(self):
        gate=TemporalRefreshController(mode='threshold');event=TemporalRefreshController()
        _,gs,_=gate.step(0.,0,'a');_,es,_=event.step(0.,0,'a')
        gate_events=[];events=[]
        for i in range(1,5):
            g,gs,_=gate.step(.8,i*100000,'a',gs);gate_events.append(g)
            e,es,_=event.step(.8,i*100000,'a',es);events.append(e)
        self.assertFalse(any(gate_events));self.assertTrue(any(events))

    def test_forced_scene_gap_and_initial_reset(self):
        for mode in ('threshold','continuous','event'):
            c=TemporalRefreshController(mode=mode)
            fired,state,info=c.step(1.,0,'a');self.assertTrue(fired);self.assertEqual(info['reason'],'initial')
            for stamp,scene,forced,reason in [(100000,'a',True,'forced'),(1,'b',False,'scene'),(300000,'a',False,'gap')]:
                fired,new,info=c.step(1.,stamp,scene,dict(state,membrane=3.),forced)
                self.assertTrue(fired);self.assertEqual(new['membrane'],0.);self.assertEqual(info['reason'],reason)
                self.assertFalse(info['spike'])

    def test_invalid_inputs_rejected(self):
        c=TemporalRefreshController();state=dict(membrane=0.,timestamp_us=10,scene='a')
        for evidence,stamp in [(-1.,11),(float('nan'),11),(float('inf'),11),(0.,10),(0.,9),(0.,11.)]:
            with self.assertRaises(ValueError):c.step(evidence,stamp,'a',state)
        with self.assertRaises(ValueError):TemporalRefreshController(tau=0.)


if __name__=='__main__':unittest.main()
