"""Bidirectional assembly must stay inside inference and retain exact bytes."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/strong_ann_baselines'))
from cosdh_wire import CoSDHWire, envelope
from compute_cosdh_adapter import CoSDHAdapter


class CoSDHGateTests(unittest.TestCase):
    def test_memory_assembly_matches_export_both_directions(self):
        w=object.__new__(CoSDHWire);w.parts={i:bytes([i])*13 for i in range(4)}
        w.request=b'request';w.frame_id='000211';w.pose=np.eye(4,dtype=np.float32)
        w.late_count=0;w.details=[]
        expected=[('demand',envelope(w.frame_id,0,w.pose)+w.request),
                  ('response',envelope(w.frame_id,1,w.pose)+struct.pack('<4sI',b'CFW1',4)+b''.join(w.parts[i] for i in range(4)))]
        self.assertEqual(w.packets(),expected)
        with tempfile.TemporaryDirectory() as td:
            w.output=Path(td);row=w.finish(0)
            self.assertEqual(row['total_bytes'],sum(len(p) for _,p in expected))
            for direction,blob in expected:self.assertEqual((w.output/('000000_'+direction+'.bin')).read_bytes(),blob)

    def test_stale_message_is_rejected(self):
        a=CoSDHAdapter(None,None,None,['a'],{});a.messages=[('demand',b'old')]
        a.reset('a')
        with self.assertRaises(ValueError):a.packets()

    def test_assembly_occurs_before_infer_returns(self):
        import types
        wire=types.SimpleNamespace(begin=lambda batch,fid:None,packets=lambda:[('demand',b'fresh')])
        a=CoSDHAdapter(None,None,wire,['a'],{});a.reset('a')
        module=types.ModuleType('opencood.tools.inference_utils')
        module.inference_late_fusion=lambda batch,model,dataset:'result'
        with patch.dict(sys.modules,{'opencood.tools.inference_utils':module}):
            self.assertEqual(a.infer({}),'result')
        wire.packets=lambda:(_ for _ in ()).throw(AssertionError('outside time'))
        self.assertEqual(a.packets(),[('demand',b'fresh')])


if __name__=='__main__':unittest.main()
