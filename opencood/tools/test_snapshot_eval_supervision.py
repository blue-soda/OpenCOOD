"""Reject adopting a trainer, another experiment, or changed evaluation flags."""
from pathlib import Path
import unittest
from opencood.tools.supervise_e3dsnn_snapshot_eval import validate_commands, DRIVER, TRAINER


class EvalSupervisionTests(unittest.TestCase):
    def commands(self):
        root=Path('/test/run');arm='fixed2k';folder=root/arm
        parent=['python','-m',DRIVER,'--root',str(root),'--arm',arm,'--mode','finish']
        child=['python','-m',TRAINER,'--mode','eval','--workers','2','--seed','20260929',
               '--config',str(folder/'b8192.yaml'),'--checkpoint',str(folder/'train/best_ap70.pth'),
               '--output',str(folder/'b8192'),'--packet-roundtrip','--eval-3d','--save-predictions','--profile']
        return root,arm,parent,child

    def test_registered_evaluator(self):
        r,a,p,c=self.commands();validate_commands(p,c,r,a)

    def test_reject_other_processes(self):
        r,a,p,c=self.commands()
        for key,value in [('--mode','train'),('--checkpoint','other.pth'),('--config',str(r/a/'b4096.yaml'))]:
            changed=list(c);changed[changed.index(key)+1]=value
            with self.assertRaises(ValueError):validate_commands(p,changed,r,a)
        p[-1]='coordinate'
        with self.assertRaises(ValueError):validate_commands(p,c,r,a)

    def test_reject_partial_or_overridden_evaluation(self):
        r,a,p,c=self.commands()
        for changed in [c+['--limit','5'],c+['--seed','20260929'],c[:-1],c+['--profile']]:
            with self.assertRaises(ValueError):validate_commands(p,changed,r,a)


if __name__=='__main__':unittest.main()
