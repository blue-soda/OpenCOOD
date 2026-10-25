"""CPU-only guards and Linux parent-only termination regression."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from opencood.tools.supervise_e3dsnn_snapshot_joint import (
    DRIVER, TRAINER, validate_commands, process, same)


class SupervisionTests(unittest.TestCase):
    def commands(self):
        root=Path('/test/run');arm='multi248k'
        parent=['python','-m',DRIVER,'--root',str(root),'--arm',arm,'--mode','coordinate']
        child=['python','-m',TRAINER,'--mode','train','--config',str(root/arm/'train.yaml'),
            '--output',str(root/arm/'train'),'--checkpoint',str(root/'source.pth'),
            '--epochs','10','--selection-iou','0.7']
        return root,arm,parent,child

    def test_expected_commands(self):
        r,a,p,c=self.commands();validate_commands(p,c,r,a)

    def test_reject_wrong_experiment_and_resume(self):
        r,a,p,c=self.commands()
        with self.assertRaises(ValueError):validate_commands(p,c,r,'fixed2k')
        with self.assertRaises(ValueError):validate_commands(p,c+['--resume'],r,a)
        c[c.index('--epochs')+1]='20'
        with self.assertRaises(ValueError):validate_commands(p,c,r,a)

    def test_reject_duplicate_flags(self):
        r,a,p,c=self.commands()
        with self.assertRaises(ValueError):validate_commands(p+['--root',str(r)],c,r,a)

    @unittest.skipUnless(sys.platform=='linux','Linux /proc ownership and orphan behavior')
    def test_exact_parent_termination_preserves_child(self):
        code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(p.pid,flush=True); time.sleep(20)"
        parent=subprocess.Popen([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,
            text=True,start_new_session=True)
        child=None
        try:
            child=int(parent.stdout.readline().strip());identity=process(child)
            self.assertEqual(identity['parent'],parent.pid)
            self.assertFalse(same(dict(identity,start_ticks=identity['start_ticks']+1)))
            parent.terminate();parent.wait(timeout=5)
            self.assertTrue(same(identity))
        finally:
            if parent.poll() is None:parent.kill();parent.wait(timeout=5)
            if child is not None:
                try:os.kill(child,signal.SIGTERM)
                except ProcessLookupError:pass
            parent.stdout.close()


if __name__=='__main__':unittest.main()
