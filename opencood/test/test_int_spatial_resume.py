import json
import tempfile
import unittest
from pathlib import Path
import torch
from opencood.tools.int_spatial_resume import load_completed_spatial_resume,sha256


class SpatialResumeTests(unittest.TestCase):
    def fixture(self,root):
        torch.manual_seed(303)
        model=torch.nn.Linear(3,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=1,threshold=.001,threshold_mode='abs')
        contract=dict(data='fixed-manifest',architecture='test',phase='dense')
        history=[]
        for epoch,ap in ((1,.75),(2,.7504)):
            optimizer.zero_grad();model(torch.ones(2,3)).square().sum().backward();optimizer.step();scheduler.step(ap)
            history.append(dict(epoch=epoch,lr=optimizer.param_groups[0]['lr'],stale=epoch-1,
                validation=dict(results=dict(full=dict(ap70=ap)))))
        folder=root/'epoch_002';folder.mkdir()
        path=folder/'stage.pth'
        torch.save(dict(stage=model.state_dict(),contract=contract,epoch=2),str(path))
        best=dict(epoch=2,ap70=.7504,checkpoint=str(path),sha256=sha256(path))
        for name,value in [('status.json',dict(status='COMPLETED',epochs=2)),
            ('report.json',dict(status='COMPLETED',epochs=2,stop_reason='budget_not_convergence',best=best)),
            ('history.json',history),('best.json',best)]:
            (root/name).write_text(json.dumps(value))
        torch.save(dict(stage=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,contract=contract),str(root/'latest.pth'))
        return model,optimizer,scheduler,contract

    def test_restores_next_adam_step_and_plateau_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,optimizer,scheduler,contract=self.fixture(root)
            resume=load_completed_spatial_resume(root,contract)
            self.assertEqual(resume['epoch'],2);self.assertEqual(resume['stale'],1)
            self.assertEqual(resume['significant'],.75)
            restored=torch.nn.Linear(3,2);restored.load_state_dict(resume['saved']['stage'])
            restored_opt=torch.optim.AdamW(restored.parameters(),lr=.1)
            restored_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(restored_opt,mode='max',patience=1,threshold=.001,threshold_mode='abs')
            restored_opt.load_state_dict(resume['saved']['optimizer'])
            restored_scheduler.load_state_dict(resume['saved']['scheduler'])
            for m,o,s in ((model,optimizer,scheduler),(restored,restored_opt,restored_scheduler)):
                o.zero_grad();m(torch.full((2,3),.3)).square().sum().backward();o.step();s.step(.7505)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters())))
            self.assertEqual(scheduler.state_dict(),restored_scheduler.state_dict())
            self.assertEqual(optimizer.param_groups[0]['lr'],.0001)

    def test_live_source_is_rejected_before_reading_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'status.json').write_text(json.dumps(dict(status='RUNNING')))
            with self.assertRaisesRegex(ValueError,'completed'):
                load_completed_spatial_resume(root,{})

    def test_changed_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,_,_,contract=self.fixture(root);contract['data']='different'
            with self.assertRaisesRegex(ValueError,'contract'):
                load_completed_spatial_resume(root,contract)

    def test_latest_must_equal_completed_epoch_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,_,_,contract=self.fixture(root)
            final=root/'epoch_002'/'stage.pth';saved=torch.load(str(final))
            saved['stage']['weight']+=1;torch.save(saved,str(final))
            with self.assertRaisesRegex(ValueError,'Latest weights'):
                load_completed_spatial_resume(root,contract)

    def test_six_epoch_continuation_does_not_accept_plateau_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,_,_,contract=self.fixture(root)
            resume=load_completed_spatial_resume(root,contract,additional_epochs_max=6)
            self.assertEqual(resume['provenance']['additional_epochs_max'],6)
            for invalid in (0,-1,1.5,True):
                with self.assertRaisesRegex(ValueError,'positive integer'):
                    load_completed_spatial_resume(root,contract,additional_epochs_max=invalid)
            report=json.loads((root/'report.json').read_text());report['stop_reason']='plateau'
            (root/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'budget-limited'):
                load_completed_spatial_resume(root,contract,additional_epochs_max=6)


if __name__=='__main__':unittest.main()
