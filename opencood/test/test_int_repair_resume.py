import copy
import unittest
import torch
from opencood.tools.int_repair_resume import restore_training


class RepairResumeTests(unittest.TestCase):
    def test_next_update_and_plateau_schedule_match_uninterrupted(self):
        torch.manual_seed(4)
        model=torch.nn.Linear(3,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1)
        x=torch.randn(4,3);target=torch.randn(4,2)
        def update(m,o):
            o.zero_grad();(m(x)-target).square().mean().backward();o.step()
        history=[]
        for epoch in range(1,4):
            update(model,optimizer);scheduler.step(.5)
            history.append(dict(epoch=epoch,ap70=.5,stale_epochs=epoch-1,lr=optimizer.param_groups[0]['lr']))
        contract=dict(epochs_max=3,repair_loss='foreground')
        saved=copy.deepcopy(dict(repairs=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            contract=contract,epoch=3,best=.5,significant=.5,stale=2,history=history))
        resumed=torch.nn.Linear(3,2);opt=torch.optim.AdamW(resumed.parameters(),lr=.001)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=1)
        result=restore_training(saved,resumed,opt,sched,dict(contract,epochs_max=8))
        self.assertEqual(result[:4],(4,.5,.5,2))
        for _ in range(3):
            update(model,optimizer);update(resumed,opt);scheduler.step(.5);sched.step(.5)
            for a,b in zip(model.parameters(),resumed.parameters()):self.assertTrue(torch.equal(a,b))
            self.assertEqual(opt.param_groups[0]['lr'],optimizer.param_groups[0]['lr'])
        with self.assertRaises(AssertionError):
            restore_training(saved,resumed,opt,sched,dict(contract,epochs_max=8,repair_loss='legacy'))


if __name__=='__main__':unittest.main()
