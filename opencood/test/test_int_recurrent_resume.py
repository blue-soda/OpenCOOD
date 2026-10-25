"""A validation failure must not require a second optimizer training pass."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from opencood.tools.train_int_recurrent_deep import train_recurrent


class TestRecurrentResume(unittest.TestCase):
    def test_pending_validation_restores_model_and_skips_training(self):
        module='opencood.tools.train_int_recurrent_deep.'
        for epoch in (0,1):
            with self.subTest(epoch=epoch),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);first=root/'first';second=root/'second';first.mkdir();second.mkdir()
                provenance={k:{} for k in ('source','memory','readout','repair','basis','data')}
                for out in (first,second):(out/'contract.json').write_text(json.dumps(provenance))
                baseline=SimpleNamespace(repair=torch.nn.Linear(2,1))
                args=SimpleNamespace(recurrent_deep_period=3,recurrent_deep_epochs=0,recurrent_deep_resume=None)
                cfg={'loss':{'args':{}}}
                def mode(base,repair,period):return SimpleNamespace(deep=SimpleNamespace(max_age_us=250000),repair=repair)
                with patch(module+'RecurrentDeepDetector',side_effect=mode),patch(module+'engineering',return_value={}),patch(module+'PointPillarDirLoss'),patch(module+'evaluate',side_effect=RuntimeError('validation interrupted')):
                    with self.assertRaisesRegex(RuntimeError,'validation interrupted'):
                        train_recurrent(args,cfg,[],[],baseline,first,{},0)
                checkpoint=first/'pending_validation.pth';saved=torch.load(str(checkpoint),map_location='cpu')
                self.assertEqual(saved['next_action'],'evaluate')
                self.assertIsNone(saved['best'])  # Epoch0 is a valid resumable boundary too.
                if epoch:
                    trained=torch.nn.Linear(2,1);trained.load_state_dict(saved['repair'])
                    optimizer=torch.optim.AdamW(trained.parameters(),lr=1e-4,weight_decay=1e-4)
                    trained(torch.ones(1,2)).sum().backward();optimizer.step()
                    saved.update(repair=trained.state_dict(),optimizer=optimizer.state_dict(),epoch=1,
                        training_summary=dict(losses=[.25],depths={1:1},updates=1))
                    saved['contract']['epochs_max']=1
                    torch.save(saved,str(checkpoint))
                args.recurrent_deep_resume=str(checkpoint);args.recurrent_deep_epochs=epoch
                def evaluate(a,base,restored,*unused):
                    for name,value in saved['repair'].items():self.assertTrue(torch.equal(restored.repair.state_dict()[name],value))
                    return {'results':{'recurrent':{'ap70':.75}}}
                with patch(module+'RecurrentDeepDetector',side_effect=mode),patch(module+'engineering',return_value={}),patch(module+'PointPillarDirLoss'),patch(module+'evaluate',side_effect=evaluate) as evaluation,patch(module+'loader',side_effect=AssertionError('must not repeat training')):
                    train_recurrent(args,cfg,[],[],baseline,second,{},0)
                evaluation.assert_called_once()
                resumed=torch.load(str(second/'latest.pth'),map_location='cpu')
                self.assertEqual(resumed['next_action'],'train')
                self.assertEqual(resumed['history'][-1]['epoch'],epoch)
                self.assertEqual(resumed['history'][-1]['updates'],epoch)
                self.assertEqual(set(resumed['optimizer']['state']),set(saved['optimizer']['state']))
                for key in saved['optimizer']['state']:
                    for field,value in saved['optimizer']['state'][key].items():
                        actual=resumed['optimizer']['state'][key][field]
                        self.assertTrue(torch.equal(actual,value) if torch.is_tensor(value) else actual==value)
