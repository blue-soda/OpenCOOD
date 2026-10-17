import unittest
from opencood.tools.launch_e3dsnn_seed_repeats import repeated_jobs


class SeedRepeatTest(unittest.TestCase):
    def test_seed_rejection(self):
        with self.assertRaises(ValueError):
            repeated_jobs({}, '/old', '/new', 123)

    def test_paths_isolated_and_seed_explicit_without_changing_source(self):
        plan = {'name': 'count4_aux1_joint30', 'jobs': [
            {'name': 'count4_aux1_joint30_train', 'command': ['python', '-m', 'opencood.tools.train_e3dsnn_single',
             '--config', '/repo/cfg.yaml', '--output', '/old/count4_aux1_joint30', '--epochs', '30']},
            {'name': 'count4_aux1_joint30_on_replay', 'command': ['python', '-m', 'replay', '--evaluation-dir', '/old/count4_aux1_joint30_on']}]}
        name, jobs = repeated_jobs(plan, '/old', '/new', 20260930)
        self.assertEqual(jobs[0]['command'][-2:], ['--seed', '20260930'])
        self.assertIn('/new/' + name, jobs[0]['command'])
        self.assertEqual(jobs[1]['command'][-1], '/new/' + name + '_on')
        self.assertIn('/repo/cfg.yaml', jobs[0]['command'])
        self.assertNotIn('--seed', plan['jobs'][0]['command'])
        self.assertNotIn('--seed', jobs[1]['command'])


if __name__ == '__main__':
    unittest.main()
