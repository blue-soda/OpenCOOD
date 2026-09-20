"""Validate loader CLI controls without importing CUDA or dataset dependencies."""

import argparse
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import contextlib
import io


class TrainLoaderOptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'tools' / 'train.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'train_parser']
        namespace = {'argparse': argparse}
        exec(compile(tree, str(source), 'exec'), namespace)
        cls.parse = staticmethod(namespace['train_parser'])

    def parse_args(self, *args):
        with patch.object(sys, 'argv', ['train', '-y', 'unused.yaml'] + list(args)):
            return self.parse()

    def test_defaults_remain_disabled(self):
        args = self.parse_args()
        self.assertIsNone(args.multiprocessing_context)
        self.assertEqual(args.loader_stack_interval, 0)

    def test_spawn_and_worker_diagnostics(self):
        args = self.parse_args('--num_workers', '2', '--multiprocessing_context', 'spawn',
                               '--loader_stack_interval', '30', '--loader_stack_dir', '/tmp/stacks')
        self.assertEqual(args.num_workers, 2)
        self.assertEqual(args.multiprocessing_context, 'spawn')
        self.assertEqual(args.loader_stack_interval, 30)

    def test_invalid_combinations(self):
        for args in [('--num_workers', '-1'),
                     ('--num_workers', '0', '--multiprocessing_context', 'spawn'),
                     ('--loader_stack_interval', '-1'),
                     ('--loader_stack_interval', '30'),
                     ('--loader_stack_interval', '30', '--loader_stack_dir', '/tmp/stacks',
                      '--num_workers', '0')]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse_args(*args)


if __name__ == '__main__':
    unittest.main()
