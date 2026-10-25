import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from opencood.tools.int_artifact_paths import artifact_path,relocate_tree,read_mapping


class TestArtifactPaths(unittest.TestCase):
    def test_longest_prefix_boundary_and_immutable_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mapping.json'
            path.write_text(json.dumps({'/old':'/new','/old/data':'/dataset'}))
            original={'pcd':'/old/data/frame.pcd','checkpoint':'/old/run/model.pth','other':['/older/x',5]}
            with patch.dict(os.environ,{'OPENCOOD_INT_PATH_MAP':str(path)}):
                result=relocate_tree(original)
                self.assertEqual(result,{'pcd':'/dataset/frame.pcd','checkpoint':'/new/run/model.pth','other':['/older/x',5]})
                self.assertEqual(artifact_path('/old/data'),'/dataset')
            self.assertEqual(original['pcd'],'/old/data/frame.pcd')
            with patch.dict(os.environ,{'OPENCOOD_INT_PATH_MAP':''}):self.assertEqual(relocate_tree(original),original)

    def test_reject_traversal_relative_and_root_maps(self):
        for mapping in ({'/old':'/new/../escape'},{'/':'/new'},{'relative':'/new'}):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'mapping.json';path.write_text(json.dumps(mapping));read_mapping.cache_clear()
                with self.assertRaises(ValueError):read_mapping(str(path))
