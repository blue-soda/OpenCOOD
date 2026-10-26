import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/strong_ann_baselines'))
from compute_harness import measurement_indices, selected_indices


class SelectionTests(unittest.TestCase):
    def test_full_measurement_preserves_registered_selection_and_gate(self):
        full=measurement_indices(64)
        self.assertEqual(full,selected_indices())
        self.assertEqual(measurement_indices(4),[full[i] for i in (0,21,42,63)])
        self.assertEqual(len(set(full)),64)
        for count in [0,5,63,65]:
            with self.assertRaises(ValueError):measurement_indices(count)


if __name__=='__main__':unittest.main()
