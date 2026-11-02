import unittest
from opencood.tools.pilot_int_protected_state import schedule


class ProtectedBudgetTests(unittest.TestCase):
    def test_refresh_only_window_is_not_a_fake_optimizer_update(self):
        rows=[dict(frame=str(i),timestamp_us=i*100000,segment='a' if i<4 else 'b',supervised=True) for i in range(5)]
        reasons,updates=schedule(rows)
        self.assertEqual(reasons,['initial',None,None,None,'scene']);self.assertEqual(updates,['3'])

    def test_real_age_gap_scene_and_unlabelled_windows(self):
        times=[0,200000,400000,600000,900001,1000001,1100001,1200001]
        rows=[dict(frame=str(i),timestamp_us=t,segment='a',supervised=i>=4) for i,t in enumerate(times)]
        reasons,updates=schedule(rows)
        self.assertEqual(reasons,['initial',None,None,'age','gap',None,None,None]);self.assertEqual(updates,['7'])


if __name__=='__main__':unittest.main()
