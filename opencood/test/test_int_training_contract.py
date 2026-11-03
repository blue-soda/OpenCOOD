import unittest
from opencood.models.sub_modules.int_training_contract import require_matching_preflight


class TrainingContractTest(unittest.TestCase):
    def setUp(self):
        self.saved=dict(preflight_only=True,code_commit='fixed',prefixes=['a.','b.'],weight=.1)
        self.live=dict(preflight_only=False,code_commit='fixed',prefixes=('a.','b.'),weight=.1)

    def test_json_roundtrip_accepts_equivalent_sequence(self):
        require_matching_preflight(self.saved,self.live)
        self.assertIsInstance(self.live['prefixes'],tuple)

    def test_changed_code_or_scientific_setting_rejected(self):
        for key,value in [('code_commit','other'),('weight',.2),('prefixes',('a.',))]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,key):
                require_matching_preflight(self.saved,dict(self.live,**{key:value}))

    def test_missing_extra_and_wrong_mode_rejected(self):
        missing=dict(self.live);del missing['weight']
        for current in (missing,dict(self.live,extra=1),dict(self.live,preflight_only=True)):
            with self.assertRaises(ValueError):require_matching_preflight(self.saved,current)


if __name__=='__main__':unittest.main()
