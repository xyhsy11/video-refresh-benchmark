"""Explicitly SYNTHETIC post-exposure seal-amendment safety fixtures only."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import ucf101_proxy_common as common


class AmendmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC_amendment_')
        self.root = Path(self.temp.name)
        self.out = self.root/'results'
        self.patches = [patch.object(common, 'ROOT', self.root), patch.object(common, 'OUT', self.out)]
        for p in self.patches:
            p.start()
        names = ['ucf101_proxy_common.py', 'ucf101_proxy_verify.py']
        originals = {}
        for name in [*names, 'inference_fixture.py']:
            (self.root/name).write_bytes(b'SYNTHETIC original '+name.encode())
            originals[name] = common.sha(self.root/name)
        common.save(self.out/'evaluation_seal.json', {'sha256': originals})
        common.save(self.out/'evaluation_exposure.json', {'seal_sha256': common.sha(self.out/'evaluation_seal.json')})
        changed = {}
        for name in names:
            archived = self.root/'originals'/name
            archived.parent.mkdir(exist_ok=True)
            archived.write_bytes((self.root/name).read_bytes())
            (self.root/name).write_bytes(b'SYNTHETIC amended '+name.encode())
            changed[name] = dict(original_sha256=originals[name], archived_original='originals/'+name,
                amended_sha256=common.sha(self.root/name))
        (self.root/'scientific_fixture.csv').write_bytes(b'SYNTHETIC fixture only\n')
        common.save(self.out/'post_exposure_amendment.json', dict(
            original_seal_sha256=common.sha(self.out/'evaluation_seal.json'),
            original_exposure_sha256=common.sha(self.out/'evaluation_exposure.json'),
            changed_frozen_files=changed,
            unchanged_scientific_outputs={'scientific_fixture.csv': common.sha(self.root/'scientific_fixture.csv')}))

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_valid_explicit_amendment(self):
        common.seal_check(True)

    def test_unamended_inference_mutation_rejected(self):
        (self.root/'inference_fixture.py').write_bytes(b'SYNTHETIC changed')
        with self.assertRaises(common.Blocker):
            common.seal_check(True)

    def test_original_archive_mutation_rejected(self):
        (self.root/'originals/ucf101_proxy_verify.py').write_bytes(b'SYNTHETIC changed')
        with self.assertRaises(common.Blocker):
            common.seal_check(True)

    def test_scientific_output_mutation_rejected(self):
        (self.root/'scientific_fixture.csv').write_bytes(b'SYNTHETIC changed')
        with self.assertRaises(common.Blocker):
            common.seal_check(True)

    def test_original_exposure_mutation_rejected(self):
        common.save(self.out/'evaluation_exposure.json', {'seal_sha256': 'SYNTHETIC changed'})
        with self.assertRaises(common.Blocker):
            common.seal_check(True)


if __name__ == '__main__':
    unittest.main()
