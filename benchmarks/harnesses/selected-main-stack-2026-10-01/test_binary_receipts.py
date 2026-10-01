"""Tiny file-stat/digest receipt mocks; bytes are not executable models/code."""
import copy
import json
import unittest
from unittest.mock import patch
import test_templates as fixtures


class BinaryReceipts(unittest.TestCase):
    setUp=fixtures.Templates.setUp
    tearDown=fixtures.Templates.tearDown

    def pins(self):
        pins={}
        for arm in ['main','stack','seed']:
            path=self.work/'build'/(arm+'-frozen');path.write_bytes(b'model-free receipt fixture')
            pins[arm]={'path':str(path),'sha256':self.common.digest(path),'bytes':path.stat().st_size,
                       'revision':self.common.MANIFEST['revisions']['main' if arm=='seed' else arm]}
        return pins

    def test_exact_receipt_matches_real_stat_and_digest(self):
        pins=self.pins();self.assertEqual(self.common.validate_binaries(pins),pins)

    def test_size_schema_types_and_identity_corruption_fail_before_plan(self):
        pins=self.pins()
        cases=[]
        for value in [True,-1,3.5,'25',pins['main']['bytes']+1]:
            wrong=copy.deepcopy(pins);wrong['main']['bytes']=value;cases.append(wrong)
        for mutation in ['extra_arm','missing_arm','extra_field','missing_field','revision','digest','path']:
            wrong=copy.deepcopy(pins)
            if mutation=='extra_arm':wrong['other']=wrong['main']
            elif mutation=='missing_arm':wrong.pop('stack')
            elif mutation=='extra_field':wrong['main']['other']=1
            elif mutation=='missing_field':wrong['main'].pop('bytes')
            elif mutation=='revision':wrong['main']['revision']='wrong'
            elif mutation=='digest':wrong['main']['sha256']='0'*64
            else:wrong['main']['path']=True
            cases.append(wrong)
        for wrong in cases:
            with self.subTest(wrong=wrong):
                with patch.object(self.pg,'installation_identity',side_effect=AssertionError('plan must not reach PG')):
                    with self.assertRaises(ValueError):self.run.run_identity([],wrong,{},self.work/'seed')

    def test_read_json_rejects_duplicate_fields_and_symlink_binary_is_refused(self):
        pins=self.pins();receipt=self.work/'bad-binaries.json'
        receipt.write_text('{"main":{},"main":{},"stack":{}}')
        with self.assertRaises(ValueError):self.run.read_json(receipt)
        path=self.work/'build/main-frozen';target=self.work/'retained-bytes';path.rename(target);path.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'symlink'):self.common.validate_binaries(pins)


if __name__=='__main__':unittest.main()
