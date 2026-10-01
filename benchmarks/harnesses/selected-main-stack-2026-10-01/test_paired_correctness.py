"""Paired change-scope and actual denominator mocks; no native execution."""
import unittest
import test_templates as fixtures


class PairedCorrectness(unittest.TestCase):
    setUp=fixtures.Templates.setUp
    tearDown=fixtures.Templates.tearDown

    def classifications(self):
        items=[]
        for block in range(4):
            for source in ['main','stack']:
                items.append({'job':{'round':block,'source':source,'limit':5,'query_id':101,'sequence':'A,B'},
                    'changed_actual_reranker_inputs':block<2,'same_final_actual_inputs':True,
                    'same_final_raw_bits/order':source=='stack','all_changed_and_hot_exact':source=='stack',
                    'legacy_main_context_failure':source=='main','retrieval_or_input_context_mismatch':False})
        return items

    def test_mismatched_scope_excludes_entire_pair_and_reports_equal_denominators(self):
        items=self.classifications();items[0]['changed_actual_reranker_inputs']=False
        items[0]['retrieval_or_input_context_mismatch']=True
        correct,controls,excluded=self.analyzer.paired_correctness(items)
        self.assertEqual(len(excluded),1)
        self.assertEqual(excluded[0]['actual_input_change_by_arm'],{'main':False,'stack':True})
        self.assertTrue(all(r['samples_per_arm']==1 and r['excluded_unpaired_blocks']==1 for r in correct))
        self.assertTrue(all(r['samples_per_arm']==2 and r['excluded_unpaired_blocks']==1 for r in controls))
        exact=next(r for r in correct if r['metric']=='same_final_raw_bits/order')
        self.assertEqual((exact['before'],exact['after']),(0,1))
        self.assertIn('/1 paired blocks',exact['configuration'])

    def test_matching_scopes_retain_counts_and_incomplete_or_duplicate_pairs_fail(self):
        items=self.classifications();correct,controls,excluded=self.analyzer.paired_correctness(items)
        self.assertFalse(excluded)
        self.assertTrue(all(r['samples_per_arm']==2 for r in correct+controls))
        for invalid in [items[:-1],items+[items[0]],items[:1]]:
            with self.subTest(count=len(invalid)):
                with self.assertRaises(ValueError):self.analyzer.paired_correctness(invalid)
        items[0]['changed_actual_reranker_inputs']=1
        with self.assertRaisesRegex(ValueError,'invalid actual input-change'):self.analyzer.paired_correctness(items)


if __name__=='__main__':unittest.main()
