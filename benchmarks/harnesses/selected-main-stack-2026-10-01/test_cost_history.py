"""Model-free new-query cost history eligibility counterexamples."""
import copy
import unittest
import test_metric_host_conditions as hosts


class CostHistory(unittest.TestCase):
    setUp = hosts.HostTimingEligibility.setUp

    def new_query(self):
        observations=copy.deepcopy(self.observations)
        for item in observations:
            item['phase']='new-query'; item['row']['arm']='N'
            item['row']['before']={key:30 for key in ['offered','scored','characters','tokens',
                'padded_tokens','batches','maximum_tokens','remembered','longest']}
        return observations

    def rows(self, observations):
        return self.metrics.metric_tables(observations,{},self.conditions,self.provider)

    def test_equal_new_delta_with_different_prior_work_withholds_cost_only(self):
        observations=self.new_query()
        for item in observations:
            if item['job']['source']=='main':item['row']['before']['scored']=0
        result=self.rows(observations)
        cost=[r for r in result if r['metric'] in self.metrics.COST_METRICS]
        self.assertTrue(all(not r['same_successful_work'] and not r['stable_claim_eligible'] for r in cost))
        self.assertTrue(all(r['accuracy_eligible'] for r in result))
        self.assertTrue(all(r['same_successful_work'] for r in result if r['metric'].startswith('quality_')))

    def test_equal_prior_counters_preserve_matching_positive_independent_of_item_order(self):
        observations=self.new_query();observations.reverse()
        result=self.rows(observations)
        self.assertTrue(all(r['same_successful_work'] and r['stable_claim_eligible']
                            for r in result if r['metric'] in self.metrics.COST_METRICS))

    def test_missing_invalid_or_different_allocation_shape_cannot_qualify_new_query_cost(self):
        for mutation in ['missing', 'bool', 'shape']:
            observations=self.new_query()
            if mutation=='missing':observations[0]['row'].pop('before')
            elif mutation=='bool':observations[0]['row']['before']['scored']=True
            else:observations[0]['row']['before']['maximum_tokens']+=1
            with self.subTest(mutation=mutation):
                cost=[r for r in self.rows(observations) if r['metric'] in self.metrics.COST_METRICS]
                self.assertTrue(all(not r['stable_claim_eligible'] for r in cost))


if __name__=='__main__':unittest.main()
