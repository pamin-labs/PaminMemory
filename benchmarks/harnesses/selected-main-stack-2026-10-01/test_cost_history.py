"""Model-free new-query cost history eligibility counterexamples."""
import copy
import unittest
import test_metric_host_conditions as hosts


class CostHistory(unittest.TestCase):
    setUp = hosts.HostTimingEligibility.setUp

    def new_query(self, phase='new-query', arm='N'):
        observations=copy.deepcopy(self.observations)
        for item in observations:
            item['phase']=phase; item['row']['arm']=arm
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

    def test_changed_hot_cost_requires_equal_prior_work(self):
        # Identical zero-work replays still enter with different histories.
        observations=self.new_query('changed-hot','A')
        for item in observations:
            if item['job']['source']=='main':item['row']['before']['scored']=0
        cost=[r for r in self.rows(observations) if r['metric'] in self.metrics.COST_METRICS]
        self.assertTrue(cost and all(not r['same_successful_work'] and not r['stable_claim_eligible'] for r in cost))
        self.assertTrue(all('changed-hot' in r['preceding_work_scope'] for r in cost))
        matching=[r for r in self.rows(self.new_query('changed-hot','A')) if r['metric'] in self.metrics.COST_METRICS]
        self.assertTrue(all(r['same_successful_work'] for r in matching))

    def test_changed_hot_gate_accepts_several_calls_per_block(self):
        # Five changed-hot calls per block and arm, matching histories.
        observations=[]
        for item in self.new_query('changed-hot','A'):
            for step in range(5):
                copied=copy.deepcopy(item);copied['row']['before']['tokens']=30+step;observations.append(copied)
        cost=[r for r in self.rows(observations) if r['metric'] in self.metrics.COST_METRICS]
        self.assertTrue(cost and all(r['same_successful_work'] for r in cost))
        self.assertEqual({r['samples_per_arm'] for r in cost},{20})
        observations[0]['row']['before']['tokens']+=1
        cost=[r for r in self.rows(observations) if r['metric'] in self.metrics.COST_METRICS]
        self.assertTrue(all(not r['same_successful_work'] for r in cost))


if __name__=='__main__':unittest.main()
