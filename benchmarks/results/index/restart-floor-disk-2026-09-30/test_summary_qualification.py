"""Tiny synthetic summary checks; raw/calculator must remain unchanged."""
import copy,importlib.util,sys,unittest
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('review',Path(__file__).with_name('evidence_review.py'));review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
def fixture():
 return {'arms':{arm:{'processes':[{'peak_process_rss_bytes':123,'maintenance_rss_bytes':100} for _ in range(3)],'median_of_process_metrics':{'peak_process_rss_bytes':123,'maintenance_rss_bytes':100}} for arm in ['main','predecessor','candidate']},'comparisons':{arm:{'metric_differences':{'peak_process_rss_bytes':{'before':123,'after':124,'absolute_delta':1,'percent_delta':100/123},'maintenance_rss_bytes':{'before':100,'after':100,'absolute_delta':0,'percent_delta':0}}} for arm in ['main','predecessor']}}
def receipt():return {'index':'disk','raw_sha256':'0'*64,'hwm_peak_certified':False,'published_hwm_peak':'N/A','exact_historical_anomalies':copy.deepcopy(review.HWM_ANOMALIES['disk'])}
class Qualification(unittest.TestCase):
 def test_nulls_all_derived_peaks_without_mutating_other_evidence(self):
  original=fixture();before=copy.deepcopy(original);qualified=review.qualified_summary(original,receipt(),'1'*64);self.assertEqual(original,before)
  for arm in qualified['arms'].values():
   self.assertIsNone(arm['median_of_process_metrics']['peak_process_rss_bytes'])
   for process in arm['processes']:self.assertIsNone(process['peak_process_rss_bytes']);self.assertEqual(process['maintenance_rss_bytes'],100)
  for arm,comparison in qualified['comparisons'].items():
   self.assertTrue(all(value is None for value in comparison['metric_differences']['peak_process_rss_bytes'].values()))
   self.assertEqual(comparison['metric_differences']['maintenance_rss_bytes'],before['comparisons'][arm]['metric_differences']['maintenance_rss_bytes'])
  self.assertIs(qualified['metric_certification']['peak_process_rss_bytes']['certified'],False)
  for wrong in [True,0,None]:
   bad=receipt();bad['hwm_peak_certified']=wrong
   with self.assertRaises(AssertionError):review.qualified_summary(original,bad,'1'*64)
  bad=receipt();bad['exact_historical_anomalies'][0]['current'][-1]+=1
  with self.assertRaises(AssertionError):review.qualified_summary(original,bad,'1'*64)
if __name__=='__main__':unittest.main()
