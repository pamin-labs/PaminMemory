"""Tiny shared timing and process HWM guards, before summary arithmetic."""
import copy,importlib.util,math,sys,unittest
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('review',Path(__file__).with_name('evidence_review.py'))
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
def snapshot(hwm):return {'rss_status':['VmRSS: 1 kB',f'VmHWM: {hwm} kB'],'user_ticks':1,'system_ticks':1}
def phase(before=2,after=3):return {'phase':'open','wall_ms':1.0,'cpu_user_seconds':0.0,'cpu_system_seconds':0.01,'process_before':snapshot(before),'process_after':snapshot(after)}
class Observations(unittest.TestCase):
 def test_real_observations(self):
  for value in [0,0.0,1,1.5]:review.real_measurement(value)
  for value in [True,False,'1',None,float('nan'),float('inf'),-1]:
   with self.assertRaisesRegex(AssertionError,'real timing'):review.real_measurement(value)
  for key in ['wall_ms','cpu_user_seconds','cpu_system_seconds']:
   row=phase();row[key]=True
   with self.assertRaisesRegex(AssertionError,'real timing'):review.process_observations([row])
  with self.assertRaisesRegex(AssertionError,'real timing'):review.process_observations([{'phase':'process_total','wall_seconds':True}])
 def test_hwm_within_and_between_phases(self):
  review.process_observations([phase(),phase(3,4),{'phase':'process_total','wall_seconds':1.0}])
  with self.assertRaisesRegex(AssertionError,'declared historical anomaly changed'):review.process_observations([phase()], [{'arm':'fixture','repetition':0}])
  for rows in [[phase(4,3)],[phase(2,4),phase(3,5)]]:
   with self.assertRaisesRegex(AssertionError,'high-water mark decreased'):review.process_observations(rows)
  missing=phase();missing['process_before']=missing['process_after']=None
  review.process_observations([phase(2,4),missing,phase(4,5)])
  with self.assertRaisesRegex(AssertionError,'high-water mark decreased'):review.process_observations([phase(2,4),missing,phase(3,5)])
 def test_cpu_ticks_within_between_and_missing_snapshots(self):
  for key in ['user_ticks','system_ticks']:
   first=phase();second=phase(3,4)
   for field in ['process_before','process_after']:
    first[field][key]=10
    second[field][key]=9
   # Each phase's delta is zero/nonnegative; the inter-phase history is impossible.
   with self.assertRaisesRegex(AssertionError,'cumulative CPU counters decreased'):review.process_observations([first,second])
   missing=phase();missing['process_before']=missing['process_after']=None
   with self.assertRaisesRegex(AssertionError,'cumulative CPU counters decreased'):review.process_observations([first,missing,second])
   within=phase();within['process_before'][key]=2;within['process_after'][key]=1
   with self.assertRaisesRegex(AssertionError,'cumulative CPU counters decreased'):review.process_observations([within])
   for value in [True,1.0,-1,'1']:
    bad=phase();bad['process_before'][key]=value
    with self.assertRaisesRegex(AssertionError,'invalid process CPU tick'):review.process_observations([bad])
  healthy=[phase(),phase(3,4)]
  for row in healthy:
   for field in ['process_before','process_after']:
    row[field].update(user_ticks=10,system_ticks=20)
  review.process_observations(healthy)
if __name__=='__main__':unittest.main()
