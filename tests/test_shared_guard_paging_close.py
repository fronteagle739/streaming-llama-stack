"""Actual close observations, error retention and pinned-provider receipt behavior."""
import json
import threading
import types
import unittest
from slstack import shared_guard as adapter

class Watcher:
 def __init__(self,errors=()):
  self.errors=list(errors);self.calls=0;self.max_global_growth=0;self.job_paged=0
  self.observer=object();self.owned_paging_limit=17
 def close(self):
  self.calls+=1
  if self.errors:
   err=self.errors.pop(0)
   if err is not None:raise err
  return 'original-return'
 def details(self):return {'existing_observation':'retained','owned_paging_limit_bytes':self.owned_paging_limit}
class Receipt:
 def __init__(self):self.body=None
 def write_text(self,text):self.body=json.loads(text)
def provider_job(watcher):
 receipt=Receipt();record={'command_exit_code':0}
 tracked=adapter._PagingCloseTracker(watcher,record)
 child=types.SimpleNamespace(stdin=None,poll=lambda:0,wait=lambda timeout:0)
 job=types.SimpleNamespace(finished=threading.Event(),monitor=None,owned=object(),child=child,
  kernel_job=None,failure=None,record=record,watcher=tracked,receipt=receipt,
  _stop=lambda:{'confirmed':True,'identities':[]})
 return job,receipt

class CloseContract(unittest.TestCase):
 def test_never_called_is_false_not_inferred_from_cleanup(self):
  record={'release_allowed':True,'cleanup':{'confirmed':True}};w=Watcher();t=adapter._PagingCloseTracker(w,record)
  self.assertFalse(record['paging_telemetry_closed']);self.assertEqual(w.calls,0)
  self.assertTrue(record['release_allowed']);self.assertFalse(adapter._paging_telemetry_closed(types.SimpleNamespace(record=record)))
 def test_true_only_after_actual_normal_close_and_return_preserved(self):
  record={'release_allowed':False};w=Watcher();t=adapter._PagingCloseTracker(w,record)
  self.assertEqual(t.close(),'original-return');self.assertEqual(w.calls,1)
  self.assertIs(record['paging_telemetry_closed'],True);self.assertEqual(record['paging_telemetry_close_attempts'],1)
  self.assertFalse(record['release_allowed']) # wrapper never grants resource release
 def test_exception_retained_and_reraised(self):
  record={'release_allowed':True};t=adapter._PagingCloseTracker(Watcher([RuntimeError('close failed')]),record)
  with self.assertRaisesRegex(RuntimeError,'close failed'):t.close()
  self.assertIs(record['paging_telemetry_closed'],False);self.assertFalse(record['release_allowed'])
  self.assertIn('close failed',record['paging_telemetry_close_error'])
 def test_baseexception_does_not_waive_lease(self):
  record={'release_allowed':True};t=adapter._PagingCloseTracker(Watcher([KeyboardInterrupt('interrupted close')]),record)
  with self.assertRaises(KeyboardInterrupt):t.close()
  self.assertFalse(record['paging_telemetry_closed']);self.assertFalse(record['release_allowed'])
 def test_retry_cannot_erase_failed_attempt(self):
  record={'release_allowed':True};w=Watcher([RuntimeError('first failure'),None]);t=adapter._PagingCloseTracker(w,record)
  with self.assertRaises(RuntimeError):t.close()
  record['release_allowed']=True # provider may recompute owned cleanup before retry
  t.close();self.assertEqual(w.calls,2);self.assertEqual(record['paging_telemetry_close_attempts'],2)
  self.assertFalse(record['paging_telemetry_closed']);self.assertFalse(record['release_allowed']);self.assertIn('first failure',record['paging_telemetry_close_error'])
 def test_paging_fields_checks_details_not_hidden(self):
  record={};w=Watcher();t=adapter._PagingCloseTracker(w,record)
  self.assertIs(t.observer,w.observer);t.owned_paging_limit=29
  self.assertEqual(w.owned_paging_limit,29);self.assertEqual(t.details(),w.details())
  t.transient='value';self.assertEqual(w.transient,'value');del t.transient;self.assertFalse(hasattr(w,'transient'))
 def test_canonical_bool_refuses_missing_none_strings_and_integers(self):
  for value in (None,False,1,'true'):
   self.assertIs(adapter._paging_telemetry_closed(types.SimpleNamespace(record={'paging_telemetry_closed':value})),False)
  self.assertIs(adapter._paging_telemetry_closed(types.SimpleNamespace(record={})),False)
  self.assertIs(adapter._paging_telemetry_closed(types.SimpleNamespace(record={'paging_telemetry_closed':True})),True)
 def test_real_pinned_provider_close_persists_success_and_keeps_cleanup(self):
  job,sink=provider_job(Watcher());adapter.GuardedJob.close(job)
  self.assertEqual(job.watcher.calls,1);self.assertIs(sink.body['paging_telemetry_closed'],True)
  self.assertEqual(sink.body['existing_observation'],'retained');self.assertTrue(sink.body['release_allowed'])
  self.assertEqual(sink.body['status'],'PASS');self.assertIsNone(job.failure)
 def test_real_pinned_provider_close_failure_holds_lease(self):
  job,sink=provider_job(Watcher([RuntimeError('observer close failed')]));adapter.GuardedJob.close(job)
  self.assertIs(sink.body['paging_telemetry_closed'],False);self.assertFalse(sink.body['release_allowed'])
  self.assertIn('observer close failed',sink.body['failure']);self.assertEqual(sink.body['status'],'FAILED_OR_CANCELLED')
  self.assertFalse(adapter._paging_telemetry_closed(job))

if __name__=='__main__':unittest.main(verbosity=2)
