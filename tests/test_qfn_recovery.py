import unittest
from slstack.qfn_recovery import restart_allowed
from slstack.paging_limits import shared_owned_paging_limit_bytes


class PeerRecovery(unittest.TestCase):
    def record(self,**changes):
        return dict(dict(exit_code=1,reason=None,release_allowed=True,
            cleanup_confirmed=True,paging_telemetry_closed=True,owned_paged_bytes=0,child_pid=123),**changes)

    def test_certified_ordinary_exit_is_bounded(self):
        for code in (0,1,15,-9,-15,0xC000013A):
            self.assertTrue(restart_allowed(self.record(exit_code=code),0,2))
        self.assertFalse(restart_allowed(self.record(),2,2))
        self.assertFalse(restart_allowed(self.record(),0,0))

    def test_every_guard_or_protection_failure_stops(self):
        for change in (dict(reason='HOST MEMORY LIMIT'),dict(reason='job-attributed pagefile/swap growth'),
                dict(reason='stop requested'),dict(owned_paged_bytes=shared_owned_paging_limit_bytes()+4096),dict(release_allowed=False),
                dict(cleanup_confirmed=False),dict(paging_telemetry_closed=False),
                dict(exit_code=0xC0000005),dict(exit_code=3)):
            self.assertFalse(restart_allowed(self.record(**change),0,2),change)
        row=self.record();row.pop('reason');self.assertFalse(restart_allowed(row,0,2))
        for text in ('CUDA out of memory','allocation failed','QFN_HOST_PIN_FAILED',
                     '[qfn] FATAL: unhandled exception'):
            self.assertFalse(restart_allowed(self.record(),0,2,text))

    def test_shared_paging_allowance_is_not_a_zero_limit(self):
        limit=shared_owned_paging_limit_bytes()
        self.assertEqual(limit,64*1024**2)
        for observed in (0,4096,limit):
            self.assertTrue(restart_allowed(self.record(owned_paged_bytes=observed),0,2))
        for observed in (-1,limit+4096):
            self.assertFalse(restart_allowed(self.record(owned_paged_bytes=observed),0,2))
