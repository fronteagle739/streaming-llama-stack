"""Reference-bank owner-header regression; real model parity remains separate."""
import hashlib
import struct
import unittest
from unittest.mock import patch
from tests.combo_ultra_parity import qualification_identity,await_producer_capacity


class ResumeIdentityTest(unittest.TestCase):
    def test_resume_keeps_the_complete_bank_hash_including_owner_epoch(self):
        prior=dict(session='a'*32,owner_epoch=17)
        def bank_hash(epoch):
            return hashlib.sha256(struct.pack('<4Q',0x3145424f52503444,epoch,384,129280)+b'unchanged-vector-fixture').hexdigest()
        with patch('tests.combo_ultra_parity.secrets.token_hex',return_value='b'*32),patch('tests.combo_ultra_parity.secrets.randbits',return_value=23):
            fresh_session,fresh_epoch=qualification_identity()
            session,epoch=qualification_identity(prior)
        self.assertNotEqual(bank_hash(fresh_epoch),bank_hash(prior['owner_epoch']))
        self.assertEqual((session,epoch),(prior['session'],prior['owner_epoch']))
        self.assertEqual(bank_hash(epoch),bank_hash(prior['owner_epoch']))

    def test_invalid_retained_identity_is_refused(self):
        for session,epoch in (('g'*32,17),('a'*31,17),('a'*32,True),('a'*32,0),('a'*32,1<<63)):
            with self.subTest(session=session,epoch=epoch),self.assertRaises(ValueError):
                qualification_identity(dict(session=session,owner_epoch=epoch))

    def test_capacity_wait_keeps_full_budget_and_waits_before_launch(self):
        now=[0.];delays=[];proof={};cfg=dict(producer_host_budget_gib=6.25,min_available_gib=2)
        required=int(8.25*(1<<30));readings=iter((required-4096,required))
        def sleep(seconds):delays.append(seconds);now[0]+=seconds
        with patch('tests.combo_ultra_parity.atomic_json'):
            await_producer_capacity(cfg,proof,'unused',lambda:False,clock=lambda:now[0],sleep=sleep,available=lambda:next(readings))
        self.assertEqual(delays,[50]);self.assertEqual(len(proof['producer_admission_samples']),2)
        self.assertEqual(proof['producer_admission_samples'][-1]['required_bytes'],required)

    def test_capacity_wait_expires_without_a_producer_or_budget_change(self):
        now=[0.];proof={}
        def sleep(seconds):now[0]+=seconds
        with patch('tests.combo_ultra_parity.atomic_json'),self.assertRaisesRegex(MemoryError,'no producer launched'):
            await_producer_capacity(dict(producer_host_budget_gib=6.25,min_available_gib=2),proof,'unused',lambda:False,
                clock=lambda:now[0],sleep=sleep,available=lambda:3<<30)
        self.assertEqual(now[0],150)

    def test_capacity_wait_honours_cancellation(self):
        with patch('tests.combo_ultra_parity.atomic_json'),self.assertRaisesRegex(RuntimeError,'cancelled'):
            await_producer_capacity(dict(producer_host_budget_gib=6.25,min_available_gib=2),{},'unused',lambda:True,
                available=lambda:10<<30)


if __name__=='__main__':unittest.main()
