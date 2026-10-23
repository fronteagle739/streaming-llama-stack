"""Reference-bank owner-header regression; real model parity remains separate."""
import hashlib
import struct
import unittest
from unittest.mock import patch
from tests.combo_ultra_parity import qualification_identity


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


if __name__=='__main__':unittest.main()
