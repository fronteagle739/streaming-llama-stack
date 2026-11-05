import json,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from slstack.config import ROOT
from slstack.qfn_installation import launch_arguments,validate_launch_contract,QfnReceiver
from slstack.model_compatibility import digest
from slstack.pairing import installation_handshake


class InstallationTests(unittest.TestCase):
    def test_complete_launch_flags_are_bound_but_output_is_per_run(self):
        root=ROOT/'runtime/fixtures'/('qfn-install-'+uuid.uuid4().hex);root.mkdir(parents=True)
        argv=['mac','--context','131072','--option-ds4-qwen4-mtp-depth','3','--output','first']
        row={'profile':{'mac':{'context':131072}},'model_settings_sha256':'a'*64,
             'role_argv':{'mac':launch_arguments(argv),'rtx':['rtx']}}
        file=root/'resolved.json';file.write_text(json.dumps(row))
        self.assertEqual(validate_launch_contract(file,'mac',argv[:-1]+['second']),digest(row))
        for candidate in (argv+['--warm-weights','0'],[x if x!='131072' else '8192' for x in argv]):
            with self.assertRaisesRegex(ValueError,'Actual peer flags'):validate_launch_contract(file,'mac',candidate)
        row['ignored_capability']=True;file.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError,'Unknown or incomplete'):validate_launch_contract(file,'mac',argv)

    def test_selected_producer_binary_and_revision_are_not_substitutable(self):
        r=QfnReceiver.__new__(QfnReceiver);r.cfg={'producer_sha256':'a'*64,'producer_source_revision':'b'*40}
        r._expected_extra=lambda:{}
        with self.assertRaisesRegex(ValueError,'executable/revision'):
            r.validate_pair_extra({'producer_sha256':'c'*64,'producer_source_revision':'b'*40,'tier':'accurate'})

    def test_cached_installation_requalifies_after_preset_or_source_pin_changes(self):
        root=ROOT/'runtime/fixtures'/('qfn-install-'+uuid.uuid4().hex);root.mkdir(parents=True)
        identity={'producer_sha256':'a'*64,'tier':'accurate','preset_sha256':'b'*64}
        health={'receiver_sha256':'c'*64,'paired_producer_sha256':'a'*64,'paired_tier':'accurate'}
        file=root/'peer.json';calls=[]
        def request(route,body):calls.append(route);return {'status':'qualified'}
        installation_handshake(request,'http://127.0.0.1:18796',identity,health,file,lambda _:None)
        self.assertEqual(calls,['/pair','/pair/qualification','/pair/qualification','/pair/accept','/decode/setup'])
        calls.clear();installation_handshake(request,'http://127.0.0.1:18796',identity,health,file,lambda _:None)
        self.assertEqual(calls,['/decode/setup'])
        calls.clear();installation_handshake(request,'http://127.0.0.1:18796',dict(identity,preset_sha256='d'*64),health,file,lambda _:None)
        self.assertIn('/pair/accept',calls)

    def test_family_adapter_preserves_full_qfn_sampler_ranges(self):
        from slstack.paired_generation import decode_request
        from slstack.generation import defaults
        r=QfnReceiver.__new__(QfnReceiver);r._vocab=248320
        body={'logit_bias':{'200000':-2},'top_k':2048,'quality_speed':7.5}
        self.assertEqual(r.normalize_request(body),body)
        with self.assertRaises(ValueError):decode_request(body)
        profile=defaults();profile['sampler']['top_k']=2048;profile['value']=7.5
        for phase in profile['phases'].values():phase.update(top_k=4096,bias=7.5)
        normalized=r.normalize_request({'slstack_generation':profile})
        self.assertTrue(normalized['slstack_phase_samplers']['answer']['top_k']==4096)
