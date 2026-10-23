import unittest
from unittest.mock import patch
from slstack.combo_crossover import crossover
from slstack.combo_release import release_barrier
from slstack.combo_session import ComboSession,prefix_digest,render_body,changed_span_tokens


class Device:
    def gpu(self,*args):return 32<<30


class SequenceTest(unittest.TestCase):
    def test_new_user_span_excludes_retained_large_history_and_template_footer(self):
        history=list(range(10000));footer=[20000,20001]
        self.assertEqual(changed_span_tokens(history+[30001,30002,30003]+footer,history+footer),3)
        self.assertEqual(changed_span_tokens(history+footer,history+footer),0)

    def test_multimodal_input_refuses_before_a_native_owner_is_created(self):
        cfg=dict();session=ComboSession(cfg,device=Device(),qualification_route='ultra')
        with patch('slstack.combo_session.Native') as native:
            with self.assertRaisesRegex(ValueError,'text documents'):
                with session.prepare(dict(messages=[dict(role='user',content=[dict(type='image_url',image_url='test')])])):pass
            native.assert_not_called()
        session.close()

    def test_zero_setting_does_not_relabel_retained_biased_state(self):
        session=ComboSession(dict(),device=Device(),qualification_route='native')
        session._note_quality(.35);self.assertEqual(session.provenance,'manual-bias')
        session._note_quality(0);self.assertEqual(session.provenance,'manual-bias')
        session._note_quality(dict(reasoning=0,tool_call=0,answer=0,code=0))
        self.assertEqual(session.provenance,'phase-bias')
        session.close()

    def test_normal_studio_history_toggle_preserves_current_tool_reasoning(self):
        body=dict(include_previous_reasoning=False,messages=[
            dict(role='assistant',content='<think>old thoughts</think>answer',reasoning_content='old'),
            dict(role='user',content='follow-up'),
            dict(role='assistant',content='tool round',reasoning_content='current')])
        rendered=render_body(body)
        self.assertNotIn('include_previous_reasoning',rendered)
        self.assertEqual(rendered['messages'][0],dict(role='assistant',content='answer'))
        self.assertEqual(rendered['messages'][2]['reasoning_content'],'current')
        self.assertEqual(body['messages'][0]['reasoning_content'],'old')
        body['include_previous_reasoning']=True
        self.assertEqual(render_body(body)['messages'],body['messages'])
        body['include_previous_reasoning']=0
        with self.assertRaises(ValueError):render_body(body)

    def test_release_certificate_is_required_before_gpu_poll(self):
        class Runtime:
            def close(self):return dict(owned_job_empty=False,owned_pid_gone=True,owned_process_signaled=True)
        class NoPoll:
            def gpu(self,*args):raise AssertionError('Retirement must precede GPU recovery')
        with self.assertRaisesRegex(RuntimeError,'certificate'):
            release_barrier(Runtime(),NoPoll(),dict(gpu_uuid='id'),32<<30,lambda:False)

    def test_measured_crossover_rejects_missing_or_slower_producer(self):
        self.assertEqual(crossover(10000,2000,10),101)
        for values in ((0,2000,10),(10000,10,10),(10000,float('nan'),10)):
            with self.assertRaises(ValueError):crossover(*values)

    def run_sequence(self,fail_release=False,changed_history=False,probe=False):
        events=[];tokens=list(range(200));cfg=dict(host_budget_gib=9,min_available_gib=2,native_host_budget_gib=8,
            gpu_uuid='id',combo_identity={})
        class Ram:
            capacity=1<<20
            def close(self):events.append('RAM close')
        class Native:
            def __init__(self,*args,restore=None,**kwargs):
                events.append('native restore' if restore else 'native open');self.checkpoint=None if not restore else cp(200);self.launch_seconds=.1
            def render(self,body):events.append('render');return tokens
            def snapshot(self):return cp(200)
            def anchor(self,ram):return dict(written_bytes=100)
            def sync(self,tokens):raise AssertionError('Long prompt must use ULTRA')
        class Producer:
            def __init__(self,*args,**kwargs):events.append('ULTRA open');self.checkpoint=cp(200);self.prefill_seconds=.2;self.directory='owned-ultra'
        def cp(position):return dict(position=position,tokens_sha256=prefix_digest(tokens,position),payload_sha256='a'*64)
        def checkpoint(owner,cp):
            session.state=Ram();session.contract={};session.payload={};return {}
        def release(owner,*args):
            events.append('release '+type(owner).__name__)
            if fail_release:raise RuntimeError('GPU recovery refused')
            return dict(certified=True)
        session=ComboSession(cfg,device=Device(),qualification_route='ultra')
        if changed_history:
            session.state=Ram();session.payload={};tokens.reverse()
            # The restored native checkpoint identifies the original prefix.
            saved=prefix_digest(list(range(200)),200)
            original=Native.__init__
            def altered(self,*args,**kwargs):original(self,*args,**kwargs);self.checkpoint['tokens_sha256']=saved
            Native.__init__=altered
        with patch('slstack.combo_session.Native',Native),patch('slstack.combo_session.ComboProducer',Producer),patch('slstack.combo_session.release_barrier',side_effect=release),patch.object(session,'_checkpoint',side_effect=checkpoint),patch.object(session,'_ram',return_value=Ram()):
            try:
                def check(owner):events.append('full-vector probe');return dict(vectors=384)
                with session.prepare(dict(messages=[dict(role='user',content='document')]),qualification_probe=check if probe else None) as (native,body,metrics):events.append('decode')
            except RuntimeError:
                self.assertTrue(fail_release);self.assertTrue(session.poison)
            except ValueError:
                self.assertTrue(changed_history);self.assertTrue(session.poison)
        session.close();return events

    def test_gpu_owners_are_sequential_with_release_between_each(self):
        events=self.run_sequence()
        self.assertLess(events.index('release Native'),events.index('ULTRA open'))
        self.assertLess(events.index('release Producer'),events.index('native restore'))
        self.assertLess(events.index('native restore'),events.index('decode'))

    def test_qualification_probe_reopens_saved_frontier_before_http_decode(self):
        events=self.run_sequence(probe=True)
        index=events.index('full-vector probe')
        self.assertEqual(events[index+1:index+4],['release Native','native restore','decode'])

    def test_failed_native_release_never_launches_producer_or_fallback(self):
        events=self.run_sequence(fail_release=True)
        self.assertNotIn('ULTRA open',events);self.assertNotIn('decode',events)

    def test_changed_history_outside_anchor_refuses_before_ultra(self):
        events=self.run_sequence(changed_history=True)
        self.assertNotIn('ULTRA open',events);self.assertNotIn('decode',events)


if __name__=='__main__':unittest.main()
