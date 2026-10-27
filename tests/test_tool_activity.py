import json
import unittest
from unittest.mock import Mock
from slstack.config import load
from slstack.router import Stack
from slstack.tool_activity import activity, without_activity
from slstack.ds4_tool_parse import finish_message
from slstack.combo_single import Splitter
from slstack.media_profiles import python_for, profile_for


def call(i):
    return dict(id='call-'+str(i),type='function',function=dict(name='set_quality_speed',arguments='{"value":0.35}'))


def event(delta, finish=None):
    return dict(choices=[dict(index=0,delta=delta,finish_reason=finish)])


class ToolActivityTests(unittest.TestCase):
    def stack(self):
        engine=Mock();engine.models=None
        stack=Stack(load(overrides={'gpu_uuid':'GPU-test'}),engine)
        stack.tools.call=Mock(return_value=dict(value=.35))
        return stack,engine

    def test_stream_runs_beyond_eight_tool_rounds_and_keeps_activity_out_of_context(self):
        stack,engine=self.stack()
        engine.chat_events.side_effect=[iter([event(dict(tool_calls=[dict(index=0,**call(i))]),'tool_calls')]) for i in range(12)]+[iter([event(dict(content='Done.'),'stop')])]
        rows=list(stack.chat_events(dict(messages=[dict(role='user',content='Continue')])) )
        self.assertEqual(stack.tools.call.call_count,12)
        self.assertEqual(engine.chat_events.call_count,13)
        text=''.join(c['delta'].get('content','') for row in rows for c in row.get('choices',[]))
        self.assertEqual(text.count('```stack-tool'),24)
        self.assertIn('"status": "running"',text)
        self.assertIn('"status": "complete"',text)
        self.assertEqual(rows[-1]['choices'][0]['finish_reason'],'stop')
        self.assertEqual(without_activity(text).strip(),'Done.')
        context=engine.chat_events.call_args.args[0]['messages']
        self.assertFalse(any('stack-tool' in str(message.get('content')) for message in context))

    def test_nonstream_has_no_tool_round_limit(self):
        stack,engine=self.stack()
        engine.chat.side_effect=[dict(choices=[dict(message=dict(role='assistant',content='',tool_calls=[call(i)]))]) for i in range(12)]+[dict(choices=[dict(message=dict(role='assistant',content='Done.'),finish_reason='stop')])]
        result=stack.chat(dict(messages=[dict(role='user',content='Continue')]))
        self.assertEqual(stack.tools.call.call_count,12)
        self.assertIn('Done.',result['choices'][0]['message']['content'])

    def test_stream_forwards_prompt_and_decode_counts_before_completion(self):
        stack,engine=self.stack()
        prompt=dict(prompt_n=2000,cache_n=1200,predicted_n=0)
        token=dict(prompt_n=2000,cache_n=1200,predicted_n=1)
        engine.chat_events.return_value=iter([
            dict(choices=[],timings=prompt),
            dict(event(dict(content='Hi')),timings=token),event({},'stop')])
        rows=list(stack.chat_events(dict(messages=[dict(role='user',content='Hello')])))
        self.assertEqual(rows[0]['choices'],[])
        self.assertEqual(rows[0]['timings'],prompt)
        self.assertEqual(rows[1]['timings'],token)
        self.assertEqual(rows[-1]['timings'],token)

    def test_malformed_dsml_is_a_diagnostic_not_answer_text_or_an_execution(self):
        raw='<｜DSML｜｜DSML｜ calls [generate_image] <｜DSML｜ parameter="prom">broken'
        splitter=Splitter(True);pieces=[]
        for letter in 'Thinking</think>Start.'+raw:pieces.extend(splitter.feed(letter))
        result=finish_message('Thinking</think>Start.'+raw,has_tools=True,think=True,finish='stop')
        self.assertEqual(result['calls'],[])
        self.assertEqual(result['content'],'Start.')
        self.assertEqual(result['tool_error']['raw'],raw)
        self.assertNotIn('DSML',''.join(text for _,text in pieces))
        self.assertEqual(splitter.remainder('content',result['content']),'')

    def test_activity_cannot_escape_its_fence_and_is_removed_only_from_assistant_context(self):
        trace=activity(dict(id='one',function=dict(name='generate_image',arguments=json.dumps(dict(prompt='```\nunsafe-looking text')))),'running')
        self.assertEqual(without_activity(trace).strip(),'')
        stack,_=self.stack()
        body=stack.prepare_chat(dict(messages=[dict(role='assistant',content=trace+'Ready'),dict(role='user',content=trace)]))
        self.assertEqual(body['messages'][0]['content'].strip(),'Ready')
        self.assertEqual(body['messages'][1]['content'],trace)

    def test_qwen_worker_is_isolated_and_portable(self):
        cfg=dict(runtime_root='portable-runtime',studio_python='existing-general-python')
        self.assertEqual(profile_for('anima-1.0'),'studio')
        self.assertEqual(python_for(cfg,'krea2-native'),'existing-general-python')
        self.assertIn('media-qwen21-venv',python_for(cfg,'qwen-image-2.1'))
        self.assertNotIn('existing-general-python',python_for(cfg,'qwen-image-2.1'))


if __name__=='__main__':unittest.main()
