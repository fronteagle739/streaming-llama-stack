"""General decoder reservations end on success, failure and cancellation."""
import threading
import unittest
from unittest.mock import Mock

from slstack.general_engine import GeneralEngines


class GeneralRequests(unittest.TestCase):
    def fixture(self,*,fail=False):
        engine=GeneralEngines.__new__(GeneralEngines)
        engine.cfg={'general':{'mode':'streaming'}};engine.lock=threading.RLock()
        engine.select=Mock();engine.remote=None;engine.llm=None
        remote=Mock();job=Mock();received=[]
        def events(body):
            received.append(body)
            yield {'choices':[{'delta':{'content':'first'}}]}
            if fail:raise RuntimeError('decode failed')
            yield {'choices':[{'delta':{'content':'second'}}]}
        remote.events.side_effect=events
        def ready():engine.remote=remote;engine.llm=job
        engine.ready=ready
        return engine,remote,job,received

    def test_sampler_fields_survive_and_success_releases_the_producer(self):
        engine,remote,job,received=self.fixture()
        body=dict(messages=[{'role':'user','content':'test'}],temperature=.8,top_p=.9,
                  top_k=37,min_p=.02,seed=123,max_tokens=256)
        self.assertEqual(len(list(engine.chat_events(body))),2)
        self.assertEqual(received,[body])
        job.stop.assert_called_once();remote.close.assert_called_once()
        self.assertIsNone(engine.llm);self.assertIsNone(engine.remote)

    def test_failure_releases_the_producer(self):
        engine,remote,job,_=self.fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError,'decode failed'):list(engine.chat_events({}))
        job.stop.assert_called_once();remote.close.assert_called_once()

    def test_browser_cancellation_releases_the_producer(self):
        engine,remote,job,_=self.fixture()
        stream=engine.chat_events({});next(stream);stream.close()
        job.stop.assert_called_once();remote.close.assert_called_once()


if __name__=='__main__':unittest.main()
