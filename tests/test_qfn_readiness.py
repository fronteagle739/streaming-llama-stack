"""Exercise real readiness framing and identity refusal over loopback sockets."""
import importlib.util
import json
from pathlib import Path
import socket
import threading
import unittest

spec=importlib.util.spec_from_file_location('qfn_readiness',Path(__file__).resolve().parents[1]/'examples/dual-machine/qfn/readiness.py')
ready=importlib.util.module_from_spec(spec);spec.loader.exec_module(ready)


class ReadinessTests(unittest.TestCase):
    def exchange(self,mutate=lambda row:None,close=False):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
        port=listener.getsockname()[1];requests=[]
        def server():
            with listener,listener.accept()[0] as peer:
                pending=b''
                while b'\n' not in pending:pending+=peer.recv(4096)
                requests.append(json.loads(pending))
                if close:return
                row=dict(ok=True,event='paired',family='qwen4exp',kv_bits=16,context=131072,
                    model_sha256='a'*64,compatibility_sha256='b'*64)
                mutate(row)
                data=(json.dumps(dict(ok=True,event='ready'))+'\n'+json.dumps(row)+'\n').encode()
                # Deliberately split frames; the client must preserve partial JSON.
                peer.sendall(data[:7]);peer.sendall(data[7:])
        thread=threading.Thread(target=server);thread.start()
        try:return ready.wait_ready(port,'c'*64,'a'*64,'b'*64,2),requests
        finally:thread.join(3);self.assertFalse(thread.is_alive())

    def test_loaded_pair_and_partial_frames(self):
        row,requests=self.exchange();self.assertEqual(row['context'],131072)
        self.assertEqual(requests[0]['op'],'pair');self.assertNotIn('tokens',requests[0])

    def test_identity_mismatches_fail(self):
        for key,value in [('model_sha256','d'*64),('compatibility_sha256','d'*64),
                          ('kv_bits',8),('family','deepseek41'),('event','ready-wrong')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.exchange(lambda row:row.update({key:value}))

    def test_rejection_and_disconnect_fail(self):
        with self.assertRaises(RuntimeError):self.exchange(lambda row:row.update(ok=False,error='bad pairing'))
        with self.assertRaises(RuntimeError):self.exchange(close=True)


if __name__=='__main__':unittest.main()
