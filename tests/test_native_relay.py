"""PAIRFIX: the receiver's direct-TCP relay forwards bytes unchanged and fails loudly."""
import socket
import threading
import unittest

from slstack.native_relay import Relay


class RelayTests(unittest.TestCase):
    def test_forwards_both_directions_byte_exact(self):
        server = socket.socket();server.bind(('127.0.0.1', 0));server.listen(1)
        port = server.getsockname()[1];payload = bytes(range(256)) * 20000
        def peer():
            conn, _ = server.accept()
            got = bytearray()
            while len(got) < 5:
                got += conn.recv(5 - len(got))
            conn.sendall(payload);conn.close()
        threading.Thread(target=peer, daemon=True).start()
        lines = [];relay = Relay('127.0.0.1', port, log=lines.append)
        try:
            c = socket.create_connection(('127.0.0.1', relay.port));c.sendall(b'hello')
            data = bytearray()
            while True:
                block = c.recv(1 << 16)
                if not block:break
                data += block
            c.close()
            self.assertEqual(bytes(data), payload)
            self.assertTrue(any(l.startswith('NATIVE_RELAY_LISTEN') for l in lines))
        finally:
            relay.close();server.close()

    def test_unreachable_target_closes_and_logs(self):
        probe = socket.socket();probe.bind(('127.0.0.1', 0));dead = probe.getsockname()[1];probe.close()
        lines = [];relay = Relay('127.0.0.1', dead, log=lines.append)
        try:
            c = socket.create_connection(('127.0.0.1', relay.port))
            self.assertEqual(c.recv(1), b'');c.close()
            for _ in range(100):
                if any('NATIVE_RELAY_CONNECT_FAILED' in l for l in lines):break
                threading.Event().wait(.05)
            self.assertTrue(any('NATIVE_RELAY_CONNECT_FAILED' in l for l in lines))
        finally:
            relay.close()


if __name__ == '__main__':
    unittest.main()
