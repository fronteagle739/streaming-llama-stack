"""Direct-TCP native session path on the Mac (no SSH tunnel).

macOS Local Network privacy refuses LAN connections from a freshly built,
unapproved ds4-server (errno 65, EHOSTUNREACH), while the receiver controller's
Python may connect to the RTX broker. The receiver therefore gives the engine a
loopback endpoint and forwards each connection, byte for byte, to the paired
RTX broker over the existing link. The protocol, token and every owner SHA-256
check are unchanged (the relay sees opaque bytes). A forward that cannot reach
the RTX closes the engine's connection and logs NATIVE_RELAY_CONNECT_FAILED,
so the session fails loudly; the SSH-tunnel route (a loopback producer address)
stays available as the explicit fallback and never engages silently.
"""
import socket
import threading
import time

CHUNK = 1 << 20


class Relay:
    def __init__(self, target_host, target_port, log=print):
        self.target = (target_host, int(target_port))
        self.log = log
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', 0))
        self.server.listen(16)
        self.port = self.server.getsockname()[1]
        self.closed = False
        self.lock = threading.Lock()
        self.sockets = set()
        self.thread = threading.Thread(target=self._accept, name='native-relay', daemon=True)
        self.thread.start()
        self.log(f'NATIVE_RELAY_LISTEN local=127.0.0.1:{self.port} target={target_host}:{target_port} transport=direct-tcp')

    def _accept(self):
        while not self.closed:
            try:
                client, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._connect, args=(client,), daemon=True).start()

    def _connect(self, client):
        started = time.monotonic()
        try:
            upstream = socket.create_connection(self.target, timeout=10)
        except OSError as exc:
            self.log(f'NATIVE_RELAY_CONNECT_FAILED target={self.target[0]}:{self.target[1]} errno={exc.errno} error={exc}')
            client.close()
            return
        upstream.settimeout(None)
        for sock in (client, upstream):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for option in (socket.SO_RCVBUF, socket.SO_SNDBUF):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, option, 8 << 20)
                except OSError:
                    pass
        with self.lock:
            self.sockets.update((client, upstream))
        totals = [0, 0]
        def pump(src, dst, index):
            buffer = bytearray(CHUNK)
            view = memoryview(buffer)
            try:
                while True:
                    n = src.recv_into(view)
                    if not n:
                        break
                    dst.sendall(view[:n])
                    totals[index] += n
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
        down = threading.Thread(target=pump, args=(upstream, client, 1), daemon=True)
        down.start()
        pump(client, upstream, 0)
        down.join()
        for sock in (client, upstream):
            sock.close()
        with self.lock:
            self.sockets.difference_update((client, upstream))
        if totals[1] >= 64 << 20:
            seconds = time.monotonic() - started
            self.log(f'NATIVE_RELAY_STREAM bytes_to_mac={totals[1]} bytes_to_rtx={totals[0]} seconds={seconds:.3f} mb_per_s={totals[1] / max(seconds, 1e-9) / 1e6:.1f}')

    def close(self):
        self.closed = True
        try:
            self.server.close()
        except OSError:
            pass
        with self.lock:
            for sock in list(self.sockets):
                try:
                    sock.close()
                except OSError:
                    pass
            self.sockets.clear()
