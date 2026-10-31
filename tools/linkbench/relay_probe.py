"""Runs the production Mac relay (slstack.native_relay.Relay) for linkbench.

  python3 relay_probe.py loopback TARGET_PORT       # unmodified Relay on 127.0.0.1
  python3 relay_probe.py lan BIND_IP PORT TARGET_PORT  # same pump, listening on the link

'loopback' measures the pump alone (engine->relay->local target).  'lan' mirrors
the production order (engine -loopback-> relay -LAN-> RTX) as
(RTX -LAN-> relay -loopback-> Mac target): the same hops and the same pump code.
Prints RELAY_PORT=<n> and serves until killed.
"""
import socket, sys, time
sys.path.insert(0, sys.argv[-1] if sys.argv[-1].endswith('slstack') else '.')
from native_relay import Relay

mode = sys.argv[1]
if mode == 'loopback':
    r = Relay('127.0.0.1', int(sys.argv[2]))
else:
    class LanRelay(Relay):
        def __init__(self, bind_ip, port, target_port):
            orig = socket.socket.bind
            def bind(sock, addr):
                return orig(sock, (bind_ip, port) if addr == ('127.0.0.1', 0) else addr)
            socket.socket.bind = bind
            try:
                super().__init__('127.0.0.1', target_port)
            finally:
                socket.socket.bind = orig
    r = LanRelay(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
print(f'RELAY_PORT={r.port}', flush=True)
while True:
    time.sleep(3600)
