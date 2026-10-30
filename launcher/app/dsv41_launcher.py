"""One-click DSV4.1 launcher runtime (Windows and macOS).

The packers bake one JSON config next to this file (see launcher/README.md). This script
only starts the stack's own entry points (examples/run_preset.py, scripts/bootstrap.py),
waits on their health endpoints, opens the stack web UI and stops them gracefully.
It never kills a process it did not start, and never force-kills one it did.

    python -B dsv41_launcher.py --config launcher.json
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

WINDOWS = os.name == 'nt'


def say(text=''):
    print(time.strftime('%H:%M:%S ') + text if text else '', flush=True)


def http(url, body=None, headers=None, timeout=3):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method='GET' if body is None else 'POST',
                                     headers={'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def try_http(url, **kw):
    try:
        return http(url, **kw)
    except (OSError, ValueError, urllib.error.URLError):
        return None


def port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


class Keys:
    """Key input without blocking the progress loop: single keys in a Windows console,
    otherwise lines from stdin (macOS Terminal, or a script driving the launcher)."""
    def __init__(self):
        self.pending = []
        self.console = WINDOWS and sys.stdin is not None and sys.stdin.isatty()
        if not self.console:
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in sys.stdin:
            self.pending.append((line.strip() or '\n')[0].lower())

    def get(self):
        if self.console:
            import msvcrt
            while msvcrt.kbhit():
                self.pending.append(msvcrt.getwch().lower())
        return self.pending.pop(0) if self.pending else None

    def ask(self, prompt, choices):
        say(prompt + ('' if self.console else '   (type the letter, then Enter)'))
        self.pending.clear()
        while True:
            key = self.get()
            if key in choices:
                return key
            time.sleep(.1)


class Child:
    """A stack entry point started by this launcher, output to its own log file."""
    def __init__(self, name, argv, cwd, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.name, self.log = name, log_dir / (name + '-' + time.strftime('%Y%m%d-%H%M%S') + '.log')
        stream = self.log.open('wb')
        flags = {}
        if WINDOWS:
            # Hidden console and its own process group: Ctrl+C and closing this window do not
            # reach the stack; only this launcher's graceful stop does.
            flags['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000  # CREATE_NO_WINDOW
        else:
            flags['start_new_session'] = True
            # A launcher started from a background shell inherits SIGINT ignored; the stack's
            # graceful Ctrl+C stop path needs the default handler in its processes.
            flags['preexec_fn'] = lambda: signal.signal(signal.SIGINT, signal.SIG_DFL)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        self.proc = subprocess.Popen(argv, cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, **flags)
        say(f'Started {name} (pid {self.proc.pid}); log: {self.log}')

    def alive(self):
        return self.proc.poll() is None

    def tail(self, lines=25):
        try:
            return '\n'.join(self.log.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])
        except OSError:
            return ''

    def interrupt(self):
        """macOS: SIGINT to the outermost `python -m slstack` process. That is the stack's own
        Ctrl+C path: its handler stops the owned engine tree, then the wrappers exit normally."""
        rows = subprocess.run(['ps', '-axo', 'pid=,ppid=,command='], capture_output=True, text=True).stdout
        table = {}
        for row in rows.splitlines():
            parts = row.split(None, 2)
            if len(parts) == 3:
                table.setdefault(int(parts[1]), []).append((int(parts[0]), parts[2]))
        target, queue = None, [self.proc.pid]
        while queue and target is None:
            for child, command in table.get(queue.pop(0), []):
                if ' -m slstack ' in command + ' ':
                    target = child
                    break
                queue.append(child)
        target = target or self.proc.pid
        say(f'Asking {self.name} to stop (SIGINT to pid {target})')
        os.kill(target, signal.SIGINT)

    def wait(self, what):
        """Wait for the process to exit on its own; report progress (no fixed deadline)."""
        last = time.monotonic()
        while self.alive():
            if time.monotonic() - last > 15:
                say(f'  still stopping {what}...')
                last = time.monotonic()
            time.sleep(.5)
        say(f'{what} stopped (exit {self.proc.returncode}).')


class Launcher:
    def __init__(self, cfg):
        self.cfg, self.keys, self.children = cfg, Keys(), []
        self.root = Path(cfg['stack_root'])
        self.python = cfg['python']
        self.work = Path(cfg['runtime_root']) / 'launcher'
        self.work.mkdir(parents=True, exist_ok=True)
        self.ui = cfg['ui_url'].rstrip('/')

    # ---------- configs and commands (the stack's own entry points) ----------
    def write(self, name, data):
        target = self.work / name
        target.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        return str(target)

    def run_preset(self, *args):
        return [self.python, '-B', str(self.root / 'examples/run_preset.py'), *args]

    def bootstrap(self, config, *args):
        return [self.python, '-B', str(self.root / 'scripts/bootstrap.py'), '--config', config, *args]

    def start(self, name, argv):
        child = Child(name, argv, self.root, self.work / 'logs')
        self.children.append(child)
        return child

    # ---------- coexistence checks ----------
    def busy(self):
        """Reasons another engine holds the GPU or the ports; empty when free."""
        reasons, cfg = [], self.cfg
        studio = try_http(cfg.get('studio_url', 'http://127.0.0.1:8765') + '/presets/active')
        if studio and (studio.get('server_live') or studio.get('mac_live')) and cfg['side'] == 'rtx':
            name = (studio.get('preset') or {}).get('name', 'an LLM preset')
            reasons.append(f'Studio has "{name}" loaded (eject it in Studio to free the GPU).')
        if cfg['side'] == 'rtx' and cfg['stack'].get('gpu_uuid'):
            try:
                used = subprocess.run(['nvidia-smi', '--id=' + cfg['stack']['gpu_uuid'], '--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                      timeout=20).stdout.strip()
                if used and int(used.split()[0]) > cfg.get('gpu_busy_mib', 2048):
                    reasons.append(f'The RTX 5090 already has {used} MiB in use by another program.')
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        if cfg['side'] == 'mac':
            running = subprocess.run(['pgrep', '-fl', 'ds4-server'], capture_output=True, text=True).stdout.strip()
            if running:
                reasons.append('A ds4-server engine is already running on this Mac (probably Studio):\n    '
                               + running.splitlines()[0][:160])
        for port in cfg.get('ports', []):
            host, _, number = port.rpartition(':')
            if port_open(host or '127.0.0.1', number):
                reasons.append(f'Port {port} is already in use.')
        return reasons

    def coexist(self):
        """Never force anything: use what is running, wait for it to go, or quit."""
        health = try_http(self.ui + '/health')
        if health and health.get('status') == 'router-ready':
            key = self.keys.ask(f'A DSV4.1 stack is already running at {self.ui}.\n'
                                '  [U] use it (open the web UI)   [Q] quit', 'uq')
            if key == 'u':
                webbrowser.open(self.ui)
            return False
        while True:
            reasons = self.busy()
            if not reasons:
                return True
            say('The GPU or an engine is already in use:')
            for reason in reasons:
                say('  - ' + reason)
            studio = self.cfg.get('studio_url', 'http://127.0.0.1:8765')
            key = self.keys.ask(f'  [U] use the running one (open Studio, {studio})   [W] wait until it is free\n'
                                '  [C] start anyway (the stack still refuses if GPU/RAM do not fit)   [Q] quit', 'uwcq')
            if key == 'u':
                webbrowser.open(studio)
                return False
            if key == 'q':
                return False
            if key == 'c':
                return True
            say('Waiting for the GPU/engine to be released (checking every 10 s; press Q to give up)...')
            while self.busy():
                for _ in range(100):
                    if self.keys.get() == 'q':
                        return False
                    time.sleep(.1)

    # ---------- readiness (progress-based: wait while our processes are alive) ----------
    def until(self, test, what, children):
        last_note, started = None, time.monotonic()
        while True:
            for child in children:
                if not child.alive():
                    raise RuntimeError(f'{child.name} exited (code {child.proc.returncode}) before {what}.\n'
                                       f'--- last lines of {child.log} ---\n{child.tail()}')
            result = test()
            if result is True:
                return
            if result and result != last_note:
                say('  ' + str(result))
                last_note = result
            if self.keys.get() == 'q':
                raise KeyboardInterrupt
            if time.monotonic() - started > 60 and not result:
                say(f'  still waiting for {what}...')
                started = time.monotonic()
            time.sleep(1)

    def router_up(self):
        health = try_http(self.ui + '/health')
        return True if health and health.get('status') == 'router-ready' else None

    def load_model(self, children):
        """Ask the router to load (or pair) the engine now and report its status events."""
        rows = (try_http(self.ui + '/v1/models') or {}).get('data') or []
        if not rows:
            raise RuntimeError('The router lists no model')
        failure, done = [], []

        def load():
            # Returns only once the engine (or the pairing) is ready; the stack owns its liveness.
            try:
                http(self.ui + '/models/load', {'model': rows[0]['id']}, timeout=None)
                done.append(True)
            except urllib.error.HTTPError as exc:
                failure.append(exc.read().decode(errors='replace'))
            except OSError as exc:
                failure.append(repr(exc))
        threading.Thread(target=load, daemon=True).start()

        def ready():
            if failure:
                raise RuntimeError('Engine start failed: ' + failure[0])
            if done:
                return True
            return (try_http(self.ui + '/health') or {}).get('model_status')
        self.until(ready, 'the engine was ready', children)

    # ---------- the four launch shapes ----------
    def launch(self):
        cfg, mode, side = self.cfg, self.cfg['mode'], self.cfg['side']
        stack = self.write('stack.local.json', cfg['stack'])
        if side == 'rtx':
            if mode == 'single':
                argv = self.run_preset('--single', 'ultra-cone', '--context', cfg.get('context', '1m'),
                                       '--webui', '--config', stack)
                say('Starting DSV4.1 single machine (RTX, ULTRA exact cone, 1M context)')
            else:
                argv = self.run_preset('--preset', cfg['preset'], '--machine', 'rtx', '--config', stack)
                say(f'Starting DSV4.1 dual machine, RTX prefill side ({cfg["preset"]}); Mac: {cfg["stack"]["mac"]}')
            router = self.start('stack', argv)
            self.until(self.router_up, 'the router answered', [router])
            say('Router ready at ' + self.ui)
            if mode == 'dual':
                say('Pairing with the Mac decoder (start the Mac launcher too; waiting for it)...')
            else:
                say('Loading the engine...')
            self.load_model([router])
        else:
            if mode == 'single':
                engine = self.start('decoder', self.bootstrap(stack, 'decode'))
                url = cfg['engine_url']
                say('Starting DSV4.1 single machine (Mac ds4 decoder, DSpark, Metal layer prefill)')
                self.until(lambda: True if try_http(url + '/v1/models') else 'loading the model...',
                           'the Mac engine answered', [engine])
            else:
                engine = self.start('receiver', self.run_preset('--preset', cfg['preset'], '--machine', 'mac',
                                                                '--config', stack))
                url = cfg['engine_url']
                say(f'Starting DSV4.1 dual machine, Mac decode side and receiver/relay ({cfg["preset"]})')
                self.until(lambda: True if (try_http(url + '/health') or {}).get('ready') else None,
                           'the Mac receiver answered', [engine])
                say('Receiver ready at ' + url + '; waiting for the RTX launcher to pair...')
                rtx = cfg['rtx_broker']
                host, _, port = rtx.rpartition(':')

                def paired():
                    health = try_http(url + '/health') or {}
                    if not health.get('paired_producer_sha256'):
                        return 'waiting for the RTX to pair'
                    if not port_open(host, port):
                        return f'paired before; waiting for the RTX prefill broker at {rtx}'
                    return True
                self.until(paired, 'the RTX paired', [engine])
                say('Paired with the RTX prefill side.')
            ui = self.start('webui', self.bootstrap(self.write('ui.local.json', cfg['ui']), 'start', '--webui'))
            self.until(self.router_up, 'the web UI router answered', [engine, ui])
            self.load_model([engine, ui])
        say('READY. Opening the web UI: ' + self.ui)
        webbrowser.open(self.ui)

    # ---------- graceful stop, newest first ----------
    def stop(self):
        for child in reversed(self.children):
            if not child.alive():
                continue
            if WINDOWS or child.name == 'webui':
                say(f'Asking {child.name} to shut down (POST {self.ui}/v1/stack/shutdown)')
                if try_http(self.ui + '/v1/stack/shutdown', body={}) is None and not WINDOWS:
                    child.interrupt()
            else:
                child.interrupt()
            child.wait(child.name)

    def watch(self):
        say('Press Q to stop DSV4.1 (graceful).  Press O to open the web UI again.'
            + ('' if self.keys.console else '  (letter, then Enter)'))
        while all(child.alive() for child in self.children):
            key = self.keys.get()
            if key == 'q':
                return
            if key == 'o':
                webbrowser.open(self.ui)
            time.sleep(.2)
        for child in self.children:
            if not child.alive():
                say(f'{child.name} exited (code {child.proc.returncode}). Last lines of {child.log}:')
                print(child.tail(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if WINDOWS:
        os.system('title ' + cfg['title'])
    say(cfg['title'])
    say('Stack: ' + cfg['stack_root'])
    launcher, code = Launcher(cfg), 0
    try:
        if launcher.coexist():
            launcher.launch()
            launcher.watch()
    except KeyboardInterrupt:
        say('Stop requested.')
    except Exception as exc:  # show it, then stop what we started
        say('ERROR: ' + str(exc))
        code = 1
    finally:
        try:
            launcher.stop()
        except KeyboardInterrupt:
            say('Stop interrupted; the stack keeps shutting down on its own.')
    say('Done. Press Enter to close this window.')
    try:
        if launcher.keys.console:
            input()
        else:  # the key reader thread owns stdin
            while launcher.keys.get() is None:
                time.sleep(.2)
    except (EOFError, KeyboardInterrupt):
        pass
    return code


if __name__ == '__main__':
    sys.exit(main())
