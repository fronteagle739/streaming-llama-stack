# One-click DSV4.1 launchers

Small packers that turn the settings in `config/` into a double-click launcher:
a Windows `.exe` for the RTX machine and a macOS `.app` (plus `.dmg`) for the Mac.
Each launcher starts DeepSeek V4.1 Flash with the stack's fastest example setup,
waits until it is ready and opens the stack's web UI.

| Launcher | Machine | What it runs (the README's fastest setup) |
|---|---|---|
| **DSV41 Single.exe** | RTX | `examples/run_preset.py --single ultra-cone --context 1m --webui`: packed FP4, exact cone, 1M context |
| **DSV41 Single.app** | Mac | `slstack decode` with the ds4 backend: Mac decode FP4 file, DSpark, Metal layer prefill on |
| **DSV41 Dual.exe** | RTX | `examples/run_preset.py --preset dsv41-ultra-cone-fp4 --machine rtx`: the RTX prefill side |
| **DSV41 Dual.app** | Mac | `examples/run_preset.py --preset dsv41-ultra-cone-fp4 --machine mac`: Mac decode and the receiver's direct-TCP relay |

For two machines, start **DSV41 Dual** on both, in either order. Each waits for the
other: the RTX pairs with the Mac receiver (`192.168.3.1:8180`), and the Mac waits for
that pairing and for the RTX prefill broker (`192.168.3.3:8090`). The RTX opens the web
UI at `http://127.0.0.1:8088`. The Mac also opens one there: the stack router in
`general.mode=remote`, sending its chats to the paired receiver.

The launchers start only the stack's own entry points (`examples/run_preset.py`,
`scripts/bootstrap.py`). They do not start engines themselves. Readiness is
progress-based: a launcher waits as long as its stack processes are alive, and watches
`/health` and the router's status events.

## Status window and controls

The `.exe` runs in its console window. The `.app` opens a Terminal window. The window
shows each step, the log file of every process it starts, and the last log lines if
anything exits early.

- **Q** stops DSV4.1 gracefully. The router gets `POST /v1/stack/shutdown`; the Mac
  receiver and decoder get the stack's own Ctrl+C path. The launcher then waits until
  each process has exited. Nothing is force-killed.
- **O** opens the web UI again.
- If you close the window instead, the stack keeps running. The next launch finds it
  and offers to use it.

On a Mac, type the letter and press Enter.

## When the GPU or an engine is already in use

Before it starts anything, the launcher checks for:
- a DSV4.1 stack already answering on its web UI port;
- another app's LLM already loaded (optional `studio_url`: a Streaming Studio daemon's `/presets/active`);
- RTX VRAM in use (`nvidia-smi`), or a running `ds4-server` on the Mac;
- its stack ports already taken.

It then explains what it found and offers these choices:
- **U**: use the running one. This opens that stack's web UI, or the other app's page (`studio_url`).
- **W**: wait until it is free.
- **C**: start anyway. The stack's own GPU and RAM admission still refuses if it does not fit.
- **Q**: quit.

It never stops another program; free the GPU in the program that holds it. Nothing else in the launcher depends on Studio.

## Build

The settings live in `config/single.json` and `config/dual.json`. Each has an `rtx` and
a `mac` section. Put your own paths, GPU UUID and pairing token in
`config/single.local.json` and `config/dual.local.json`, which Git ignores. Set `runtime_root` there too (for example `launcher/dist/runtime` in this checkout; logs, the stack's core Python environment and pairing state live there). They are
merged over the defaults, and the packer refuses while any `SET_...` placeholder is left.
`pair_token` at the top of `dual.local.json` must be the same on both machines: a random
16-128 character URL-safe string.

Windows (uses the C# compiler that ships with Windows):

```powershell
powershell -File launcher/pack_windows.ps1 -Mode single -Shortcuts   # -> launcher/dist/DSV41 Single.exe
powershell -File launcher/pack_windows.ps1 -Mode dual   -Shortcuts   # -> launcher/dist/DSV41 Dual.exe
```

macOS (uses `codesign`, `hdiutil` and the Dock defaults, which ship with macOS):

```sh
sh launcher/pack_macos.sh single --out "$HOME/Applications/DSV41 Launchers" --dock
sh launcher/pack_macos.sh dual   --out "$HOME/Applications/DSV41 Launchers" --dock
```

A packer bakes the settings, the launcher runtime (`app/dsv41_launcher.py`) and the
Python path into the launcher. Repack after you change a setting. Each build overwrites
the same `.exe` or `.app` in place.

Other packer options:
- `-Python` / `--python`: the Python 3.12 that runs the stack. Default: `python` on
  Windows, `python3.12` on macOS.
- `-StackRoot` / `--stack-root`: the stack checkout to run. Default: this repository.

`--dock` backs up `~/Library/Preferences/com.apple.dock.plist` first. Windows does not
let programs pin themselves to the taskbar, so pin it yourself: right-click the Desktop
shortcut and choose **Pin to taskbar** (on Windows 11, first **Show more options**).

### Before the first launch

The stack must be set up on each machine as in the main README:
- **RTX:** either `slstack.ps1 install --all`, or point the local config at an already
  installed producer. That means `llama_producer`, `producer_sha256`,
  `producer_source_revision`, `studio_root`, `ds4_command`, `model_receipt` and the
  native FFN DLL in `env_overrides`. Pairing accepts an installed producer when this
  checkout's `engines/streaming-llama` source matches its recorded revision.
- **Mac:** build ds4 once with
  `python3.12 examples/run_preset.py --preset dsv41-ultra-cone-fp4 --machine mac --config <stack config> --setup`.
- **First start:** `scripts/bootstrap.py` creates the stack's core Python environment in
  `runtime_root`, from the pinned `requirements.txt`.

### Icons

`icons/single.svg` shows one robot riding a river, and `icons/dual.svg` shows two.
`python -B launcher/icons/build_icons.py` renders them with a headless Chromium browser
and writes `icons/built/*.ico` (16-256 px) and `*.icns`.

### First launch on a new machine

- **Windows:** a locally built `.exe` carries no download mark, so SmartScreen does not
  ask. In the dual setup, Windows may ask once to allow Python on the network for the
  prefill broker.
- **macOS:** a locally built, ad-hoc-signed app opens without a Gatekeeper prompt. A
  copy taken from the `.dmg` to another Mac needs right-click > **Open** once. On the
  first dual start, macOS may ask once to allow Python on the local network for the
  receiver relay.
