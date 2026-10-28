# Setup and chat walkthrough

Use the exact Flash 1017 Q4 model described in the [README](../README.md#requirements). Media and LoRA workflows are experimental.

## Install

```powershell
git clone --recurse-submodules https://github.com/fronteagle739/streaming-llama-stack.git
cd streaming-llama-stack
./slstack.ps1 install --all
./slstack.ps1 install --all --check
```

Create `my.local.json` with `model`, `gpu_uuid` and explicit host budgets. Start with the README example; other keys come from [config.example.json](../config.example.json). Installation builds the producer, `ds4-decode.dll`, native server and chat UI. It does not fetch model weights.

## Start a conversation

```powershell
./slstack.ps1 --config my.local.json check
powershell -File scripts/start-dsv41-single-262k.ps1 -DryRun
powershell -File scripts/start-dsv41-single-262k.ps1 -WebUI
```

Open the printed router URL. The first prompt loads the producer. Full-file hashing runs once when the model identity is first recorded and reports progress. Ask a short question, append a document, then ask a follow-up: the producer retains the conversation and processes new tokens. Appended prefill shows processing progress; completed replies show decode tok/s.

The preset uses growable f32 context at 262,144 tokens, 49,152-token sweeps with a 61,440 fold cap, DirectStorage read parts 8 and a 512 MiB expert-tier reserve. The 1M script admits 999,424 tokens with bf16 context. `-DryRun` prints resolved settings without starting an engine.

Ask the model to switch itself to Max speed using its quality/speed tool. The tool reports requested/applied settings. For precise code, select Max quality or Balanced. [Live controls](../docs/LIVE_GENERATION.md)

## Media

Select existing model/component folders in Settings → Stack. Anima requires its pipeline directory and matching text encoder; other families have their own components. Request a small image, such as a red lighthouse, with explicit dimensions, steps and seed. The tool card shows progress and returns the image. Uploads and generated images can condition later tools. [Media setup](../docs/MEDIA_SETUP_VERIFICATION.md)

The earlier native-server example used `start --serving native`: [chat](walkthrough/chat-desktop.png), [phone chat](walkthrough/chat-phone.png), [quality tool](walkthrough/quality-desktop.png) and [image](walkthrough/image-desktop.png). These screenshots illustrate that native path. Current checkpoint behavior is described in [media and state](../docs/STACK_MEDIA_AND_STATE.md).

## Stop or change preset

```powershell
powershell -File scripts/start-dsv41-single-262k.ps1 -Stop
powershell -File scripts/start-dsv41-single-1m.ps1 -WebUI
# Stop this preset before starting another:
powershell -File scripts/start-dsv41-single-1m.ps1 -Stop
```

Each launcher accepts `-Config`, `-Model`, `-DryRun`, `-WebUI` and `-Stop`. Use the same config to stop. Foreground launchers stop with Ctrl-C. Native shutdown drains storage work and releases host/GPU memory.

For approximate YOCO use the corresponding `*-ultra-yoco.ps1` launcher. [Quality and timing](../docs/ULTRAYOCO_QUALIFICATION.md). For optional Mac decode, follow [two-machine setup](../docs/TWO_MACHINE.md).
