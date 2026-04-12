# EdgeBot Reflex — Setup Guide (Windows + VSCode)

## Project structure

```
edgebot-reflex/
├── wokwi/
│   ├── sketch.ino          ← ESP32 firmware (Arduino C++)
│   ├── diagram.json        ← Wokwi circuit (ESP32 + sensors + LEDs)
│   ├── wokwi.toml          ← Wokwi VSCode extension config
│   ├── sketch.yaml         ← Arduino CLI board config
│   └── build_and_run.bat   ← One-click compile script (Windows)
├── edgebot_bridge.py       ← Python AI bridge (Mamba + SmolVLA)
└── README.md               ← This file
```

---

## Why wokwi.toml exists

The Wokwi VSCode extension needs `wokwi.toml` to find compiled
firmware. It does NOT run `.ino` files directly — it needs
Arduino CLI to compile them first into an `.elf` binary.

---

## Step 1 — Install VSCode extensions

Open VSCode → Extensions (Ctrl+Shift+X) → install:
- **Wokwi for VS Code** (by Wokwi)
- **Python** (by Microsoft)

First-time licence: F1 → "Wokwi: Request Free License" → browser prompt.

---

## Step 2 — Compile the ESP32 firmware

Open the `wokwi/` folder in File Explorer and double-click
`build_and_run.bat`. It will:

1. Download Arduino CLI if not installed
2. Install the ESP32 board core (~150MB, one-time, ~2 min)
3. Compile `sketch.ino` → `build/esp32.esp32.esp32/sketch.ino.elf`

Subsequent runs take ~20 seconds.

---

## Step 3 — Start the Wokwi simulator

1. In VSCode: **File → Open Folder → select the `wokwi/` folder**
   (important: open the wokwi subfolder, not the parent)
2. Press **F1** → "Wokwi: Start Simulator"
3. Serial Monitor shows:
   ```
   {"status": "ready", "firmware": "EdgeBotReflex-v1"}
   {"dist_cm": 82.3, "ts": 1234}
   ```
4. LEDs respond: Red = danger reflex, Green = OK, Blue = VLA plan

---

## Step 4 — Run the Python AI bridge

Open a second terminal in VSCode:

```bash
pip install pyserial          # one-time
python edgebot_bridge.py
```

Runs in stub mode by default (no GPU needed). Output:
```
[Mamba  14:32:01] dist= 82.0cm  cmd=FWD        latency=0.04ms
[SmolVLA 14:32:01] plan=PLAN_FWD     latency=0.10ms
[Status] dist=12.3cm | mamba=STOP | smolvla=PLAN_STOP | arbiter=STOP
```

To connect the bridge to Wokwi's virtual serial port, find the
COM port in the Wokwi panel and set `SERIAL_PORT = "COMx"` in
`edgebot_bridge.py`. Otherwise FakeSerial runs automatically.

---

## Step 5 — Browser fallback (no install needed)

1. https://wokwi.com → New Project → ESP32
2. Paste `sketch.ino` into the editor
3. Click **{...}** and paste `diagram.json`
4. Press Play — use the Serial Monitor to watch sensor frames
5. Run `python edgebot_bridge.py` separately (uses FakeSerial)

---

## Step 6 — Real models (needs GPU)

```python
USE_REAL_MODELS = True   # in edgebot_bridge.py
```

```bash
pip install transformers torch accelerate lerobot
```

Models loaded automatically:
- `tiiuae/falcon-mamba-7b-instruct` (~14GB, 8GB+ VRAM)
- `lerobot/smolvla-base` (~900MB, runs on CPU)

For laptop (no GPU): use the GGUF quantized Falcon Mamba via
`llama-cpp-python` — swap `load_falcon_mamba()` accordingly.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "wokwi.toml not found" | Open the `wokwi/` subfolder in VSCode, not the parent |
| "elf file not found" | Run `build_and_run.bat` first |
| "Wokwi license required" | F1 → "Wokwi: Request Free License" |
| No sensor data in bridge | Leave SERIAL_PORT as default; FakeSerial activates automatically |
| Arduino CLI download fails | Get it from https://arduino.cc/en/software, add to PATH |
