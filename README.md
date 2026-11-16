# acoustic-detector

A Raspberry Pi 5 microphone sensor that hears a nearby drone and reports it: an ATAK marker
(CoT over TCP to OpenTAKServer) and a Meshtastic text via the blufor-tracker mesh bridge. Built as
an "art of the possible" ISR training demo. It is a sensor only: it never controls, steers or
engages anything.

Everything runs on the Pi CPU. No Hailo, no internet needed. Google Gemini is an optional
second opinion that you can switch on.

## How it works

Every 0.5 s the last ~1 s of audio (16 kHz mono) goes through three independent checks:

| Component | What it is | Needs |
|---|---|---|
| **Prop comb** | Hand-built spectral check. A multirotor makes a harmonic comb whose fundamental (rotation rate x blades) sits around 150–700 Hz for a 5" FPV quad. It's matched per 128 ms frame, so throttle changes don't smear it. Petrol engines also make combs, but at 20–60 Hz, so they score low. | nothing |
| **YAMNet** | Google's pretrained AudioSet classifier (521 classes), run with LiteRT. AudioSet has **no "drone" class**, so the drone score counts `Aircraft`, `Aircraft engine`, `Propeller, airscrew`, `Helicopter`, `Fixed-wing aircraft, airplane`, `Light engine (high frequency)`, `Buzz`, `Insect`, `Bee, wasp, etc.`, `Mosquito`, `Fly, housefly` and `Whir` (insect classes weighted lower). A separate **confuser** score counts `Lawn mower`, `Chainsaw`, `Power tool`, `Motorcycle`, vehicle and engine classes, `Hair dryer`, `Vacuum cleaner` and `Speech`. | `models/yamnet.tflite` + `ai-edge-litert` |
| **My drone** (calibration) | Cosine similarity to reference embeddings built from clips of *your* drone (64-band spectral shape, plus the YAMNet score vector when the model is loaded). If you also enrol background clips, the score becomes "closer to my drone than to this field". | 3–5 recorded clips |

Fusion: `confidence = base + 0.45·yamnet·(1−base) − 0.5·confuser`, where `base` is the prop-comb
score, or the average of prop comb and calibration once a drone is enrolled. YAMNet adds evidence
but can't trigger an alert on its own, and a confident "lawn mower" call pulls the score down.

Alerting: the confidence is smoothed, then a **hysteresis** state machine raises a detection after
2 windows ≥ 0.65 and clears after 6 windows < 0.40. Another detection within the **60 s cooldown**
is logged as `suppressed` and sends no mesh text. All of these can be tuned in the env file.

On a detection:
- **ATAK**: a CoT `a-u-A-M-F-Q` (unknown UAV; set `COT_AFFILIATION=h` for hostile) placed **at the
  sensor's position**, with `ce` set to the detection radius and a remarks field stating that it
  shows presence near the sensor, not the target's location. While the detection stays active the
  marker is re-sent every 15 s. A friendly sensor-point marker (`a-f-G-E-S`, "ACOUSTIC OP1") is sent
  every 60 s.
- **Mesh**: `{"text": "ACOUSTIC: drone detected conf 0.87 @ OP1", "to": null, "channel": 0,
  "source": "acoustic-detector"}` published to `meshbus/tx` on 127.0.0.1:1883. The mesh bridge owns
  the radio, and this tool never opens the serial port.
- **Gemini** (if enabled): the last 4 s are sent for a second opinion, which is shown in its own panel.
- Each event is appended to `/var/lib/acoustic-detector/detections.jsonl`.

If OpenTAKServer or the MQTT broker is down, the sensor logs it, keeps detecting, and reconnects
with backoff.

## Hardware

- Raspberry Pi 5 (64-bit Pi OS Bookworm), SSD boot.
- **A USB microphone.** The Pi 5 has no built-in mic. Any class-compliant USB mic or USB sound
  card works. An omnidirectional lavalier on a cheap USB dongle is fine. A foam or fur windshield
  makes a big difference outdoors.
- Optional: the BFT kit (OpenTAKServer + Heltec V3 Meshtastic node via `mesh-bridge`) for ATAK
  markers and mesh texts.
- Test target: a 5" FPV quad.

## Install (Pi)

```bash
git clone <this repo> && cd acoustic-detector
sudo bash deploy/install.sh
sudo nano /etc/acoustic-detector.env     # set SENSOR_NAME, SENSOR_LAT, SENSOR_LON
```

The install script is idempotent and non-interactive. It installs apt packages, creates an
`acoustic` service user in the `audio` group, puts the code in `/opt/acoustic-detector` with a venv,
downloads the YAMNet model (~4 MB), creates `/var/lib/acoustic-detector`, copies the env example to
`/etc/acoustic-detector.env` if one doesn't exist, and installs the systemd unit **without enabling
it**. pi-profiles starts and stops it:

```bash
sudo pi-profiles acoustic-detector   # mic sensor on its own
sudo pi-profiles bft-acoustic        # + opentakserver + mesh-bridge (ATAK markers + mesh alerts)
journalctl -u acoustic-detector -f
```

Web UI: `http://<pi-ip>:8100`.

`ACOUSTIC_INPUT` in the env file selects the input: `auto` (USB mic if `arecord -l` finds one,
otherwise simulate), `mic`, `simulate`, or the path to a clip to replay in a loop. To pick a
specific device, list them with
`/opt/acoustic-detector/.venv/bin/python -m acoustic --list-devices` and set `AUDIO_DEVICE`.

## Run on a laptop (no hardware)

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements-dev.txt
scripts/fetch_model.sh                    # optional: YAMNet
.venv/bin/python -m acoustic --simulate   # http://localhost:8100
.venv/bin/python -m pytest
```

`--simulate` plays a looping synthetic soundscape: background, a drone pass, background, a
lawnmower, background, a generator. The UI shows the "sim truth" next to the detector's view. The
synthetic drone is a pure harmonic comb, so the prop-comb check catches it but **YAMNet doesn't
recognise it** (it hears "Music"). To see YAMNet properly, replay a real recording:

```bash
.venv/bin/python -m acoustic --file my_fpv_pass.wav --loop   # WAV / FLAC / OGG / MP3
```

## Teach it your drone (calibration)

In the UI:
1. Hover or fly the quad 5–30 m from the mic and press **Record drone (5 s)**. Do 3–5 clips at
   different throttle and distance. A clip with no audible prop tone is flagged and ignored where
   possible.
2. With the quad powered off, press **Record background (5 s)** once or twice. This is what stops
   wind and distant traffic matching your drone.
3. The "My drone" bar now moves, and it counts for half of the base score.

You can also enrol from files, for example audio pulled from GoPro footage:

```bash
ffmpeg -i GX010123.MP4 -vn -ac 1 -ar 16000 fpv.wav
sudo -u acoustic DATA_DIR=/var/lib/acoustic-detector \
  /opt/acoustic-detector/.venv/bin/python -m acoustic --enroll fpv.wav --enroll-background field.wav
```

Clips are stored in `/var/lib/acoustic-detector/samples/{drone,background}/`. **Clear** deletes them.

## Demo script

1. `sudo pi-profiles bft-acoustic`. Open the UI on a laptop or phone and ATAK on a tablet. Check
   that the chips read *Input MIC*, *ATAK connected* and *Mesh connected*, and that the Gemini chip
   reads *OFFLINE* (or *CLOUD* if you chose to enable it).
2. Talk next to the mic: the waterfall moves, and the confidence stays low. *"It's listening for a
   specific signature, not just loudness."*
3. **Calibrate** (2 minutes): hover the FPV about 10 m away and record 3 drone clips, then 1
   background clip with it disarmed.
4. Land the quad about 150 m out. Fly it slowly **toward the Pi**. On the waterfall, the prop comb
   appears as stacked horizontal lines that bend with throttle. Confidence climbs and the banner goes
   red with **DRONE DETECTED**. At the same moment the ATAK marker appears at the OP with its radius
   ring, and the mesh node shows `ACOUSTIC: drone detected conf 0.8x @ OP1`.
5. Fly away again: after about 3 s below threshold it clears. Fly back within 60 s and the log shows
   `suppressed` with no second mesh text. *"That's the cooldown, so the mesh isn't spammed."*
6. Optional: press **Ask Gemini** to show a cloud second opinion next to the on-device result.
   Discuss when you would or wouldn't want a sensor that depends on the cloud.
7. False-positive discussion: start a petrol generator, a strimmer, or a car engine near the mic,
   and talk about what triggers and why.

## Honest limitations

- **Range is short.** A 5" FPV is loud up close but drops into wind and ambient noise quickly.
  Roughly, with a basic USB mic in a quiet field: aggressive flight **~100–200 m**, gentle cruise or
  hover **~50–100 m**, and far less in wind, near roads or downwind of the drone. A windshield, a
  better mic, or a parabolic dish all help. These figures are expectations, not measurements from
  this build. Measure your own on the day.
- **No bearing, no range.** One mic gives presence only. That's why the ATAK marker sits on the
  sensor with a radius ring. Bearing needs a mic array (TDOA), which would be a good next step.
- **False positives.** Lawnmowers, strimmers, chainsaws, generators, motorbikes, model aircraft,
  wasps near the mic and some electric tools all make harmonic or buzzy sound. Low-frequency
  engines are mostly rejected by the comb check, and YAMNet's confuser classes pull the score down.
  But a two-stroke strimmer at high revs can sit right in the prop band. Calibration with background
  clips from the actual site is the best mitigation. Treat a detection as a cue to look, not as
  identification.
- **False negatives.** Wind noise on the mic, heavy rain, and loud nearby sound (vehicles,
  generators running at the OP) mask the drone. Gliding or low-throttle drones are much quieter.
- **YAMNet wasn't trained on drones.** The class mapping is a heuristic. It helps on real drone
  audio, and you can see in the UI what it thinks it hears.
- **Thresholds are untuned against real flights.** The defaults were chosen on synthetic audio. Run
  the demo once with the real quad and adjust `DETECT_ON`/`DETECT_OFF` if needed.
- **CPU:** the per-window analysis is a few ms on a laptop, and YAMNet adds about 10–30 ms per window
  on a Pi 5 (an estimate). Fine at two windows a second, but this hasn't been profiled on a Pi yet.

## Gemini (optional cloud mode)

Set `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`, default `gemini-3.8-flash`) in
`/etc/acoustic-detector.env`. With `GEMINI_AUTO=1`, each new detection sends the last 4 s of audio,
and **Ask Gemini** sends the last 4 s on demand (at most once every 30 s automatically). The verdict
is shown in its own panel and never changes the on-device confidence or alerts.

**Cloud mode uploads audio to Google.** Use it only with demo audio. Never use it on an exercise
or operation where the sound environment (voices, vehicles, location cues) is sensitive. If the
key isn't set, the UI shows **OFFLINE** and nothing leaves the Pi.

## Configuration

All settings live in `/etc/acoustic-detector.env`. See
[`deploy/acoustic-detector.env.example`](deploy/acoustic-detector.env.example). The most important
ones are `SENSOR_NAME`, `SENSOR_LAT`/`SENSOR_LON` (without them no ATAK markers are sent),
`DETECT_ON`/`DETECT_OFF`/`ALERT_COOLDOWN`, `TAK_HOST`/`TAK_PORT`, `MQTT_*` and `MESH_TO`.

## Layout

```
acoustic/            python package (python -m acoustic)
  dsp.py             resampling, waterfall, prop-comb features, spectral embedding
  yamnet.py          LiteRT YAMNet wrapper + drone/confuser class mapping
  calibration.py     enrolment and cosine-similarity scoring
  detector.py        fusion + hysteresis/cooldown
  engine.py          audio ring buffer, analysis loop, outputs, recording
  sources.py         USB mic (sounddevice), file replay, simulator source
  simulator.py       synthetic soundscape
  cot.py meshbus.py gemini.py   outputs
  web/               FastAPI app + static UI
deploy/              install.sh, start.sh, systemd unit, env example
profiles.d/          pi-profiles configs (acoustic-detector, bft-acoustic)
tests/               pytest
```

YAMNet model and class map © Google, Apache-2.0.
