# acoustic-detector

A Raspberry Pi 5 microphone sensor that hears a nearby drone and reports it: an ATAK marker
(CoT over TCP to OpenTAKServer) and a Meshtastic text via a mesh bridge. Built as
a drone-sound detection demo. It is a sensor only: it never controls, steers or
engages anything.

Everything runs on the Pi CPU. No Hailo, no internet needed. Google Gemini is an optional
second opinion that you can switch on.

## How it works

Every 0.5 s the last ~1 s of audio (16 kHz mono) goes through three independent checks:

| Component | What it is | Needs |
|---|---|---|
| **Prop comb** | Hand-built spectral check. A multirotor makes a harmonic comb whose fundamental (rotation rate x blades) sits around 150–700 Hz for a 5" FPV quad. It's matched per 128 ms frame, so throttle changes don't smear it. Petrol engines also make combs, but at 20–60 Hz, so they score low. Voices are combs too, but they stop and start between syllables and their pitch wanders, so the score is damped unless the tone is **continuous** and **steady**. | nothing |
| **YAMNet** | Google's pretrained AudioSet classifier (521 classes), run with LiteRT. AudioSet has **no "drone" class**, so the drone score counts `Aircraft`, `Aircraft engine`, `Propeller, airscrew`, `Helicopter`, `Fixed-wing aircraft, airplane`, `Light engine (high frequency)`, `Buzz`, `Insect`, `Bee, wasp, etc.`, `Mosquito`, `Fly, housefly` and `Whir` (insect classes weighted lower). A separate **confuser** score counts `Lawn mower`, `Chainsaw`, `Power tool`, `Motorcycle`, vehicle and engine classes, `Hair dryer`, `Vacuum cleaner` and `Speech`. | `models/yamnet.tflite` + `ai-edge-litert` |
| **My drone** (calibration) | Cosine similarity to reference embeddings built from clips of *your* drone (64-band spectral shape, plus the YAMNet score vector when the model is loaded). If you also enrol background clips, the score becomes "closer to my drone than to this field". | 3–5 recorded clips |

Fusion: `confidence = base + 0.45·yamnet·(1−base) − 0.5·confuser`, where `base` is the prop-comb
score, or the average of prop comb and calibration once a drone is enrolled. YAMNet adds evidence
but can't trigger an alert on its own, and a confident "lawn mower" call pulls the score down.

Alerting: the confidence is smoothed, then a **hysteresis** state machine raises a detection after
3 windows (1.5 s) ≥ 0.65 and clears after 6 windows < 0.40. Another detection within the **60 s cooldown**
is logged as `suppressed` and sends no mesh text. All of these can be tuned in the env file.

On a detection:
- **ATAK**: a CoT `a-u-A-M-F-Q` (unknown UAV; set `COT_AFFILIATION=h` for hostile) placed **at the
  sensor's position**, with `ce` set to the detection radius and a remarks field stating that it
  shows presence near the sensor, not the target's location. While the detection stays active the
  marker is re-sent every 15 s. A friendly sensor-point marker (`a-f-G-E-S`, "ACOUSTIC MIC1") is sent
  every 60 s.
- **Mesh**: `{"text": "ACOUSTIC: drone detected conf 0.87 @ MIC1", "to": null, "channel": 0,
  "source": "acoustic-detector"}` published to `meshbus/tx` on 127.0.0.1:1883. The mesh bridge owns
  the radio, and this tool never opens the serial port.
- **Labelling**: in `--simulate` the CoT remarks (detection and sensor marker) start with
  `SIMULATED: ` and mesh texts start with `[SIM] `; in `--file` replay they start with `RECORDED: `
  and `[REC] `. Live microphone input has no prefix. That includes the office demo, where drone
  audio played through a speaker is a real acoustic detection.
- **Rate limits**: at most one mesh text per 30 s, and at most one CoT event per 5 s per uid.
  Anything over is dropped and logged.
- **Gemini** (if enabled): the last 4 s are sent for a second opinion, which is shown in its own panel.
- Each event is appended to `/var/lib/acoustic-detector/detections.jsonl`.

Every 10 s a retained heartbeat goes to `isrkit/heartbeat/acoustic-detector`
(`{"tool","host","url","mode","status","ts"}`, where mode is `cpu`, `gemini`, `simulate` or
`replay`, and status is `ok`, `degraded` or `error`). `host` is `ISR_HOST` if set, otherwise
the lowercased short hostname. A clean exit leaves `stopped`. A crash
leaves `error` through the MQTT last-will. The Pi launcher uses this to list the tool wherever
it runs.

**TAK connection.** OpenTAKServer drops CoT from senders that aren't a registered EUD, so the
sensor keeps one long-lived TCP connection to `TAK_HOST:TAK_PORT`
([`acoustic/tak_client.py`](acoustic/tak_client.py), standard library only). On every
(re)connect it waits 3 s, then registers with a stable uid `isrkit-acoustic-detector-<ISR_HOST>`
and callsign `ACOUSTIC <ISR_HOST>` (an `a-f-G-E-S` SA with a contact endpoint, the Cyan group and
`takv`), re-sent every 60 s. All markers then go out on that connection under their own uids. With
no sensor position the registration uses the never-drawn unknown point (0,0, errors 9999999,
stale = start). Because the callsign must be unique on the server, give each station its own
`ISR_HOST`.

If OpenTAKServer or the MQTT broker is down, the sensor logs it, keeps detecting, and reconnects
with backoff.

## Hardware

- Raspberry Pi 5 (64-bit Pi OS Bookworm), SSD boot.
- **A USB microphone.** The Pi 5 has no built-in mic. The recommended mic is the **Blue Yeti
  Nano** (see below). Any class-compliant USB mic or USB sound card also works. A foam or fur
  windshield makes a big difference outdoors.
- Optional: the BFT kit (OpenTAKServer + Heltec V3 Meshtastic node via `mesh-bridge`) for ATAK
  markers and mesh texts.
- Test target: a 5" FPV quad.

### Recommended mic: Blue Yeti Nano

This is the demo mic. It's a class-compliant USB condenser (48 kHz / 24-bit, 20 Hz–20 kHz) and
needs no drivers on Pi OS or macOS.

- **Auto-selected.** With no `AUDIO_DEVICE` set, the sensor picks any input whose name contains
  "Yeti Nano" ahead of other inputs: the Mac's built-in mic and webcams, or other USB audio. The
  preference list is `AUDIO_PREFER` (comma-separated name fragments, default `Yeti Nano`).
- **48 kHz is handled.** On the Pi, ALSA `hw:` devices don't convert formats, so the Nano opens at
  its native 48 kHz (stereo if mono is refused). It's downmixed and resampled to 16 kHz for YAMNet
  with a continuous anti-alias filter. The UI's Microphone line shows what was negotiated, e.g.
  `Yeti Nano: USB Audio (hw:2,0) @ 48000 Hz, 2ch downmixed -> resampled to 16000`. On macOS,
  CoreAudio delivers 16 kHz directly. `AUDIO_RATE=48000` forces the native rate if you want it.
- **Pattern.** It has two capsules and two patterns: **cardioid** (picks up from the front) and
  **omni**. For the office demo use **cardioid**, with the front of the mic facing the speaker
  playing drone audio. This rejects some room noise from behind. Outdoors, where the drone
  could come from anywhere, **omni** is the better choice. The pattern is switched in Logitech G
  HUB / Blue Sherpa on a Mac or PC. Whether it stays set once the mic is moved to the Pi isn't
  verified here, so check the setting before the demo.
- **Gain.** Reviews disagree on whether the Nano has a hardware gain knob (the front dial is
  headphone volume and mute). Set the level in software instead: on the Pi with `alsamixer -c <card>`
  → F4 → *Mic* capture at about mid (50–70%), on a Mac in G HUB / Sherpa. Aim for the UI's
  **Input peak** at ~0.05–0.3 with normal speech at 1 m and never red. Make sure the **mute** light
  isn't on: tap the front dial to toggle mute.
- **Power.** Logitech's spec sheet gives **5 V / 150 mA** from USB bus power. That's well within
  the Pi 5's 600 mA USB limit, which applies when the Pi isn't on a 5 A PSU, so no powered hub is
  needed for the mic alone.
- **Tested here:** I ran 30 s of a real office on a Mac through `scripts/run-laptop.sh` with the
  Nano auto-selected. The noise floor (10th percentile level) was **−45 dBFS** with a median of
  −43 dBFS, and the loudest window was −27 dBFS. YAMNet heard *Inside, small room*, *Typing*,
  *Speech* and *Music*. Confidence peaked at **0.02** (the threshold is 0.65), and no events were
  logged. The forced 48 kHz resample path also ran live on the Nano without overflows.

## Install (Pi)

**The Pi has no internet once it is the hotspot.** Fetch the model assets while you are still
online, on the Pi or on the laptop you copy the repo from:

```bash
bash scripts/prefetch.sh     # YAMNet model -> models/yamnet.tflite; checks the class map
```

`acoustic/data/yamnet_class_map.csv` is committed (Apache 2.0, from Google's YAMNet). The model
file (~4 MB) isn't, so copy the repo's `models/` directory along with the code. Then on the Pi:

```bash
git clone <this repo> && cd acoustic-detector    # or the copy you brought over
sudo bash deploy/install.sh                      # add --skip-apt when offline
sudo nano /etc/acoustic-detector.env     # set SENSOR_NAME, SENSOR_LAT, SENSOR_LON
```

If YAMNet can't run (no LiteRT wheel, no model, no class map), the sensor **still starts** and runs
prop-comb detection only. It logs one warning, the UI's model chip says *YAMNet unavailable
(offline install?)*, and the heartbeat is `degraded` with that text in `detail`.

The install script is idempotent and non-interactive. It installs apt packages, creates an
`acoustic` service user in the `audio` group, puts the code in `/opt/acoustic-detector` with a venv,
downloads the YAMNet model (~4 MB), creates `/var/lib/acoustic-detector`, copies the env example to
`/etc/acoustic-detector.env` if one doesn't exist, and installs the systemd unit **without enabling
it**. pi-profiles starts and stops it:

```bash
sudo pi-profiles acoustic-detector   # mic sensor on its own
journalctl -u acoustic-detector -f
```

ATAK markers need OpenTAKServer (`opentakserver`), and mesh alerts need the mesh bridge
(`mesh-bridge`) plus RabbitMQ. Combined scenario profiles that run this sensor alongside them are
defined in the pi-platform (pi-profiles) repo, not here.

Web UI: `http://<pi-ip>:8100`.

`ACOUSTIC_INPUT` in the env file selects the input: `auto` (USB mic if `arecord -l` finds one,
otherwise simulate), `mic`, `simulate`, or the path to a clip (audio or MP4/MOV) to replay in a
loop.

## USB microphone setup (Pi OS)

Any class-compliant USB mic or USB sound card works. No drivers are needed.

1. Plug it in and check ALSA sees it:
   ```bash
   arecord -l            # e.g. "card 2: Device [USB PnP Sound Device], device 0: USB Audio"
   ```
2. Record 5 s and play it back (or copy it off and listen):
   ```bash
   arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/mic.wav
   aplay /tmp/mic.wav    # needs an output device; otherwise scp it to a laptop
   ```
3. Set the capture gain: `alsamixer -c 2`, press **F4** (Capture), and raise the mic level to
   about 70–80%. If the device has **Auto Gain Control**, turn it **off** (select it and press `M`).
   AGC pumps the background up whenever it's quiet, which inflates the noise floor.
4. Pick the device. By default a Yeti Nano is used if present, otherwise the first USB capture
   device. To change it, use the
   **Microphone** panel in the web UI (choose it, then press **Use**; the choice is saved in
   `/var/lib/acoustic-detector/audio_device.txt` and survives restarts), or set
   `AUDIO_DEVICE=<index or part of the name>` in the env file. To list the devices:
   `sudo -u acoustic /opt/acoustic-detector/.venv/bin/python -m acoustic --list-devices`.
5. Check the **Input peak** meter: speech at 1 m should reach about 0.05–0.3. Red means clipping:
   turn the gain down. A flat meter means the wrong device or the gain is at zero.

If the mic is unplugged, the sensor logs the error, shows it in the UI, reports `error` in its
heartbeat, and retries every 3 s. It finds the mic again after a replug, even if the card number
has changed.

## Running on a laptop

The Pi at the sensor position is the normal setup. A laptop is the fallback for when the USB mic misbehaves on
the day. It uses the laptop's built-in mic and still reports to the Pi hub over the hotspot.

1. Join the Pi's hotspot (the hub is `isr-pi.local`, 192.168.4.1).
2. Set the environment (the platform's `laptop/start-mac.sh` does this from
   `~/.config/isr-kit/station.env`):
   `TAK_HOST=isr-pi.local MQTT_HOST=isr-pi.local MQTT_USER=isrkit MQTT_PASS=...`
   plus `SENSOR_NAME`, `SENSOR_LAT` and `SENSOR_LON`.
3. Run it:
   ```bash
   scripts/run-laptop.sh                   # macOS / Linux
   ```
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run-laptop.ps1   # Windows
   ```
   The first run creates `.venv` (Python 3.11+, or uses `uv`) and downloads YAMNet. The script runs
   in the foreground and Ctrl-C/SIGTERM exits cleanly. The UI is at `http://<laptop>.local:8100`.
   It reads config only from the environment, with defaults `isr-pi.local`, user `isrkit`, port
   8100, and data in `~/.local/share/isr-kit/acoustic-detector`. Set `ACOUSTIC_INPUT=simulate` or
   `ACOUSTIC_INPUT=/path/clip.mp4` to run without the mic.
4. On macOS, the first run asks for **Microphone** permission for the terminal app. If the input
   peak stays flat, allow it in System Settings → Privacy & Security → Microphone and restart.

Laptop mics apply noise suppression and AGC, and they are usually close to a fan. Expect less
range than with the USB mic, and enrol a fresh background clip on that laptop.

## Development run (no hardware)

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements-dev.txt
scripts/prefetch.sh                       # YAMNet model + class map (run while online)
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

You can also **enrol from existing footage**. In the UI, under *From file*, pick an MP4/MOV
(GoPro, phone, DVR) or a WAV/MP3, choose *as drone* or *as background*, and press **Enrol file**.
The audio track is extracted on the device (PyAV, no ffmpeg needed) and the first 120 s are used.
The result shows what percentage of the file contained a prop tone. From the command line:

```bash
sudo -u acoustic DATA_DIR=/var/lib/acoustic-detector \
  /opt/acoustic-detector/.venv/bin/python -m acoustic --enroll GX010123.MP4 --enroll-background office.wav
```

**Which footage has usable audio?**
- **DJI O3 / O4 air-unit recordings**: these may have **no audio track**. One forum reply says the
  O4 doesn't record audio, and nothing official confirms either way. Just try one: the UI says
  "has no audio track" if so.
- **GoPro on the quad**: has audio, but the mic is centimetres from the props and full of wind.
  It's still the right harmonic signature, so it's worth enrolling.
- **A phone or GoPro filming the quad from the ground**: the best source, because it's what the
  sensor actually hears.

Clips are stored in `/var/lib/acoustic-detector/samples/{drone,background}/`. **Clear** deletes them.

## Office demo (no flying)

The drone doesn't need to fly: play its sound through a speaker toward the mic.

**Audio to play.** In order of preference:
1. Your own footage of the 5" quad with an audio track (phone or GoPro filming from the ground).
   Play the MP4 from the laptop, or extract the audio first.
2. A public-domain clip: [*Quadrocopter Take-Off And Landing*](https://freesound.org/s/263022/) by
   Alcappuccino on Freesound, licensed **CC0** (no attribution needed). It's a DJI Phantom 2
   recorded from a GoPro on the drone, 59 s, 48 kHz WAV, and downloading it needs a free Freesound
   account. A Phantom has bigger, slower props than a 5" FPV, so its tone sits lower, but it's well
   inside the detector's range.

**Setup.** Run the Pi with its USB mic (or the laptop fallback). Open the UI on the projector. Use a
laptop or Bluetooth speaker about 1–2 m from the mic, starting at **low volume**. Then:

1. **Baseline (1 min).** Leave the room noise running: talking, typing, air-con. The waterfall
   moves, while the confidence stays low and the log stays empty. Point out the *YAMNet top* line
   saying "Speech" when people talk. Press **Record background (5 s)** once while people chat
   normally. This teaches the sensor what "this room" sounds like.
2. **Enrol (1 min).** Under *From file*, enrol your drone footage *as drone* (or play it and press
   **Record drone (5 s)** two or three times). The "My drone" bar becomes active.
3. **Approach.** Play the drone clip and slowly raise the volume. The stacked horizontal lines of
   the prop comb appear on the waterfall. The *Prop comb* and *My drone* bars rise, and when the
   confidence crosses the red tick for 1.5 s the banner goes **DRONE DETECTED**. The ATAK marker and
   the mesh text `ACOUSTIC: drone detected conf … @ MIC1` follow immediately.
4. **Recede.** Lower the volume and the confidence falls. About 3 s below the amber tick, the log
   shows `clear`.
5. **Cooldown.** Raise the volume again within 60 s: the log shows `suppressed` and no second mesh
   text is sent.
6. **Try to fool it.** Talk loudly, whistle, hum, or play music near the mic. Discuss what moves the
   meter and what doesn't. A sustained whistle or hum is the honest weak spot: it's a steady tone,
   exactly the feature being looked for.

Tips: a speaker playing at moderate volume 1–2 m away reads like a real quad tens of metres away.
Too loud a speaker clips the mic (the peak meter turns red). Small Bluetooth speakers can't
reproduce much below ~150 Hz, which is fine for prop tones.

## Tuning knobs

| Env var | Default | Effect |
|---|---|---|
| `DETECT_ON` | 0.65 | Confidence needed to alert. Raise it to 0.75–0.8 if a noisy room false-triggers, lower it to 0.55 for more range. |
| `DETECT_OFF` | 0.40 | Confidence below which the detection starts to clear. |
| `DETECT_ON_FRAMES` | 3 | Consecutive 0.5 s windows above `DETECT_ON` before alerting. Raise it to 4–6 to ignore short bursts (a shout, a door). |
| `DETECT_OFF_FRAMES` | 6 | Consecutive windows below `DETECT_OFF` before clearing. |
| `ALERT_COOLDOWN` | 60 | Seconds before a new detection sends another mesh alert. |
| `USE_YAMNET` | 1 | Keep it on indoors. YAMNet's "Speech" penalty is what keeps voices well clear of the threshold. |
| Background clips | none | The single best false-positive fix: enrol 1–3 clips of the room or site with no drone. |

**Office noise check (synthetic):** I checked the defaults against 4 synthesised voices (macOS
`say`, about 40 s each), synthetic keyboard typing and synthetic air-con (broadband plus mains hum
plus fan blade tones at 128 Hz and 300 Hz). Nothing triggered. With YAMNet on, speech peaked at 0.29
confidence and keyboard/air-con at ≤ 0.04. With YAMNet **off**, high-pitched voices peaked at
0.62–0.71, which is uncomfortably close to the threshold. So keep YAMNet on indoors, or raise
`DETECT_ON` and enrol a background clip. I also ran 20 s of a real room through a Yeti Nano USB mic on a
Mac (YAMNet heard speech, music and a passing vehicle): confidence peaked at **0.04** and nothing
was logged. That's one room and one sample, so watch the meter for a minute in the actual room
before the demo.

## Demo script (field, flying)

1. Start a pi-platform scenario profile that includes `acoustic-detector`, `opentakserver` and
   `mesh-bridge` (see the pi-profiles repo). Open the UI on a laptop or phone and ATAK on a tablet. Check
   that the chips read *Input MIC*, *ATAK connected* and *Mesh connected*, and that the Gemini chip
   reads *OFFLINE* (or *CLOUD* if you chose to enable it).
2. Talk next to the mic: the waterfall moves, and the confidence stays low. *"It's listening for a
   specific signature, not just loudness."*
3. **Calibrate** (2 minutes): hover the FPV about 10 m away and record 3 drone clips, then 1
   background clip with it disarmed.
4. Land the quad about 150 m out. Fly it slowly **toward the Pi**. On the waterfall, the prop comb
   appears as stacked horizontal lines that bend with throttle. Confidence climbs and the banner goes
   red with **DRONE DETECTED**. At the same moment the ATAK marker appears at the sensor with its radius
   ring, and the mesh node shows `ACOUSTIC: drone detected conf 0.8x @ MIC1`.
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
- **Steady tones.** Whistling, humming, a singing kettle, some electric motors and a tonal fan can
  all look like a prop tone. Pitch stability and continuity are exactly what the detector rewards.
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

**Cloud mode uploads audio to Google.** Use it only with demo audio. Never use it with recordings
where the sound environment (voices, vehicles, location cues) is sensitive. If the
key isn't set, the UI shows **OFFLINE** and nothing leaves the Pi.

## Configuration

All settings live in `/etc/acoustic-detector.env`. See
[`deploy/acoustic-detector.env.example`](deploy/acoustic-detector.env.example). The most important
ones are `SENSOR_NAME`, `SENSOR_LAT`/`SENSOR_LON` (the shared OP position; without them, or
at 0,0, no ATAK markers are sent; `--simulate` alone falls back to a demo position), `ISR_HOST`,
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
  cot.py meshbus.py gemini.py   outputs (tak_client.py: registered TAK connection)
  web/               FastAPI app + static UI
deploy/              install.sh, start.sh, systemd unit, env example
profiles.d/          pi-profiles config for this tool only (combined profiles live in pi-profiles)
tests/               pytest
```

## License

MIT, see [LICENSE](LICENSE). The YAMNet model (downloaded at install, not committed) and
`acoustic/data/yamnet_class_map.csv` come from Google under the Apache 2.0 licence. No audio
recordings are committed: tests and `--simulate` use synthetic signals only.
