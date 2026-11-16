#!/usr/bin/env bash
# Run acoustic-detector on a Mac/Linux laptop joined to the Pi hotspot, using the laptop's mic.
# Config comes from the ENVIRONMENT only (the platform's laptop/start-mac.sh sources
# ~/.config/isr-kit/station.env first). Runs in the foreground; SIGTERM/Ctrl-C exits cleanly
# and leaves a retained "stopped" heartbeat. UI on 0.0.0.0:${WEB_PORT:-8100}.
#
#   ACOUSTIC_INPUT  mic (default) | simulate | /path/to/clip.(wav|mp3|mp4|mov)
#   AUDIO_DEVICE    input index or name (see: scripts/run-laptop.sh --list-devices)
#   TAK_HOST/TAK_PORT, MQTT_HOST/MQTT_PORT/MQTT_USER/MQTT_PASS, SENSOR_NAME/SENSOR_LAT/SENSOR_LON
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export TAK_HOST="${TAK_HOST:-isr-pi.local}"
export TAK_PORT="${TAK_PORT:-8087}"
export MQTT_HOST="${MQTT_HOST:-isr-pi.local}"
export MQTT_PORT="${MQTT_PORT:-1883}"
export MQTT_USER="${MQTT_USER:-isrkit}"
export WEB_PORT="${WEB_PORT:-8100}"
export SENSOR_NAME="${SENSOR_NAME:-LAPTOP}"
export DATA_DIR="${DATA_DIR:-$HOME/.local/share/isr-kit/acoustic-detector}"
export PYTHONUNBUFFERED=1
mkdir -p "$DATA_DIR"
if [ -z "${MQTT_PASS:-}" ]; then
    echo "WARNING: MQTT_PASS not set; mesh alerts and heartbeat will fail to authenticate" >&2
fi

# First run: create the venv (Python 3.11+; uses uv if available).
PY="$REPO/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Creating virtualenv in $REPO/.venv (first run only)..."
    if command -v uv >/dev/null 2>&1; then
        uv venv -q -p 3.11 "$REPO/.venv"
        uv pip install -q -p "$PY" -r requirements.txt
        uv pip install -q -p "$PY" -r requirements-ml.txt || echo "WARNING: LiteRT not installed; spectral-only"
    else
        BASE=""
        for c in python3.13 python3.12 python3.11; do command -v "$c" >/dev/null 2>&1 && BASE="$c" && break; done
        [ -n "$BASE" ] || { echo "Need Python 3.11+ (or uv) to set up the venv" >&2; exit 1; }
        "$BASE" -m venv "$REPO/.venv"
        "$PY" -m pip install -q --upgrade pip
        "$PY" -m pip install -q -r requirements.txt
        "$PY" -m pip install -q -r requirements-ml.txt || echo "WARNING: LiteRT not installed; spectral-only"
    fi
fi
[ -s models/yamnet.tflite ] || bash scripts/fetch_model.sh || echo "WARNING: no YAMNet model (offline?); spectral-only"

if [ "${1:-}" = "--list-devices" ]; then
    exec "$PY" -m acoustic --list-devices
fi

ARGS=()
case "${ACOUSTIC_INPUT:-mic}" in
    mic) ;;
    simulate) ARGS+=(--simulate) ;;
    *) ARGS+=(--file "$ACOUSTIC_INPUT" --loop) ;;
esac

# macOS: the first run triggers a Microphone permission prompt for the terminal app. If the
# level meter stays flat, allow it under System Settings > Privacy & Security > Microphone.
exec "$PY" -m acoustic ${ARGS[@]+"${ARGS[@]}"} "$@"
