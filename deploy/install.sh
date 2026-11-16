#!/usr/bin/env bash
# Install acoustic-detector on the Pi. Idempotent, non-interactive. Does NOT enable at boot:
# pi-profiles starts/stops it.
# Usage: sudo bash deploy/install.sh
set -euo pipefail

APP=acoustic-detector
APP_DIR=/opt/$APP
DATA_DIR=/var/lib/$APP
ENV_FILE=/etc/$APP.env
SVC_USER=acoustic
MODEL_URL="https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo bash deploy/install.sh" >&2
    exit 1
fi

echo "=== $APP: Pi install ==="

echo "[1/7] System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip libportaudio2 libsndfile1 alsa-utils curl rsync

echo "[2/7] Service user '$SVC_USER' (in group audio)"
if ! id -u "$SVC_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin --groups audio "$SVC_USER"
else
    usermod -aG audio "$SVC_USER"
fi

echo "[3/7] Copying code to $APP_DIR"
mkdir -p "$APP_DIR/models"
rsync -a --delete \
    --exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'models' \
    --exclude '__pycache__' --exclude '.pytest_cache' \
    "$SRC_DIR/" "$APP_DIR/"
chmod +x "$APP_DIR/deploy/start.sh" "$APP_DIR/scripts/"*.sh

echo "[4/7] Python venv"
[ -x "$APP_DIR/.venv/bin/python" ] || python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q
if ! "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements-ml.txt" -q; then
    echo "WARNING: LiteRT runtime failed to install; YAMNet disabled, spectral detection still works."
fi

echo "[5/7] YAMNet model"
if [ ! -s "$APP_DIR/models/yamnet.tflite" ]; then
    if [ -s "$SRC_DIR/models/yamnet.tflite" ]; then
        cp "$SRC_DIR/models/yamnet.tflite" "$APP_DIR/models/yamnet.tflite"
    elif curl -fsSL --retry 3 -o "$APP_DIR/models/yamnet.tflite.part" "$MODEL_URL"; then
        mv "$APP_DIR/models/yamnet.tflite.part" "$APP_DIR/models/yamnet.tflite"
    else
        rm -f "$APP_DIR/models/yamnet.tflite.part"
        echo "WARNING: could not download YAMNet (offline?). Re-run install with internet, or copy"
        echo "         yamnet.tflite to $APP_DIR/models/. Spectral detection works without it."
    fi
fi

echo "[6/7] Data dir and config"
mkdir -p "$DATA_DIR/samples/drone" "$DATA_DIR/samples/background"
chown -R "$SVC_USER:$SVC_USER" "$DATA_DIR"
if [ ! -f "$ENV_FILE" ]; then
    install -m 600 -o root -g root "$APP_DIR/deploy/$APP.env.example" "$ENV_FILE"
    echo "Created $ENV_FILE: set SENSOR_NAME / SENSOR_LAT / SENSOR_LON before the demo."
else
    echo "Keeping existing $ENV_FILE"
fi

echo "[7/7] systemd unit (not enabled; pi-profiles starts it)"
install -m 644 "$APP_DIR/deploy/$APP.service" "/etc/systemd/system/$APP.service"
systemctl daemon-reload
if [ -d /opt/pi-profiles/profiles.d ]; then
    cp "$APP_DIR/profiles.d/"*.conf /opt/pi-profiles/profiles.d/
    echo "Installed pi-profiles configs from $APP_DIR/profiles.d"
fi

echo ""
echo "=== Install complete ==="
echo "  sudo pi-profiles acoustic-detector   # standalone sensor"
echo "  (combined scenario profiles with OpenTAKServer + mesh bridge come from pi-profiles)"
echo "  journalctl -u $APP -f                # logs"
echo "  Web UI: http://<pi-ip>:8100"
