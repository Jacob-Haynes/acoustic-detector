#!/usr/bin/env bash
# Download every model asset while you still have internet, BEFORE joining the Pi's hotspot.
# Then copy the repo to the Pi (the cable, a USB stick, or a local git mirror plus this models/ dir)
# and run deploy/install.sh there. install.sh copies models/yamnet.tflite from the repo.
#
#   YAMNet model      models/yamnet.tflite                  (~4 MB, Apache-2.0, Google MediaPipe)
#   YAMNet class map  acoustic/data/yamnet_class_map.csv    (committed; re-fetched only if missing)
#
# Without these, the sensor still runs, with prop-comb (spectral) detection only.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_URL="https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
MAP_URL="https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

fetch() {   # fetch URL DEST: atomic, never leaves a partial file behind
    local url=$1 dest=$2
    mkdir -p "$(dirname "$dest")"
    if curl -fsSL --retry 3 -o "$dest.part" "$url"; then
        mv "$dest.part" "$dest"
        echo "  fetched $dest"
    else
        rm -f "$dest.part"
        echo "  FAILED to fetch $url (offline?)" >&2
        return 1
    fi
}

ok=1
echo "YAMNet model"
if [ -s models/yamnet.tflite ]; then echo "  models/yamnet.tflite present"; else fetch "$MODEL_URL" models/yamnet.tflite || ok=0; fi
echo "YAMNet class map"
if [ -s acoustic/data/yamnet_class_map.csv ]; then
    echo "  acoustic/data/yamnet_class_map.csv present"
else
    fetch "$MAP_URL" acoustic/data/yamnet_class_map.csv || ok=0
fi
if [ "$(wc -l < acoustic/data/yamnet_class_map.csv 2>/dev/null || echo 0)" -lt 522 ]; then
    echo "  class map looks incomplete (expected 521 classes + header)" >&2; ok=0
fi

if [ "$ok" -eq 1 ]; then
    echo "All assets present. You can go offline now."
else
    echo "Some assets are missing. The sensor will run prop-comb-only until they are fetched." >&2
    exit 1
fi
