#!/usr/bin/env bash
# Download the YAMNet LiteRT model (Apache-2.0, ~4 MB) into ./models for local runs and tests.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
URL="https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
if [ -s models/yamnet.tflite ]; then echo "models/yamnet.tflite already present"; exit 0; fi
curl -fsSL --retry 3 -o models/yamnet.tflite.part "$URL"
mv models/yamnet.tflite.part models/yamnet.tflite
echo "Saved models/yamnet.tflite"
