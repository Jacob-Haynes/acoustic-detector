#!/usr/bin/env bash
# Pick the input: ACOUSTIC_INPUT=auto (default) | mic | simulate | /path/to/clip.wav
set -euo pipefail
cd /opt/acoustic-detector
PY=/opt/acoustic-detector/.venv/bin/python
INPUT="${ACOUSTIC_INPUT:-auto}"
read -r -a EXTRA <<< "${ACOUSTIC_ARGS:-}"

case "$INPUT" in
    simulate)
        exec "$PY" -m acoustic --simulate "${EXTRA[@]}" ;;
    mic)
        exec "$PY" -m acoustic "${EXTRA[@]}" ;;
    auto)
        if arecord -l 2>/dev/null | grep -q '^card'; then
            echo "Capture device found:"; arecord -l | grep '^card'
            exec "$PY" -m acoustic "${EXTRA[@]}"
        fi
        echo "No capture device found: starting in simulation mode"
        exec "$PY" -m acoustic --simulate "${EXTRA[@]}" ;;
    *)
        if [ -f "$INPUT" ]; then
            exec "$PY" -m acoustic --file "$INPUT" --loop "${EXTRA[@]}"
        fi
        echo "ACOUSTIC_INPUT='$INPUT' is not auto/mic/simulate or an existing file" >&2
        exit 1 ;;
esac
