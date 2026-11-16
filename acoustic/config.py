"""Runtime configuration, read from environment (/etc/acoustic-detector.env) and CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16000          # everything internal runs at 16 kHz mono (YAMNet's rate)
WINDOW_SAMPLES = 15600       # 0.975 s analysis window (YAMNet's fixed input size)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Demo coordinates used in --simulate when no sensor position is configured.
DEMO_LAT = 52.0000
DEMO_LON = -1.0000


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val.strip()


def _env_float(name: str, default: float | None) -> float | None:
    val = _env(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    val = _env(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = _env(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _default_model_path() -> Path:
    for candidate in (
        Path("/opt/acoustic-detector/models/yamnet.tflite"),
        REPO_ROOT / "models" / "yamnet.tflite",
    ):
        if candidate.exists():
            return candidate
    return REPO_ROOT / "models" / "yamnet.tflite"


def _default_data_dir() -> Path:
    if Path("/var/lib/acoustic-detector").is_dir() and os.access("/var/lib/acoustic-detector", os.W_OK):
        return Path("/var/lib/acoustic-detector")
    return REPO_ROOT / "data"


@dataclass
class Settings:
    sensor_name: str = "MIC1"
    sensor_lat: float | None = None
    sensor_lon: float | None = None
    sensor_hae: float = 0.0
    # Rough acoustic detection radius shown on the ATAK marker (metres).
    detect_radius_m: float = 300.0
    cot_affiliation: str = "u"          # u = unknown, h = hostile
    cot_sensor_marker: bool = True

    tak_host: str = "127.0.0.1"
    tak_port: int = 8087
    tak_enabled: bool = True

    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_user: str = "guest"
    mqtt_pass: str = "guest"
    mesh_enabled: bool = True
    mesh_to: str | None = None
    mesh_channel: int = 0
    mesh_clear_msg: bool = False

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_auto: bool = True            # ask Gemini automatically on each new detection
    gemini_min_interval_s: float = 30.0

    audio_device: str | None = None
    audio_prefer: tuple = ("Yeti Nano",)  # auto-select these mics first (name substring)
    audio_rate: int | None = None         # force a capture rate (default: 16 kHz if allowed, else native)
    web_port: int = 8100
    public_url: str | None = None       # URL advertised in the heartbeat (default: auto)
    data_dir: Path = field(default_factory=_default_data_dir)
    yamnet_model: Path = field(default_factory=_default_model_path)
    use_yamnet: bool = True

    detect_on: float = 0.65
    detect_off: float = 0.40
    on_frames: int = 3                  # consecutive 0.5 s windows above detect_on to trigger
    off_frames: int = 6                 # consecutive windows below detect_off to clear
    alert_cooldown_s: float = 60.0
    min_level_dbfs: float = -65.0       # quieter than this is treated as silence

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.sensor_name = _env("SENSOR_NAME", s.sensor_name)
        s.sensor_lat = _env_float("SENSOR_LAT", None)
        s.sensor_lon = _env_float("SENSOR_LON", None)
        s.sensor_hae = _env_float("SENSOR_HAE", 0.0)
        s.detect_radius_m = _env_float("DETECT_RADIUS_M", s.detect_radius_m)
        aff = (_env("COT_AFFILIATION", s.cot_affiliation) or "u").lower()
        s.cot_affiliation = aff if aff in ("u", "h", "s") else "u"
        s.cot_sensor_marker = _env_bool("COT_SENSOR_MARKER", s.cot_sensor_marker)
        s.tak_host = _env("TAK_HOST", s.tak_host)
        s.tak_port = _env_int("TAK_PORT", s.tak_port)
        s.mqtt_host = _env("MQTT_HOST", s.mqtt_host)
        s.mqtt_port = _env_int("MQTT_PORT", s.mqtt_port)
        s.mqtt_user = _env("MQTT_USER", s.mqtt_user)
        s.mqtt_pass = _env("MQTT_PASS", s.mqtt_pass)
        s.mesh_to = _env("MESH_TO", None)
        s.mesh_channel = _env_int("MESH_CHANNEL", s.mesh_channel)
        s.mesh_clear_msg = _env_bool("MESH_CLEAR_MSG", s.mesh_clear_msg)
        s.gemini_api_key = _env("GEMINI_API_KEY", None)
        s.gemini_model = _env("GEMINI_MODEL", s.gemini_model)
        s.gemini_auto = _env_bool("GEMINI_AUTO", s.gemini_auto)
        s.audio_device = _env("AUDIO_DEVICE", None)
        prefer = _env("AUDIO_PREFER")
        if prefer is not None:
            s.audio_prefer = tuple(p.strip() for p in prefer.split(",") if p.strip())
        rate = _env_int("AUDIO_RATE", 0)
        s.audio_rate = rate or None
        s.web_port = _env_int("WEB_PORT", s.web_port)
        s.public_url = _env("PUBLIC_URL", None)
        if _env("DATA_DIR"):
            s.data_dir = Path(_env("DATA_DIR"))
        if _env("YAMNET_MODEL"):
            s.yamnet_model = Path(_env("YAMNET_MODEL"))
        s.use_yamnet = _env_bool("USE_YAMNET", s.use_yamnet)
        s.on_frames = max(1, _env_int("DETECT_ON_FRAMES", s.on_frames))
        s.off_frames = max(1, _env_int("DETECT_OFF_FRAMES", s.off_frames))
        s.detect_on = _env_float("DETECT_ON", s.detect_on)
        s.detect_off = _env_float("DETECT_OFF", s.detect_off)
        s.alert_cooldown_s = _env_float("ALERT_COOLDOWN", s.alert_cooldown_s)
        return s

    @property
    def has_position(self) -> bool:
        from acoustic.cot import valid_position

        return valid_position(self.sensor_lat, self.sensor_lon)     # unset or 0,0: no markers
