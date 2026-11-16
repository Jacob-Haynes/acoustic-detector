# Run acoustic-detector on a Windows laptop joined to the Pi hotspot, using the laptop's mic.
# Config comes from environment variables only (same names as scripts/run-laptop.sh).
# Foreground; Ctrl-C exits cleanly and leaves a retained "stopped" heartbeat. UI on 0.0.0.0:8100.
#   powershell -ExecutionPolicy Bypass -File scripts\run-laptop.ps1 [-ListDevices]
param([switch]$ListDevices)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

function Default($name, $value) {
    if (-not [Environment]::GetEnvironmentVariable($name)) { [Environment]::SetEnvironmentVariable($name, $value) }
}
Default "TAK_HOST" "isr-pi.local"
Default "TAK_PORT" "8087"
Default "MQTT_HOST" "isr-pi.local"
Default "MQTT_PORT" "1883"
Default "MQTT_USER" "isrkit"
Default "WEB_PORT" "8100"
Default "SENSOR_NAME" "LAPTOP"
Default "DATA_DIR" (Join-Path $env:LOCALAPPDATA "isr-kit\acoustic-detector")
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path $env:DATA_DIR | Out-Null
if (-not $env:MQTT_PASS) { Write-Warning "MQTT_PASS not set; mesh alerts and heartbeat will fail to authenticate" }

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Creating virtualenv (first run only)..."
    py -3.11 -m venv .venv
    & $Py -m pip install -q --upgrade pip
    & $Py -m pip install -q -r requirements.txt
    & $Py -m pip install -q -r requirements-ml.txt
    if ($LASTEXITCODE -ne 0) { Write-Warning "LiteRT not installed; spectral-only" }
}
$Model = Join-Path $Repo "models\yamnet.tflite"
if (-not (Test-Path $Model)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Repo "models") | Out-Null
    try {
        Invoke-WebRequest -UseBasicParsing -OutFile $Model `
            "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
    } catch { Write-Warning "No YAMNet model (offline?); spectral-only" }
}

if ($ListDevices) { & $Py -m acoustic --list-devices; exit $LASTEXITCODE }

$ArgsList = @("-m", "acoustic")
$Src = if ($env:ACOUSTIC_INPUT) { $env:ACOUSTIC_INPUT } else { "mic" }
if ($Src -eq "simulate") { $ArgsList += "--simulate" }
elseif ($Src -ne "mic") { $ArgsList += @("--file", $Src, "--loop") }
& $Py @ArgsList @args
exit $LASTEXITCODE
