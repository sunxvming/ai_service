# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the service
python main.py

# Run a model smoke test (standalone scripts, not pytest)
python tests/test_phone.py
python tests/test_fire_smoke.py
python tests/test_ppe.py      # helmet / lifejacket / workwear
# Each test accepts --image <path> --show flags; output images go to tests/output/

# Download / check model files
python tools/download_models.py         # download all models to models/pt/
python tools/download_models.py --list  # show download status

# Install dependencies
pip install -r requirements.txt
# Note: sophon.sail (BM1688 SDK) is NOT on PyPI; install separately on BM1688 hardware
```

Interactive API docs (running service): `http://127.0.0.1:9001/docs`

## Architecture

### Process model

The service runs as multiple OS processes:

- **Main process** (`main.py`): HTTP API (9001), Qt broadcast TCP (9002), internal relay TCP (9003), `ProcessManager` (spawns/monitors workers).
- **Inference workers** (`core/worker/inference_worker.py`): one per enabled camera. Each owns its own RTSP decoder, inference models (for its assigned algorithms only), `ByteTracker`, `EventEngine`, and `AlarmManager`.

Workers connect to the relay on port 9003 and send newline-delimited JSON messages (same format as port 9002). The main process forwards them verbatim to Qt clients.

### Startup sequence (`main.py`)

1. Load `config/config.yaml`, init logger
2. Start `TCPServer` on port 9002 (Qt client broadcast)
3. Init SQLite DB (`data/cameras.db`)
4. Start `CameraManager` (loads camera list from DB, no decoders)
5. Start `InternalTCPRelay` on port 9003 (must be ready before workers connect)
6. Create `ProcessManager`; link it to `CameraManager`
7. Start uvicorn (FastAPI, port 9001)
8. `ProcessManager.start_all()` — spawns N worker processes (one per enabled `IpcInfo` row)

### Per-frame processing (inside each `InferenceWorker`)

```
VideoDecoder.get_latest_frame()
  → MultiModelInferenceManager.infer_for_algorithms(frame, camera's algo_ids)
      [ThreadPoolExecutor — only this camera's models]
  → ByteTracker.update(detections, frame)
  → send DetectionResult JSON → port 9003 → main → port 9002 → Qt client
  → EventEngine.process_detections(...)   # sliding window suppression
  → AlarmManager.create_alarm(...)        # cooldown, screenshot, DB persist
  → send AlarmEvent JSON → port 9003 → main → port 9002 → Qt client
```

Camera status is broadcast by each worker every 5 s via port 9003.

### Dynamic camera management (API-driven)

`CameraManager` CRUD methods propagate to `ProcessManager`:
- `add_camera()` → `spawn_worker(ipc_id)` if enabled
- `update_camera()` → `restart_worker(ipc_id)` if enabled (any config change); `stop_worker(ipc_id)` if disabled
- `remove_camera()` → `stop_worker(ipc_id)` + DB delete

Worker processes that die unexpectedly are restarted by `ProcessManager._monitor_loop()` every 10 s.

### Key new files

| File | Role |
|---|---|
| `core/worker/inference_worker.py` | Child-process class + `run_inference_worker` entry point |
| `core/process_manager.py` | Spawn/monitor/restart worker processes |
| `core/tcp/internal_relay.py` | `InternalTCPRelay` — port 9003 listener, forwards to `TCPServer` |

### Configuration

`config/config.yaml` controls runtime behavior. Key sections:

- `tcp.port` / `tcp.internal_port` (9003) / `tcp.internal_connect_host` (127.0.0.1)
- `decoder.backend`: `opencv` (one `VideoDecoder` per worker process) or `sophon` (one `SophonVideoDecoder` per worker process, requires BM1688 hardware with SAIL SDK installed)
- `inference.backend`: `pytorch` or `sophon`; `inference.device`: `cpu` or `cuda`
- `inference.imgsz`: model input size (default `320`)
- `inference.inference_timeout`: per-model timeout in ms (default `80.0`)
- `api.port` / `api.workers` (must be 1 — more workers would fork the ProcessManager)

Algorithm-level config (confidence, label map, cooldown, window/threshold) lives in the `algorithms` SQLite table. Changing it requires restarting the affected worker(s) via the update-camera API (or a full service restart).

### Adding a new detection model

1. Add a `.pt` file to `models/pt/`
2. Insert a row into the `algorithms` table with `model_path`, `confidence_threshold`, `label_map` (JSON), `alarm_type`, `alarm_window`, `alarm_threshold`, `alarm_cooldown`
3. Assign the algorithm to cameras via `PUT /api/v1/ipc/{id}` (`algorithm_ids` field, comma-separated IDs)
4. The camera's worker process restarts automatically and loads the new model

### Internal message protocol (port 9003)

Workers reuse `DetectionResult.to_dict()`, `AlarmEvent.to_dict()`, and `CameraStatus.to_dict()` — identical to what the Qt client receives on port 9002. The relay just forwards lines without parsing logic.

### Database

SQLite at `data/cameras.db` with four tables: `IpcInfo`, `algorithms`, `alarm_events`, `alarm_scene`. Schema auto-migrates on startup. Each process (main + every worker) holds its own SQLAlchemy connection pool. Camera RTSP credentials are Fernet-encrypted using `CAMERA_ENCRYPTION_KEY` from `.env`.
