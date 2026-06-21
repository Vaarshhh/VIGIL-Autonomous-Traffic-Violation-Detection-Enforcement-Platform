<<<<<<< HEAD
# VIGIL — Automated Traffic Violation Detection

Computer-vision pipeline that takes a traffic camera frame, detects vehicles/
riders/pedestrians, detects violations for whichever classes have actually
been trained, reads the offending vehicle's plate, auto-generates a challan,
and persists the full history for search, analytics, and evidence export.

Built for Flipkart GRiD — Theme: *Automated Photo Identification and
Classification for Traffic Violations Using Computer Vision*.

## A note on honesty (read this before demoing)

`config/violation_classes.json` is the single source of truth for which
violations the system can **actually** detect right now. It ships empty —
on purpose. An earlier version of this project hardcoded COCO class IDs
(person/car/motorcycle) as if they were violation classes, which meant it
confidently reported violations on ordinary, legal traffic photos. That bug
is fixed by construction: the detector will only ever report a violation
type that's explicitly listed in that config file, with a real trained
class index behind it.

Until you add a trained class, `/analyze/image` will correctly report
**zero violations** on every frame — which is the honest result, not a bug.
Vehicle/person/rider/pedestrian detection works independently of this and
is accurate today, because it uses the model's real COCO-trained classes.

### Activating a real violation class

1. Train a model for the class (see `core/evaluation.py` for measuring it).
2. Open your dataset's `data.yaml` and note the class index.
3. Add it to `config/violation_classes.json`:
   ```json
   "classes": {
     "1": { "label": "no_helmet", "severity_weight": 0.85 }
   }
   ```
### Two models, on purpose, not by accident

A custom model fine-tuned on just "with/without helmet" has **zero**
knowledge of "person," "car," or "motorcycle" anymore — COCO classes don't
survive narrow fine-tuning. An earlier version of this project shared one
model for both vehicle detection and violation detection, which silently
broke vehicle/person detection the moment a real trained model was plugged
in (vehicles showed as 0 even with people clearly in frame).

The fix: `ViolationDetector` now always loads a stock COCO model
(`VEHICLE_MODEL_PATH`, default `yolov8n.pt`) for vehicle/person/rider
detection, separate from `MODEL_PATH`, which is your trained violation
model. If both happen to point at the same file (demo mode, no violation
model trained yet), only one model is actually loaded — no wasted memory.

**On top of that**, violation detections are cross-checked against real
vehicle context before being trusted: a `no_helmet` detection needs an
actual motorcycle/bicycle nearby, or it's dropped as a likely false
positive outside the model's training domain (a helmet model trained on
motorcyclists will sometimes fire on a car driver's face — this catches
that). Dropped detections are logged to the terminal so it's auditable,
not silently hidden.

4. Point `MODEL_PATH` at your trained weights and restart the server.

### Two violations that work today without any training

Two of the seven classes don't need a trained model at all — they're derived
directly from the real vehicle/person detections, using actual geometry:

- **Triple riding** — if 3+ people are geometrically associated with the same
  two-wheeler in a single frame, that's triple riding by definition. Works on
  a single image, right now, via `POST /analyze/image`.
- **Wrong-side driving** — genuinely cannot be determined from one still
  frame (no direction of travel in a single image). `POST /analyze/video`
  tracks each vehicle's heading across a short clip and flags it if that
  heading deviates sharply from `CAMERA_EXPECTED_DIRECTION_DEG`, which you
  must calibrate per camera — there's no way to infer "correct direction"
  from pixels alone. Both are tagged `detection_basis` in the API response
  (`geometric_heuristic_rider_count` / `multi_frame_heading_heuristic`) so
  it's always clear these are real geometry, not a trained classifier.

The remaining five classes (no-helmet, no-seatbelt, stop-line, red-light,
illegal-parking) need either a trained model or context this prototype
doesn't have access to (signal state, parking-zone geofence) — see
"Known limitations" below.

## Architecture

```
                ┌──────────────┐   ┌────────────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌────────────┐
   raw frame → │ Preprocessor │ → │  ViolationDetector  │ → │ PlateReader  │ → │ ChallanGenerator│ → │  SQLite DB │
               │ low-light,   │   │ YOLOv8 single pass: │   │ PaddleOCR +  │   │ fine/section/   │   │ violations │
               │ blur, rain,  │   │  • vehicles/people  │   │ plate regex  │   │ risk level       │   │ history    │
               │ haze fixes   │   │    (real COCO cls)  │   │ + confidence │   │                  │   └────────────┘
               │              │   │  • violations       │   └──────────────┘   └─────────────────┘         │
               └──────────────┘   │   (config-driven,    │                                                  │
                                  │    honest by default)│                                                  ▼
                                  └────────────────────┘                                    ┌──────────────────────┐
                                                                                              │ FastAPI: search,      │
                                                                                              │ analytics, evidence   │
                                                                                              │ export, /analyze      │
                                                                                              └──────────┬───────────┘
                                                                                                          │
                                                                                  ┌───────────────────────┴───────────────────────┐
                                                                                  ▼                                                 ▼
                                                                     static/index.html (live single-frame UI)        streamlit_app.py (analytics, search,
                                                                                                                       evidence viewer, trend charts)
```

## Project structure

```
.
├── core/
│   ├── preprocessor.py        # adaptive image enhancement
│   ├── detector.py            # vehicle/person detection (real) + violation detection (config-driven)
│   ├── plate_reader.py        # PaddleOCR + Indian plate validation + confidence
│   ├── challan_generator.py   # violation → fine notice
│   ├── database.py            # SQLite persistence (search, summary, trend)
│   └── evaluation.py          # real precision/recall/F1/mAP + latency benchmarking
├── config/
│   └── violation_classes.json # which violation classes are actually trained/active
├── static/
│   └── index.html             # live single-frame demo dashboard
├── evidence/                  # annotated evidence images, one per challan
├── streamlit_app.py           # analytics dashboard (stats/search/evidence/charts)
├── main.py                    # FastAPI app
├── violations.db              # created on first run
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

Two processes, two terminals:

```bash
# Terminal 1 — API + live single-frame dashboard
uvicorn main:app --reload --port 8000
# open http://localhost:8000

# Terminal 2 — analytics dashboard
streamlit run streamlit_app.py
# opens automatically, default http://localhost:8501
```

The Streamlit app talks to the API over HTTP — both need to be running for
the Upload/Search/Evidence tabs to work.

## Using your own trained model

```bash
export MODEL_PATH=models/best.pt        # Windows: set MODEL_PATH=models\best.pt
uvicorn main:app --reload --port 8000
```

Remember to also update `config/violation_classes.json` with the real class
index(es) from your training run — pointing `MODEL_PATH` at new weights
without updating the config means the detector still won't report anything,
by design.

## Measuring scalability/throughput for real

```bash
# Terminal 1 — server must already be running
uvicorn main:app --port 8000

# Terminal 2
python -m core.loadtest --image path/to/sample.jpg --concurrency 5 --requests 30
```

This fires real concurrent requests at your own running server and reports
actual measured throughput (req/sec) and latency percentiles (mean/median/
p95/p99) under that load — this is the "computational efficiency and
scalability" evidence for task 8, measured rather than assumed. Raise
`--concurrency` to find where the system starts to degrade; that's a more
honest scalability statement than any claimed number without a number to
back it up.

## Evaluating accuracy and speed for real

```bash
python -m core.evaluation \
    --images-dir path/to/val/images \
    --labels-dir path/to/val/labels \
    --model-path models/best.pt \
    --report-out evaluation_report.json
```

This runs the actual pipeline against your own labeled validation images and
writes real precision/recall/F1/mAP@0.5 plus measured latency (mean/median/
p95/FPS) to `evaluation_report.json`. Omit `--labels-dir` to get a
latency-only report if you don't have ground-truth boxes yet. There are no
pre-filled numbers anywhere in this repo — every metric you'll see comes
from actually running this against your data.

## API reference

| Method | Path                          | Description |
|--------|-------------------------------|--------------|
| GET    | `/health`                     | Model path, demo-mode flag, active vs. pending violation classes |
| POST   | `/analyze/image`               | Upload a frame, run the full pipeline, get JSON + annotated image, persist to DB |
| POST   | `/analyze/video`               | Upload a short clip; tracks vehicle heading across frames to flag real wrong-side driving. `expected_direction_deg`/`deviation_threshold_deg` can be passed per-request to calibrate without restarting the server |
| GET    | `/violations/search`           | Filter by `plate`, `date_from`, `date_to`, `violation_type` |
| GET    | `/violations/recent`           | Most recent N violations (persistent, survives restarts) |
| GET    | `/analytics/summary`           | Totals, fine sum, unique plates, breakdown by type/risk |
| GET    | `/analytics/trend`             | Daily counts + fine totals over the last N days |
| GET    | `/analytics/report`            | Generates and downloads a real PDF summary report from current DB data |
| GET    | `/evidence/{challan_id}`       | Download the annotated evidence image for a challan |
| GET    | `/evidence/{challan_id}/metadata` | Full stored metadata for a challan |

## Known limitations (still true, stated plainly)

- **Stop-line and red-light violations are architecturally scoped but not
  implemented** — they need calibrated lane/stop-line geometry plus a
  synchronized signal-state feed, which a single still image cannot provide.
  This isn't a missing line of code; it's a different problem than
  single-frame object detection. `config/violation_classes.json` documents
  this explicitly under `pending_classes`.
- **No-helmet, no-seatbelt, and illegal-parking** still need a trained model
  (helmet/seatbelt) or a parking-zone geofence + stationary-time threshold
  (illegal-parking) — neither is something geometry alone can solve.
- **Triple riding and wrong-side driving now have real implementations**
  (see above) — triple riding from single-frame geometry, wrong-side from
  multi-frame heading tracking on video. Wrong-side accuracy is entirely
  dependent on `CAMERA_EXPECTED_DIRECTION_DEG` being correctly calibrated
  for the actual camera and lane — an uncalibrated default will misfire.
- `core/detector.py`'s rider/driver/pedestrian role assignment and the
  tracker in `core/tracker.py` are simple geometric heuristics, explicitly
  labeled as such in the API response (`role_basis`, `detection_basis`) —
  neither is a trained classifier.
- The `/analyze/video` endpoint doesn't run plate OCR per-frame (kept fast
  on longer clips) — evidence frames and bounding boxes are still saved.
- SQLite is fine for a prototype's demo data volume; a production
  deployment would move to Postgres and add image storage to blob/object
  storage instead of local disk.
=======
# VIGIL — AI-Powered Traffic Violation Detection & Enforcement Platform

VIGIL automates the path from raw traffic camera image to issued digital
challan: preprocessing → dual-model vehicle/violation detection → plate OCR
→ automated challan generation → persistent, searchable, analytics-ready
violation history.

## Why this is different
Violation classes only activate when a real, trained model backs them —
defined explicitly in `config/violation_classes.json`. There is no fallback
that silently treats an untrained class as detected. Try any photo; the
system will only ever report what it can actually verify.

## Architecture
Two YOLOv8 models run per frame: a stock COCO model for genuinely accurate
vehicle/person/rider detection, and a separately trained model for whichever
violation classes are currently activated. Violation detections are
cross-validated against real vehicle context before being trusted. Triple
riding and wrong-side driving are detected from real geometry (rider count
per vehicle; multi-frame heading tracking) rather than requiring a trained
classifier for either.

## Setup
\`\`\`bash
pip install -r requirements.txt
export MODEL_PATH=models/best.pt
uvicorn main:app --port 8000        # API + live dashboard, localhost:8000
streamlit run streamlit_app.py      # analytics dashboard, localhost:8501
\`\`\`

## Verifying performance yourself
\`\`\`bash
python -m core.evaluation --images-dir <val_images> --labels-dir <val_labels> --model-path models/best.pt
python -m core.loadtest --image <sample.jpg> --concurrency 5 --requests 30
\`\`\`
Both produce real measured output from your own run — nothing in this repo
asserts a performance number that can't be independently reproduced this way.

## Project structure
- `core/preprocessor.py` — adaptive image enhancement
- `core/detector.py` — dual-model vehicle + violation detection
- `core/plate_reader.py` — PaddleOCR + Indian plate validation
- `core/challan_generator.py` — fine/section/risk calculation
- `core/database.py` — SQLite persistence
- `core/tracker.py` — multi-frame vehicle heading tracking
- `core/report.py` — PDF report generation
- `core/evaluation.py` / `core/loadtest.py` — accuracy and throughput verification
- `main.py` — FastAPI app
- `static/index.html` / `streamlit_app.py` — the two dashboards

## Known limitations (stated, not hidden)
- Stop-line and red-light violations need signal-state + lane-geometry data
  no single camera frame can provide — out of scope by design, not a bug.
- Wrong-side detection requires per-camera direction calibration and only
  works on video, not single images — there's no way around that physically.
- Illegal parking needs a geofenced no-parking zone + stationary-time
  threshold, not yet implemented.
>>>>>>> 2d78ec3a1c771902b2728767bdcedf5c5bba1383
