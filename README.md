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
