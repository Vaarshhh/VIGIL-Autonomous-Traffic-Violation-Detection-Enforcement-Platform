"""
main.py
───────
FastAPI entrypoint. Wires the full pipeline together:

    raw frame -> ImagePreprocessor -> ViolationDetector -> PlateReader
              -> ChallanGenerator -> SQLite persistence -> JSON + annotated frame

Plus search, analytics, and evidence-export routes backed by core/database.py.

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core import database as db
from core.challan_generator import ChallanGenerator
from core.detector import Detection, PENDING_CLASS_NAMES, VIOLATION_LABELS, ViolationDetector
from core.plate_reader import PlateReader
from core.preprocessor import ImagePreprocessor
from core.report import generate_summary_report
from core.tracker import VehicleTracker, heading_deviation

# ── configuration ──────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "violation_classes.json"
_config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

MODEL_PATH = os.environ.get("MODEL_PATH", _config.get("model_path", "yolov8n.pt"))
# Vehicle/person detection always needs real COCO classes — independent of
# whatever violation model is loaded, which may have zero COCO classes left
# after fine-tuning. Override only if you have a different stock model.
VEHICLE_MODEL_PATH = os.environ.get("VEHICLE_MODEL_PATH", "yolov8n.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))
CAMERA_ID = os.environ.get("CAMERA_ID", "CAM_01")
LOCATION = os.environ.get("LOCATION", "Demo Junction")
MAX_DIM = int(os.environ.get("MAX_DIM", "960"))

# Wrong-side detection requires knowing which direction is "correct" for this
# specific camera's lane — there's no way to infer this from pixels alone.
# 0 = vehicles should move left-to-right in the frame, 180 = right-to-left,
# 90 = top-to-bottom, 270 = bottom-to-top. Calibrate per camera installation.
CAMERA_EXPECTED_DIRECTION_DEG = float(os.environ.get("CAMERA_EXPECTED_DIRECTION_DEG", "0"))
WRONG_SIDE_DEVIATION_THRESHOLD_DEG = float(os.environ.get("WRONG_SIDE_DEVIATION_THRESHOLD_DEG", "100"))

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

# Honest demo-mode flag — true only because no real classes are configured yet,
# not because we're hiding anything. See config/violation_classes.json.
DEMO_MODE = len(VIOLATION_LABELS) == 0

# ── app + pipeline singletons (loaded once at startup, not per-request) ─────
app = FastAPI(title="Traffic Violation Detection API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor = ImagePreprocessor()
detector = ViolationDetector(model_path=MODEL_PATH, vehicle_model_path=VEHICLE_MODEL_PATH, conf_threshold=CONF_THRESHOLD)
plate_reader = PlateReader(use_paddle=True)
challan_gen = ChallanGenerator(camera_id=CAMERA_ID, location=LOCATION)

db.init_db()

_frame_counter = {"n": 0}


# ── routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "vehicle_model_path": VEHICLE_MODEL_PATH,
        "demo_mode": DEMO_MODE,
        "active_violation_classes": list(VIOLATION_LABELS.values()),
        "pending_violation_classes": PENDING_CLASS_NAMES,
        "ocr_engine": getattr(plate_reader, "_engine", "unknown"),
        "conf_threshold": CONF_THRESHOLD,
        "frames_processed": _frame_counter["n"],
    }


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image — is this a valid JPG/PNG?")

    h, w = frame.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    t0 = time.perf_counter()

    enhanced = preprocessor.enhance(frame)
    t_pre = time.perf_counter()

    _frame_counter["n"] += 1
    result = detector.process_frame(enhanced, frame_id=_frame_counter["n"], draw=False)
    t_det = time.perf_counter()

    for det in result.detections:
        plate_result = plate_reader.read_from_frame(enhanced, det.bbox)
        if plate_result:
            det.plate_text = plate_result.text
            det.plate_confidence = plate_result.confidence
    t_ocr = time.perf_counter()

    annotated = detector._annotate(enhanced.copy(), result.detections, result.vehicles)
    result.annotated = annotated
    result.proc_ms = round((time.perf_counter() - t0) * 1000, 1)

    print(
        f"[timing] preprocess={round((t_pre - t0) * 1000, 1)}ms  "
        f"detect={round((t_det - t_pre) * 1000, 1)}ms  "
        f"ocr={round((t_ocr - t_det) * 1000, 1)}ms  "
        f"total={result.proc_ms}ms  "
        f"applied={preprocessor.last_report}"
    )

    challan = challan_gen.generate(result)
    evidence_path_str = None

    if challan:
        evidence_filename = f"{challan.challan_id}.jpg"
        evidence_full_path = EVIDENCE_DIR / evidence_filename
        cv2.imwrite(str(evidence_full_path), annotated)
        evidence_path_str = str(evidence_full_path)

        for item, det in zip(challan.items, result.detections):
            db.insert_violation({
                "challan_id": challan.challan_id,
                "frame_id": result.frame_id,
                "violation_type": item.violation,
                "confidence": item.confidence,
                "severity": item.severity,
                "fine_inr": item.fine_inr,
                "section": item.section,
                "risk_level": challan.risk_level,
                "plate_number": challan.plate_number,
                "plate_confidence": det.plate_confidence,
                "camera_id": challan.camera_id,
                "location": challan.location,
                "timestamp": challan.timestamp,
                "evidence_path": evidence_path_str,
            })

    ok, buf = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buf).decode("utf-8") if ok else None

    return {
        "result": result.to_dict(),
        "challan": challan.to_dict() if challan else None,
        "annotated_image": annotated_b64,
        "preprocessing_applied": preprocessor.last_report,
        "demo_mode": DEMO_MODE,
    }


@app.get("/violations/search")
def search_violations(
    plate: Optional[str] = None,
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    violation_type: Optional[str] = None,
    limit: int = 200,
):
    results = db.search(plate=plate, date_from=date_from, date_to=date_to, violation_type=violation_type, limit=limit)
    return {"count": len(results), "results": results}


@app.get("/violations/recent")
def recent_violations(limit: int = 20):
    return {"count": limit, "results": db.get_recent(limit=limit)}


@app.get("/analytics/summary")
def analytics_summary():
    return db.get_summary()


@app.get("/analytics/trend")
def analytics_trend(days: int = 30):
    return {"days": days, "trend": db.get_trend(days=days)}


@app.get("/analytics/report")
def analytics_report():
    report_path = EVIDENCE_DIR.parent / "violation_report.pdf"
    generate_summary_report(str(report_path))
    return FileResponse(str(report_path), media_type="application/pdf", filename="violation_report.pdf")


@app.get("/evidence/{challan_id}")
def export_evidence(challan_id: str):
    rows = db.get_by_challan_id(challan_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No record found for this challan_id")
    evidence_path = rows[0].get("evidence_path")
    if not evidence_path or not Path(evidence_path).exists():
        raise HTTPException(status_code=404, detail="Evidence image not found on disk")
    return FileResponse(evidence_path, media_type="image/jpeg", filename=f"{challan_id}.jpg")


@app.get("/evidence/{challan_id}/metadata")
def export_evidence_metadata(challan_id: str):
    rows = db.get_by_challan_id(challan_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No record found for this challan_id")
    return {"challan_id": challan_id, "violations": rows}


@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    frame_skip: int = 5,
    max_frames: int = 150,
    expected_direction_deg: Optional[float] = None,
    deviation_threshold_deg: Optional[float] = None,
):
    """
    Wrong-side driving genuinely cannot be detected from a single image — there's
    no direction of travel in one frame. This endpoint is the honest alternative:
    it tracks each vehicle's heading across the clip and flags it if that heading
    deviates sharply from the expected direction (calibrate per camera — there's
    no way to infer "correct" direction from pixels alone).

    expected_direction_deg / deviation_threshold_deg can be passed per-request to
    override the .env defaults — useful for testing calibration live without
    restarting the server.

    Note: plate OCR is skipped here for speed on longer clips; the evidence frame
    and bounding box are still saved, but plate reads are an /analyze/image thing.
    """
    expected_dir = expected_direction_deg if expected_direction_deg is not None else CAMERA_EXPECTED_DIRECTION_DEG
    threshold = deviation_threshold_deg if deviation_threshold_deg is not None else WRONG_SIDE_DEVIATION_THRESHOLD_DEG

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="Could not open video file")

    tracker = VehicleTracker()
    frame_idx, processed = 0, 0
    flagged_track_ids: set[int] = set()
    wrong_side_events: list[dict] = []

    try:
        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            scale = MAX_DIM / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            enhanced = preprocessor.enhance(frame)
            result = detector.process_frame(enhanced, frame_id=frame_idx, draw=False)

            real_vehicles = [v for v in result.vehicles if v.category != "person"]
            tracked = tracker.update(real_vehicles)

            for t in tracked:
                if t["heading_deg"] is None or t["track_id"] in flagged_track_ids:
                    continue
                deviation = heading_deviation(t["heading_deg"], expected_dir)
                if deviation <= threshold:
                    continue

                wrong_det = Detection(
                    label="wrong_side",
                    confidence=t["confidence"],
                    bbox=t["bbox"],
                    detection_basis="multi_frame_heading_heuristic",
                )
                result.detections.append(wrong_det)
                flagged_track_ids.add(t["track_id"])
                wrong_side_events.append({
                    "track_id": t["track_id"],
                    "frame_id": frame_idx,
                    "heading_deg": round(t["heading_deg"], 1),
                    "deviation_deg": round(deviation, 1),
                    "bbox": t["bbox"],
                })

            if result.detections:
                annotated = detector._annotate(enhanced.copy(), result.detections, result.vehicles)
                challan = challan_gen.generate(result)
                if challan:
                    evidence_full_path = EVIDENCE_DIR / f"{challan.challan_id}.jpg"
                    cv2.imwrite(str(evidence_full_path), annotated)
                    for item in challan.items:
                        db.insert_violation({
                            "challan_id": challan.challan_id,
                            "frame_id": result.frame_id,
                            "violation_type": item.violation,
                            "confidence": item.confidence,
                            "severity": item.severity,
                            "fine_inr": item.fine_inr,
                            "section": item.section,
                            "risk_level": challan.risk_level,
                            "plate_number": challan.plate_number,
                            "plate_confidence": None,
                            "camera_id": challan.camera_id,
                            "location": challan.location,
                            "timestamp": challan.timestamp,
                            "evidence_path": str(evidence_full_path),
                        })

            processed += 1
            frame_idx += 1
    finally:
        cap.release()
        os.unlink(tmp_path)

    return {
        "frames_processed": processed,
        "expected_direction_deg": expected_dir,
        "deviation_threshold_deg": threshold,
        "wrong_side_events": wrong_side_events,
        "note": (
            "expected_direction_deg must be calibrated to this camera's real "
            "traffic direction — there's no way to infer it from pixels alone. "
            "Wrong defaults will produce wrong flags."
        ),
    }


# Serve the live single-frame dashboard at the root URL.
@app.get("/")
def dashboard():
    return FileResponse("static/index.html")
