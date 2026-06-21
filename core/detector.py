"""
core/detector.py
────────────────
Central detection engine.

Runs YOLOv8 in a single inference pass and splits the results into two
honestly-separated buckets:

  1. VEHICLE / PERSON detection — uses the model's real COCO-trained classes
     (person, bicycle, car, motorcycle, bus, truck). These are genuinely
     accurate even on stock yolov8n.pt, because that's what it was trained
     to do. Rider/driver/pedestrian role is then assigned by a simple,
     documented bbox-overlap heuristic — NOT a trained classifier.

  2. VIOLATION detection — uses ONLY the class mapping defined in
     config/violation_classes.json under "classes". That dict starts empty
     and stays empty until a class has been genuinely trained. This is the
     fix for the earlier bug where COCO class IDs were silently relabeled
     as violations (a "person" becoming "no_helmet", etc).
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "violation_classes.json"

# ── real COCO classes used for vehicle/person detection (genuinely accurate
#    on stock weights — this is what COCO models are actually trained for) ──
VEHICLE_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── bounding-box colours per violation (BGR) ──────────────────────────────────
BBOX_COLORS: dict[str, tuple] = {
    "no_helmet":           (0,   0,   255),   # red
    "no_seatbelt":         (0,  80,   255),   # orange-red
    "triple_riding":       (0, 165,   255),   # orange
    "wrong_side":          (0,   0,   180),   # dark red
    "stop_line_violation": (255, 0,   150),   # magenta
    "red_light_violation": (0,   0,   255),   # red
    "illegal_parking":     (255, 100,   0),   # blue-ish
}
VEHICLE_BOX_COLOR = (180, 180, 180)   # neutral gray — clearly distinct from violation boxes


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"[ViolationDetector] WARNING — {CONFIG_PATH} not found. No violation classes will be active.")
        return {"model_path": "yolov8n.pt", "model_status": "DEMO_STOCK_WEIGHTS", "classes": {}}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_CONFIG = _load_config()
VIOLATION_LABELS: dict[int, str] = {int(k): v["label"] for k, v in _CONFIG.get("classes", {}).items()}
SEVERITY_WEIGHTS: dict[str, float] = {
    v["label"]: v.get("severity_weight", 0.5) for v in _CONFIG.get("classes", {}).values()
}
PENDING_CLASS_NAMES: list[str] = [c["label"] for c in _CONFIG.get("pending_classes", [])]

# Severity weights for violations derived by geometric/multi-frame heuristics
# rather than a trained model class. Kept separate from the trained-class
# config on purpose — these don't require training data, just real geometry.
HEURISTIC_SEVERITY_WEIGHTS: dict[str, float] = {
    "triple_riding": 0.75,
    "wrong_side": 0.95,
}
TRIPLE_RIDING_MIN_RIDERS = 3
RIDER_VEHICLE_ASSOCIATION_THRESHOLD = 0.15

# Some violation classes only make sense in a specific vehicle context — a
# "no_helmet" detection on a car driver's face is almost certainly a false
# positive outside the model's training domain (helmet models are trained on
# motorcyclists, not car interiors). This cross-checks the violation model's
# output against the SEPARATE, real vehicle detection before trusting it.
VEHICLE_CONTEXT_REQUIREMENTS: dict[str, set[str]] = {
    "no_helmet": {"motorcycle", "bicycle"},
    "no_seatbelt": {"car", "bus", "truck"},
}


@dataclass
class Detection:
    label:            str
    confidence:        float
    bbox:              tuple[int, int, int, int]   # x1, y1, x2, y2
    severity:          float = 0.0
    plate_text:        Optional[str] = None
    plate_confidence:  Optional[float] = None
    detection_basis:   str = "trained_model"   # or "geometric_heuristic_rider_count", "multi_frame_heading_heuristic"

    def __post_init__(self):
        weight = SEVERITY_WEIGHTS.get(self.label) or HEURISTIC_SEVERITY_WEIGHTS.get(self.label, 0.5)
        self.severity = round(weight * self.confidence, 3)


@dataclass
class VehicleDetection:
    """A real, honestly-labeled COCO detection (person/vehicle) — not a violation."""
    category:                str                              # "person", "car", "motorcycle", ...
    confidence:              float
    bbox:                    tuple[int, int, int, int]
    role:                    Optional[str] = None              # "rider" | "driver" | "pedestrian" | None
    role_basis:              str = "heuristic_bbox_overlap"     # documents that role is NOT a trained classifier
    associated_rider_count:  Optional[int] = None               # for two-wheelers — how many riders are on THIS specific vehicle


@dataclass
class ViolationResult:
    frame_id:    int
    timestamp:   float
    detections:  list[Detection] = field(default_factory=list)
    vehicles:    list[VehicleDetection] = field(default_factory=list)
    annotated:   Optional[np.ndarray] = None   # annotated BGR frame
    proc_ms:     float = 0.0                   # inference time in ms

    @property
    def violation_count(self) -> int:
        return len(self.detections)

    @property
    def max_severity(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.severity for d in self.detections)

    def to_dict(self) -> dict:
        return {
            "frame_id":   self.frame_id,
            "timestamp":  self.timestamp,
            "proc_ms":    self.proc_ms,
            "violation_count": self.violation_count,
            "max_severity":    self.max_severity,
            "detections": [
                {
                    "label":            d.label,
                    "confidence":       round(d.confidence, 3),
                    "severity":         d.severity,
                    "bbox":             d.bbox,
                    "plate":            d.plate_text,
                    "plate_confidence": d.plate_confidence,
                    "detection_basis":  d.detection_basis,
                }
                for d in self.detections
            ],
            "vehicles": [
                {
                    "category":               v.category,
                    "confidence":             round(v.confidence, 3),
                    "bbox":                   v.bbox,
                    "role":                   v.role,
                    "role_basis":             v.role_basis,
                    "associated_rider_count": v.associated_rider_count,
                }
                for v in self.vehicles
            ],
        }


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _vertical_overlap_ratio(person_bbox: tuple, vehicle_bbox: tuple) -> float:
    """How much of the person's bbox horizontally overlaps the vehicle's footprint —
    used to associate a person with a nearby bike/car for role assignment."""
    px1, _, px2, _ = person_bbox
    vx1, _, vx2, _ = vehicle_bbox
    ix1, ix2 = max(px1, vx1), min(px2, vx2)
    overlap = max(0, ix2 - ix1)
    width = max(1, px2 - px1)
    return overlap / width


def _has_nearby_vehicle(bbox: tuple, vehicles: list, categories: set, threshold: float = 0.15) -> bool:
    """Used to cross-validate a violation detection against real vehicle
    context — e.g. a 'no_helmet' box should have an actual motorcycle/bicycle
    nearby, or it's likely a false positive outside the model's training domain."""
    for veh in vehicles:
        if veh.category not in categories:
            continue
        overlap = max(_iou(bbox, veh.bbox), _vertical_overlap_ratio(bbox, veh.bbox) * 0.5)
        if overlap > threshold:
            return True
    return False


def _assign_roles(persons: list[VehicleDetection], vehicles: list[VehicleDetection]) -> dict[int, list[VehicleDetection]]:
    """
    Heuristic, NOT a trained classifier: a person is labeled 'rider' if they
    significantly overlap a two-wheeler, 'driver' if they overlap a car/bus/truck,
    otherwise 'pedestrian'. Mutates `persons` in place.

    Returns a mapping of id(vehicle) -> list of rider persons associated with
    that specific two-wheeler, used by `_detect_triple_riding` below.
    """
    two_wheelers = [v for v in vehicles if v.category in ("motorcycle", "bicycle")]
    four_wheelers = [v for v in vehicles if v.category in ("car", "bus", "truck")]
    riders_by_vehicle: dict[int, list[VehicleDetection]] = {}

    for person in persons:
        best_overlap = 0.0
        best_role = "pedestrian"
        best_vehicle = None
        for veh in two_wheelers:
            ov = max(_iou(person.bbox, veh.bbox), _vertical_overlap_ratio(person.bbox, veh.bbox) * 0.5)
            if ov > best_overlap and ov > RIDER_VEHICLE_ASSOCIATION_THRESHOLD:
                best_overlap, best_role, best_vehicle = ov, "rider", veh
        for veh in four_wheelers:
            ov = _iou(person.bbox, veh.bbox)
            if ov > best_overlap and ov > 0.10:
                best_overlap, best_role, best_vehicle = ov, "driver", None
        person.role = best_role
        if best_role == "rider" and best_vehicle is not None:
            riders_by_vehicle.setdefault(id(best_vehicle), []).append(person)

    return riders_by_vehicle


def _detect_triple_riding(vehicles: list[VehicleDetection], riders_by_vehicle: dict[int, list[VehicleDetection]]) -> list[Detection]:
    """
    Real detection, not a trained model: if 3+ people are geometrically
    associated with the same two-wheeler, that's triple riding by definition.
    Confidence is the mean of the riders' own detection confidences;
    bbox is the union of the vehicle + all its associated riders so the
    evidence box visibly contains everyone involved.
    """
    detections: list[Detection] = []
    for veh in vehicles:
        riders = riders_by_vehicle.get(id(veh), [])
        if len(riders) >= TRIPLE_RIDING_MIN_RIDERS:
            boxes = [veh.bbox] + [r.bbox for r in riders]
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            mean_conf = sum(r.confidence for r in riders) / len(riders)
            detections.append(Detection(
                label="triple_riding",
                confidence=round(mean_conf, 3),
                bbox=(x1, y1, x2, y2),
                detection_basis="geometric_heuristic_rider_count",
            ))
    return detections


class ViolationDetector:
    """
    Runs TWO models per frame, intentionally:
      - `vehicle_model_path`: always a stock COCO model, used for genuinely
        accurate vehicle/person/rider detection.
      - `model_path`: the (possibly custom-trained) violation model.

    Why two models: a custom model fine-tuned on e.g. just "with/without
    helmet" has ZERO knowledge of "person"/"car"/"motorcycle" anymore — COCO
    classes don't survive fine-tuning on a narrow class set. Sharing one
    model for both jobs (the original design) silently breaks vehicle
    detection the moment a real violation model is plugged in.

    If both paths are identical (e.g. still on stock yolov8n.pt with no
    violation classes trained yet), only one model is actually loaded —
    no wasted memory/inference in that demo-mode case.

    Usage
    -----
    detector = ViolationDetector("models/best.pt", conf_threshold=0.45)
    result   = detector.process_frame(bgr_frame, frame_id=0)
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        vehicle_model_path: str | Path = "yolov8n.pt",
        conf_threshold: float = 0.45,
        iou_threshold:  float = 0.45,
        device: str = "cpu",
    ):
        self.conf  = conf_threshold
        self.iou   = iou_threshold

        self.violation_model = YOLO(str(model_path))
        self.violation_model.to(device)

        self._shared_model = str(vehicle_model_path) == str(model_path)
        if self._shared_model:
            self.vehicle_model = self.violation_model
        else:
            self.vehicle_model = YOLO(str(vehicle_model_path))
            self.vehicle_model.to(device)

        active = list(VIOLATION_LABELS.values())
        print(f"[ViolationDetector] violation model: {model_path} on {device}")
        print(
            f"[ViolationDetector] vehicle/person model: {vehicle_model_path} on {device}"
            + (" (sharing weights with violation model)" if self._shared_model else " (separate model)")
        )
        if active:
            print(f"[ViolationDetector] ACTIVE violation classes: {active}")
        else:
            print(
                "[ViolationDetector] No violation classes are active yet "
                f"(see {CONFIG_PATH}). Vehicle/person detection still runs normally; "
                "violation detection will correctly report zero until a real class is configured."
            )
        if PENDING_CLASS_NAMES:
            print(f"[ViolationDetector] Pending (not yet trained): {PENDING_CLASS_NAMES}")

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        draw: bool = True,
    ) -> ViolationResult:
        t0 = time.perf_counter()

        detections: list[Detection] = []
        vehicles: list[VehicleDetection] = []

        if self._shared_model:
            results = self.violation_model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = (x1, y1, x2, y2)
                    if cls_id in VIOLATION_LABELS:
                        detections.append(Detection(label=VIOLATION_LABELS[cls_id], confidence=conf, bbox=bbox))
                    elif cls_id in VEHICLE_CLASSES:
                        vehicles.append(VehicleDetection(category=VEHICLE_CLASSES[cls_id], confidence=conf, bbox=bbox))
        else:
            violation_results = self.violation_model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
            for r in violation_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VIOLATION_LABELS:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(Detection(label=VIOLATION_LABELS[cls_id], confidence=conf, bbox=(x1, y1, x2, y2)))

            vehicle_results = self.vehicle_model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
            for r in vehicle_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    vehicles.append(VehicleDetection(category=VEHICLE_CLASSES[cls_id], confidence=conf, bbox=(x1, y1, x2, y2)))

        # cross-validate trained-model violations against real vehicle context —
        # fixes e.g. a "no_helmet" false positive on a car driver with no motorcycle nearby
        validated: list[Detection] = []
        for det in detections:
            required = VEHICLE_CONTEXT_REQUIREMENTS.get(det.label)
            if required and not _has_nearby_vehicle(det.bbox, vehicles, required):
                print(
                    f"[ViolationDetector] dropped out-of-context '{det.label}' detection "
                    f"(no {'/'.join(sorted(required))} nearby — likely false positive outside training domain)"
                )
                continue
            validated.append(det)
        detections = validated

        persons = [v for v in vehicles if v.category == "person"]
        riders_by_vehicle = _assign_roles(persons, vehicles)
        for veh in vehicles:
            if veh.category in ("motorcycle", "bicycle"):
                veh.associated_rider_count = len(riders_by_vehicle.get(id(veh), []))
        detections.extend(_detect_triple_riding(vehicles, riders_by_vehicle))

        annotated = self._annotate(frame.copy(), detections, vehicles) if draw else None
        proc_ms   = round((time.perf_counter() - t0) * 1000, 1)

        return ViolationResult(
            frame_id=frame_id,
            timestamp=time.time(),
            detections=detections,
            vehicles=vehicles,
            annotated=annotated,
            proc_ms=proc_ms,
        )

    def process_video(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        frame_skip: int = 2,
    ):
        """Generator — yields ViolationResult for each processed frame."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        writer = None
        if output_path:
            w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % frame_skip == 0:
                    result = self.process_frame(frame, frame_id=frame_id)
                    if writer and result.annotated is not None:
                        writer.write(result.annotated)
                    yield result
                frame_id += 1
        finally:
            cap.release()
            if writer:
                writer.release()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        vehicles: list[VehicleDetection],
    ) -> np.ndarray:
        # vehicles/persons drawn first, thin neutral boxes, so violation boxes stay visually dominant
        for veh in vehicles:
            x1, y1, x2, y2 = veh.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), VEHICLE_BOX_COLOR, 1)
            tag = veh.role if (veh.role and veh.category == "person") else veh.category
            label_text = f"{tag.upper()} {veh.confidence:.0%}"
            if veh.associated_rider_count is not None:
                label_text += f" ({veh.associated_rider_count} rider{'s' if veh.associated_rider_count != 1 else ''})"
            cv2.putText(
                frame, label_text, (x1 + 2, y2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, VEHICLE_BOX_COLOR, 1,
            )

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = BBOX_COLORS.get(det.label, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = (
                f"{det.label.replace('_', ' ').upper()}  "
                f"{det.confidence:.0%}  sev:{det.severity:.2f}"
            )
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                frame, label_text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

            if det.plate_text:
                conf_str = f" ({det.plate_confidence:.0%})" if det.plate_confidence else ""
                cv2.putText(
                    frame, f"Plate: {det.plate_text}{conf_str}",
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, f"Violations: {len(detections)}  |  Vehicles/People: {len(vehicles)}  |  {ts}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        return frame
