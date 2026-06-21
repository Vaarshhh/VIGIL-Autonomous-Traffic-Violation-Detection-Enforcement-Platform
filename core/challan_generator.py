"""
core/challan_generator.py
─────────────────────────
Auto-generates a traffic violation notice (challan) as a dict / JSON.
This is a KEY DIFFERENTIATOR — closes the loop from detection to enforcement.
"""

from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Fine amounts (INR) per violation type ─────────────────────────────────────
FINE_SCHEDULE: dict[str, int] = {
    "no_helmet":           1000,
    "no_seatbelt":         1000,
    "triple_riding":       2000,
    "wrong_side":          5000,
    "stop_line_violation": 1000,
    "red_light_violation": 5000,
    "illegal_parking":     500,
}

MV_ACT_SECTIONS: dict[str, str] = {
    "no_helmet":           "Section 129 of MV Act 1988",
    "no_seatbelt":         "Section 138(3) of MV Act 1988",
    "triple_riding":       "Section 128 of MV Act 1988",
    "wrong_side":          "Section 119 of MV Act 1988",
    "stop_line_violation": "Section 119 of MV Act 1988",
    "red_light_violation": "Section 119 of MV Act 1988",
    "illegal_parking":     "Section 122 of MV Act 1988",
}


@dataclass
class ChallanItem:
    violation:  str
    fine_inr:   int
    section:    str
    confidence: float
    severity:   float


@dataclass
class Challan:
    challan_id:    str
    plate_number:  Optional[str]
    camera_id:     str
    location:      str
    timestamp:     str
    items:         list[ChallanItem] = field(default_factory=list)
    total_fine:    int = 0
    risk_level:    str = "LOW"
    image_ref:     Optional[str] = None   # path to annotated evidence frame

    def to_dict(self) -> dict:
        d = asdict(self)
        d["items"] = [asdict(i) for i in self.items]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary_line(self) -> str:
        viols = ", ".join(i.violation.replace("_", " ") for i in self.items)
        return (
            f"Challan {self.challan_id} | "
            f"Plate: {self.plate_number or 'UNREAD'} | "
            f"Violations: {viols} | "
            f"Fine: ₹{self.total_fine:,} | "
            f"Risk: {self.risk_level}"
        )


class ChallanGenerator:
    """
    Converts ViolationResult detections into a structured Challan.

    Usage
    -----
    gen     = ChallanGenerator(camera_id="CAM_01", location="MG Road Junction")
    challan = gen.generate(violation_result)
    print(challan.to_json())
    """

    def __init__(
        self,
        camera_id: str = "CAM_01",
        location:  str = "Unknown Junction",
    ):
        self.camera_id = camera_id
        self.location  = location

    def generate(self, result, image_ref: Optional[str] = None) -> Optional[Challan]:
        """
        result: ViolationResult from detector.py
        Returns None if no violations detected.
        """
        if not result.detections:
            return None

        plate = next(
            (d.plate_text for d in result.detections if d.plate_text), None
        )

        challan = Challan(
            challan_id=self._new_id(),
            plate_number=plate,
            camera_id=self.camera_id,
            location=self.location,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            image_ref=image_ref,
        )

        for det in result.detections:
            item = ChallanItem(
                violation=det.label,
                fine_inr=FINE_SCHEDULE.get(det.label, 500),
                section=MV_ACT_SECTIONS.get(det.label, "MV Act 1988"),
                confidence=round(det.confidence, 3),
                severity=det.severity,
            )
            challan.items.append(item)

        challan.total_fine = sum(i.fine_inr for i in challan.items)
        challan.risk_level = self._risk_level(result.max_severity)
        return challan

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _new_id() -> str:
        ts  = time.strftime("%Y%m%d%H%M%S")
        uid = str(uuid.uuid4())[:6].upper()
        return f"CHN-{ts}-{uid}"

    @staticmethod
    def _risk_level(severity: float) -> str:
        if severity >= 0.85:
            return "CRITICAL"
        if severity >= 0.70:
            return "HIGH"
        if severity >= 0.50:
            return "MEDIUM"
        return "LOW"