"""
core/tracker.py
─────────────────
Lightweight, dependency-free multi-frame tracker for vehicles.

Why this exists: "wrong side driving" cannot be determined from a single
still image — there's no direction of travel in one frame. It CAN be
determined honestly from a short video clip, by tracking each vehicle's
centroid across consecutive frames and comparing its heading angle to the
expected direction of travel for that camera/lane.

This is a simple greedy IoU-matching tracker (no deep-learning re-ID model,
no Kalman filter) — appropriate for a prototype, and transparent about what
it is: a small amount of real geometry, not a trained model.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field


def _centroid(bbox: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


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


@dataclass
class Track:
    track_id:    int
    category:    str
    history:     list[tuple[float, float]] = field(default_factory=list)  # centroids, oldest first
    last_bbox:   tuple = None
    frames_since_seen: int = 0

    def heading_deg(self) -> float | None:
        """Direction of travel in degrees (0 = pointing right/east, 90 = down/south
        in image coordinates), or None if there isn't enough history yet."""
        if len(self.history) < 3:
            return None
        x0, y0 = self.history[0]
        x1, y1 = self.history[-1]
        if math.hypot(x1 - x0, y1 - y0) < 8:   # essentially stationary — no reliable heading
            return None
        return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360


class VehicleTracker:
    """
    Call `update(vehicles)` once per frame with the current frame's
    VehicleDetection list (vehicles only — cars/motorcycles/etc, not people).
    Returns the same objects with `.track_id` and `.heading_deg` attached
    (monkeypatched on, since VehicleDetection doesn't carry these by default).
    """

    def __init__(self, iou_match_threshold: float = 0.2, max_missed_frames: int = 5, max_history: int = 15):
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_frames = max_missed_frames
        self.max_history = max_history

    def update(self, vehicles: list) -> list[dict]:
        unmatched_tracks = set(self.tracks.keys())
        enriched = []

        for veh in vehicles:
            best_id, best_iou = None, 0.0
            for tid in unmatched_tracks:
                track = self.tracks[tid]
                if track.category != veh.category or track.last_bbox is None:
                    continue
                i = _iou(veh.bbox, track.last_bbox)
                if i > best_iou and i > self.iou_match_threshold:
                    best_iou, best_id = i, tid

            if best_id is not None:
                track = self.tracks[best_id]
                unmatched_tracks.discard(best_id)
            else:
                track = Track(track_id=self._next_id, category=veh.category)
                self.tracks[track.track_id] = track
                self._next_id += 1

            track.last_bbox = veh.bbox
            track.history.append(_centroid(veh.bbox))
            track.history = track.history[-self.max_history:]
            track.frames_since_seen = 0

            enriched.append({
                "track_id": track.track_id,
                "category": veh.category,
                "confidence": veh.confidence,
                "bbox": veh.bbox,
                "heading_deg": track.heading_deg(),
            })

        # age out tracks we didn't see this frame
        for tid in list(self.tracks.keys()):
            if tid in unmatched_tracks:
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_missed_frames:
                    del self.tracks[tid]

        return enriched


def heading_deviation(heading_deg: float, expected_deg: float) -> float:
    """Angular difference in degrees, 0-180, regardless of direction of rotation."""
    diff = abs(heading_deg - expected_deg) % 360
    return min(diff, 360 - diff)
