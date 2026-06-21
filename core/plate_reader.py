"""
core/plate_reader.py
────────────────────
License plate detection + OCR.
Uses PaddleOCR (best accuracy) with EasyOCR as fallback.
Includes Indian plate format validation.
"""

from __future__ import annotations
import re
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Indian plate regex patterns ───────────────────────────────────────────────
# Format: XX 00 XX 0000  (state code | district | series | number)
_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"),   # standard
    re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"),      # variant
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$"),               # BH series
]


def _is_valid_indian_plate(text: str) -> bool:
    clean = text.upper().replace(" ", "").replace("-", "")
    return any(p.match(clean) for p in _PLATE_PATTERNS)


@dataclass
class PlateResult:
    text:             str
    confidence:       float            # mean OCR confidence across recognized text lines, 0-1
    is_valid_format:  bool             # matches a known Indian registration pattern


class PlateReader:
    """
    Extracts and validates Indian license plate text from a cropped region.

    Usage
    -----
    reader = PlateReader()
    result = reader.read(plate_roi_bgr)   # PlateResult | None
    """

    def __init__(self, use_paddle: bool = True):
        self.ocr = None
        self._init_ocr(use_paddle)

    def _init_ocr(self, use_paddle: bool):
        if use_paddle:
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR 3.x renamed/removed several init params vs 2.x:
                #   - use_angle_cls  -> use_textline_orientation
                #   - show_log       -> removed entirely (no replacement)
                self.ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                )
                self._engine = "paddle"
                print("[PlateReader] using PaddleOCR")
                return
            except ImportError:
                print("[PlateReader] PaddleOCR not found, falling back to EasyOCR")

        try:
            import easyocr
            self.ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._engine = "easy"
            print("[PlateReader] using EasyOCR")
        except ImportError:
            print("[PlateReader] WARNING — no OCR engine installed. "
                  "Run: pip install paddleocr  OR  pip install easyocr")
            self._engine = "none"

    # ── public API ────────────────────────────────────────────────────────────

    def read(self, roi: np.ndarray) -> Optional[PlateResult]:
        """
        Takes a cropped BGR image of a license plate region.
        Returns a PlateResult (text + confidence + format validity) or None.
        """
        if self._engine == "none" or self.ocr is None:
            return None

        enhanced = self._preprocess_plate(roi)
        raw_text, raw_conf = self._run_ocr(enhanced)
        if not raw_text:
            return None

        cleaned = self._clean(raw_text)
        if not cleaned or len(cleaned) < 6:
            return None   # too little to be a usable plate read

        is_valid = _is_valid_indian_plate(cleaned)
        final_text = cleaned.upper().replace(" ", "") if is_valid else cleaned
        return PlateResult(text=final_text, confidence=round(raw_conf, 3), is_valid_format=is_valid)

    def read_from_frame(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: int = 10,
    ) -> Optional[PlateResult]:
        """Crop the plate region from a full frame and read it."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        roi = frame[y1:y2, x1:x2]
        return self.read(roi) if roi.size > 0 else None

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_plate(roi: np.ndarray) -> np.ndarray:
        # upscale small plates
        h, w = roi.shape[:2]
        if w < 200:
            scale = 200 / w
            roi   = cv2.resize(roi, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
        # grayscale + threshold
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # CLAHE for uneven lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    def _run_ocr(self, img: np.ndarray) -> tuple[Optional[str], float]:
        """Returns (joined_text, mean_confidence). mean_confidence is 0.0 if nothing recognized."""
        try:
            if self._engine == "paddle":
                # PaddleOCR 3.x: .ocr() is deprecated, use .predict() instead.
                # Results are dict-like objects keyed by rec_texts / rec_scores
                # (no more [box, [text, confidence]] line tuples).
                results = self.ocr.predict(img)
                texts, scores = [], []
                for res in results:
                    rec_texts  = res["rec_texts"]
                    rec_scores = res["rec_scores"]
                    for t, s in zip(rec_texts, rec_scores):
                        if s > 0.5:
                            texts.append(t)
                            scores.append(s)
                if not texts:
                    return None, 0.0
                return " ".join(texts), sum(scores) / len(scores)

            elif self._engine == "easy":
                results = self.ocr.readtext(img, detail=1)
                texts  = [r[1] for r in results if r[2] > 0.4]
                scores = [r[2] for r in results if r[2] > 0.4]
                if not texts:
                    return None, 0.0
                return " ".join(texts), sum(scores) / len(scores)
        except Exception as e:
            print(f"[PlateReader] OCR error: {e}")
        return None, 0.0

    @staticmethod
    def _clean(text: str) -> str:
        result = text.upper().strip()
        result = re.sub(r"[^A-Z0-9 ]", "", result)
        return result
