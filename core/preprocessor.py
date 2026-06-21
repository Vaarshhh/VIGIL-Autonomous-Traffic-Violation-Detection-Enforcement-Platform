"""
core/preprocessor.py
────────────────────
Adaptive image enhancement pipeline.
This is a DIFFERENTIATOR — most teams skip preprocessing entirely.
Handles: low-light, rain streaks, motion blur, fog/haze.
"""

from __future__ import annotations
import cv2
import numpy as np


class ImagePreprocessor:
    """
    Auto-detects image condition and applies the right enhancement.

    Usage
    -----
    prep   = ImagePreprocessor()
    clean  = prep.enhance(raw_bgr_frame)
    report = prep.last_report   # what was applied
    """

    def __init__(self, target_brightness: int = 110):
        self.target_brightness = target_brightness
        self.last_report: dict = {}

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Main entry — auto-detects and applies all needed fixes."""
        report: dict[str, bool] = {}

        # 1. low-light correction
        brightness = self._mean_brightness(frame)
        if brightness < 60:
            frame = self._clahe_enhance(frame)
            frame = self._gamma_correct(frame, gamma=1.6)
            report["low_light_fix"] = True
        elif brightness < 90:
            frame = self._clahe_enhance(frame)
            report["mild_enhancement"] = True

        # 2. blur / motion-blur detection + sharpen
        if self._is_blurry(frame, threshold=80):
            frame = self._unsharp_mask(frame)
            report["sharpened"] = True

        # 3. rain streak removal (vertical streak filter)
        if self._has_rain_streaks(frame):
            frame = self._remove_rain(frame)
            report["rain_removed"] = True

        # 4. haze / fog removal (dark channel prior — lightweight version)
        if self._is_hazy(frame):
            frame = self._dehaze(frame)
            report["dehazed"] = True

        # 5. always normalise colour balance
        frame = self._white_balance(frame)

        self.last_report = report
        return frame

    # ── condition detectors ───────────────────────────────────────────────────

    @staticmethod
    def _mean_brightness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def _is_blurry(frame: np.ndarray, threshold: float = 80) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    @staticmethod
    def _has_rain_streaks(frame: np.ndarray) -> bool:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # vertical Sobel — rain streaks have strong vertical gradient variance
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ratio  = np.std(sobelx) / (np.std(sobely) + 1e-6)
        return ratio > 2.2

    @staticmethod
    def _is_hazy(frame: np.ndarray) -> bool:
        # hazy images have low contrast in dark channel
        dark = np.min(frame, axis=2)
        return float(np.mean(dark)) > 100

    # ── enhancement methods ───────────────────────────────────────────────────

    @staticmethod
    def _clahe_enhance(frame: np.ndarray) -> np.ndarray:
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _gamma_correct(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        inv   = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)]
        ).astype("uint8")
        return cv2.LUT(frame, table)

    @staticmethod
    def _unsharp_mask(frame: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
        return cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)

    @staticmethod
    def _remove_rain(frame: np.ndarray) -> np.ndarray:
        # guided filter approximation using bilateral + morphological opening
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 7, 7, 7, 21)
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        opened   = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        return cv2.addWeighted(denoised, 0.7, opened, 0.3, 0)

    @staticmethod
    def _dehaze(frame: np.ndarray) -> np.ndarray:
        # lightweight version of dark channel prior
        dark      = np.min(frame, axis=2).astype(np.float32)
        atm_light = np.percentile(frame, 99)
        t         = 1.0 - 0.9 * (dark / (atm_light + 1e-6))
        t         = np.clip(t, 0.15, 1.0)[:, :, np.newaxis]
        f         = frame.astype(np.float32)
        result    = (f - atm_light) / t + atm_light
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _white_balance(frame: np.ndarray) -> np.ndarray:
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a  = np.average(result[:, :, 1])
        avg_b  = np.average(result[:, :, 2])
        result[:, :, 1] -= (avg_a - 128) * 0.5
        result[:, :, 2] -= (avg_b - 128) * 0.5
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)