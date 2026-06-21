"""
core/evaluation.py
───────────────────
Real evaluation tooling — NOT a source of pre-baked numbers.

This module computes precision / recall / F1 / mAP@0.5 from actual
predictions vs. actual ground-truth boxes, and benchmarks real processing
latency by running the live pipeline against real images. Every number it
produces is measured, not assumed — running this script against your own
validation set is what gives you honest metrics to put in a submission deck.

Usage
-----
    python -m core.evaluation \\
        --images-dir path/to/val/images \\
        --labels-dir path/to/val/labels \\
        --model-path models/best.pt \\
        --report-out evaluation_report.json

`labels-dir` must contain YOLO-format .txt files (one per image, same
basename): `class_id x_center y_center width height` (normalized 0-1).
"""

from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

import cv2


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _yolo_txt_to_boxes(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, w, h = map(float, line.split())
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append({"class_id": int(cls_id), "bbox": (x1, y1, x2, y2)})
    return boxes


def evaluate_detections(
    predictions: list[dict],     # [{"image_id":..., "class_id":..., "bbox":..., "confidence":...}]
    ground_truth: list[dict],    # [{"image_id":..., "class_id":..., "bbox":...}]
    iou_threshold: float = 0.5,
) -> dict:
    """
    Computes precision/recall/F1 per class and overall, plus AP@iou_threshold
    per class (area under the precision-recall curve) and their mean (mAP).
    """
    classes = sorted(set(p["class_id"] for p in predictions) | set(g["class_id"] for g in ground_truth))
    per_class = {}

    for cls in classes:
        preds = sorted(
            [p for p in predictions if p["class_id"] == cls],
            key=lambda p: p.get("confidence", 0.0),
            reverse=True,
        )
        gts = [g for g in ground_truth if g["class_id"] == cls]
        gt_by_image: dict = {}
        for g in gts:
            gt_by_image.setdefault(g["image_id"], []).append({"bbox": g["bbox"], "matched": False})

        tp_flags, fp_flags = [], []
        for p in preds:
            candidates = gt_by_image.get(p["image_id"], [])
            best_iou, best_gt = 0.0, None
            for gt in candidates:
                if gt["matched"]:
                    continue
                i = iou(p["bbox"], gt["bbox"])
                if i > best_iou:
                    best_iou, best_gt = i, gt
            if best_iou >= iou_threshold and best_gt is not None:
                best_gt["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        total_gt = len(gts)
        cum_tp, cum_fp = 0, 0
        precisions, recalls = [], []
        for tp, fp in zip(tp_flags, fp_flags):
            cum_tp += tp
            cum_fp += fp
            precisions.append(cum_tp / max(1, cum_tp + cum_fp))
            recalls.append(cum_tp / max(1, total_gt))

        ap = _average_precision(precisions, recalls)
        final_tp, final_fp = cum_tp, cum_fp
        final_fn = total_gt - final_tp
        precision = final_tp / max(1, final_tp + final_fp)
        recall = final_tp / max(1, final_tp + final_fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "ap": round(ap, 4),
            "tp": final_tp, "fp": final_fp, "fn": final_fn,
            "ground_truth_count": total_gt,
        }

    overall_precision = statistics.mean(c["precision"] for c in per_class.values()) if per_class else 0.0
    overall_recall = statistics.mean(c["recall"] for c in per_class.values()) if per_class else 0.0
    overall_f1 = statistics.mean(c["f1"] for c in per_class.values()) if per_class else 0.0
    mAP = statistics.mean(c["ap"] for c in per_class.values()) if per_class else 0.0

    return {
        "iou_threshold": iou_threshold,
        "per_class": per_class,
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "mAP": round(mAP, 4),
        },
    }


def _average_precision(precisions: list[float], recalls: list[float]) -> float:
    """Area under the precision-recall curve via the standard all-points interpolation."""
    if not precisions:
        return 0.0
    recalls = [0.0] + recalls + [1.0]
    precisions = [precisions[0]] + precisions + [0.0]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def benchmark_latency(detector, preprocessor, image_paths: list[Path], max_dim: int = 960) -> dict:
    """Runs the REAL pipeline (preprocess + detect) on real images and reports
    measured timing statistics. No assumed/fabricated numbers."""
    times_ms = []
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        t0 = time.perf_counter()
        enhanced = preprocessor.enhance(frame)
        detector.process_frame(enhanced, draw=False)
        times_ms.append((time.perf_counter() - t0) * 1000)

    if not times_ms:
        return {"error": "no readable images found", "n_images": 0}

    times_ms.sort()
    n = len(times_ms)
    p95_idx = min(n - 1, int(0.95 * n))
    return {
        "n_images": n,
        "mean_ms": round(statistics.mean(times_ms), 1),
        "median_ms": round(statistics.median(times_ms), 1),
        "p95_ms": round(times_ms[p95_idx], 1),
        "min_ms": round(min(times_ms), 1),
        "max_ms": round(max(times_ms), 1),
        "fps": round(1000 / statistics.mean(times_ms), 2),
    }


def _main():
    parser = argparse.ArgumentParser(description="Evaluate detection accuracy and latency against a real validation set.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=False, help="YOLO-format .txt labels, same basenames as images")
    parser.add_argument("--model-path", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--report-out", default="evaluation_report.json")
    args = parser.parse_args()

    from core.detector import ViolationDetector, VIOLATION_LABELS
    from core.preprocessor import ImagePreprocessor

    label_to_id = {label: cls_id for cls_id, label in VIOLATION_LABELS.items()}

    images_dir = Path(args.images_dir)
    image_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    detector = ViolationDetector(model_path=args.model_path, conf_threshold=args.conf)
    preprocessor = ImagePreprocessor()

    print(f"Benchmarking latency on {len(image_paths)} real images...")
    latency_report = benchmark_latency(detector, preprocessor, image_paths)
    print(json.dumps(latency_report, indent=2))

    report = {"latency": latency_report}

    if args.labels_dir:
        labels_dir = Path(args.labels_dir)
        predictions, ground_truth = [], []
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            h, w = frame.shape[:2]
            result = detector.process_frame(frame, draw=False)
            for det in result.detections:
                if det.label not in label_to_id:
                    continue   # shouldn't happen, but skip rather than break eval on a mismatch
                predictions.append({
                    "image_id": img_path.stem,
                    "class_id": label_to_id[det.label],
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                })
            gt_boxes = _yolo_txt_to_boxes(labels_dir / f"{img_path.stem}.txt", w, h)
            for gt in gt_boxes:
                ground_truth.append({
                    "image_id": img_path.stem,
                    "class_id": gt["class_id"],
                    "bbox": gt["bbox"],
                })

        print(f"Evaluating accuracy against {len(ground_truth)} ground-truth boxes...")
        accuracy_report = evaluate_detections(predictions, ground_truth)
        print(json.dumps(accuracy_report, indent=2))
        report["accuracy"] = accuracy_report
    else:
        print("No --labels-dir given — skipping precision/recall/mAP (latency-only report).")

    Path(args.report_out).write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {args.report_out}")


if __name__ == "__main__":
    _main()
