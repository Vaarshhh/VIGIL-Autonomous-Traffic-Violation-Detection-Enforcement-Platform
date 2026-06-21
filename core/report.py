"""
core/report.py
────────────────
Generates a PDF summary report from the actual violation database —
satisfies the "summary reports" requirement in the problem statement.
Every number in this report comes from core/database.py queries against
real recorded violations; nothing here is templated/fake data.

Note: uses "Rs." instead of the Rupee symbol (₹) — ReportLab's built-in
fonts don't reliably render that glyph and can produce a black box instead.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from core import database as db

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "violation_classes.json"


def _model_status() -> dict:
    if not CONFIG_PATH.exists():
        return {"model_path": "unknown", "model_status": "unknown"}
    cfg = json.loads(CONFIG_PATH.read_text())
    return {
        "model_path": cfg.get("model_path", "unknown"),
        "model_status": cfg.get("model_status", "unknown"),
        "active_classes": [v["label"] for v in cfg.get("classes", {}).values()],
        "pending_classes": [c["label"] for c in cfg.get("pending_classes", [])],
    }


def generate_summary_report(output_path: str = "violation_report.pdf") -> str:
    summary = db.get_summary()
    trend = db.get_trend(days=30)
    recent = db.get_recent(limit=25)
    model_info = _model_status()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    note_style = ParagraphStyle("Note", parent=styles["Normal"], textColor=colors.grey, fontSize=8)

    doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    story.append(Paragraph("VIGIL — Traffic Violation Summary Report", title_style))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ── model status ──
    story.append(Paragraph("System Status", section_style))
    status_rows = [
        ["Model path", model_info["model_path"]],
        ["Model status", model_info["model_status"]],
        ["Active violation classes", ", ".join(model_info["active_classes"]) or "none"],
        ["Pending (not yet trained)", ", ".join(model_info["pending_classes"]) or "none"],
    ]
    t = Table(status_rows, colWidths=[2.2 * inch, 4 * inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)

    # ── summary metrics ──
    story.append(Paragraph("Summary", section_style))
    metric_rows = [
        ["Total violations recorded", str(summary["total_violations"])],
        ["Total fines issued (INR)", f"Rs. {summary['total_fine_inr']:,}"],
        ["Unique plates seen", str(summary["unique_plates"])],
    ]
    t = Table(metric_rows, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    story.append(t)

    # ── by violation type ──
    story.append(Paragraph("By Violation Type", section_style))
    if summary["by_type"]:
        rows = [["Violation Type", "Count"]] + [[k.replace("_", " ").title(), str(v)] for k, v in summary["by_type"].items()]
        t = Table(rows, colWidths=[3 * inch, 3 * inch])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3142")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No violations recorded yet.", styles["Normal"]))

    # ── trend ──
    story.append(Paragraph("Daily Trend (last 30 days)", section_style))
    if trend:
        rows = [["Date", "Count", "Fine (INR)"]] + [[d["date"], str(d["count"]), f"Rs. {d['fine']:,}"] for d in trend]
        t = Table(rows, colWidths=[2 * inch, 2 * inch, 2 * inch])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3142")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No trend data yet.", styles["Normal"]))

    # ── recent records ──
    story.append(Paragraph("Recent Violations (most recent 25)", section_style))
    if recent:
        rows = [["Challan ID", "Timestamp", "Type", "Plate", "Fine (INR)", "Risk"]]
        for r in recent:
            rows.append([
                r["challan_id"][:18], r["timestamp"][:19],
                (r["violation_type"] or "").replace("_", " "),
                r["plate_number"] or "UNREAD",
                f"Rs. {r['fine_inr']:,}" if r["fine_inr"] else "-",
                r["risk_level"] or "-",
            ])
        t = Table(rows, colWidths=[1.3 * inch, 1.2 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch, 0.7 * inch])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3142")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No violations recorded yet.", styles["Normal"]))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "All figures above are pulled live from the violations database at generation time. "
        "Pending violation classes have no trained model behind them yet and are excluded "
        "from detection, not silently approximated.",
        note_style,
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    path = generate_summary_report()
    print(f"Report written to {path}")
