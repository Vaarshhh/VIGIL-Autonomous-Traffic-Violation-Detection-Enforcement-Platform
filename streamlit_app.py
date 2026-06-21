"""
streamlit_app.py
─────────────────
Analytics dashboard: live stats, search by plate/date/type, evidence viewer,
trend charts. Talks to the FastAPI backend over HTTP rather than re-loading
YOLO/PaddleOCR itself — keeps this process light and avoids running the
models twice on the same laptop.

Run alongside the API (in a second terminal):
    streamlit run streamlit_app.py
"""

import os
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="VIGIL Analytics", page_icon="🛑", layout="wide")


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API_BASE}{path} — is `uvicorn main:app` running? ({e})")
        return None


st.title("🛑 VIGIL — Violation Analytics")

health = api_get("/health")
if health:
    cols = st.columns(4)
    cols[0].metric("Frames processed", health.get("frames_processed", 0))
    cols[1].metric("OCR engine", health.get("ocr_engine", "—"))
    active = health.get("active_violation_classes", [])
    cols[2].metric("Active violation classes", len(active))
    cols[3].metric("Mode", "DEMO (stock weights)" if health.get("demo_mode") else "LIVE MODEL")
    if health.get("demo_mode"):
        st.warning(
            f"Running in demo mode — no violation classes are trained/active yet. "
            f"Pending: {', '.join(health.get('pending_violation_classes', [])) or 'none listed'}. "
            f"This is intentional: the system reports zero violations rather than guessing."
        )

st.divider()

tab_overview, tab_search, tab_upload, tab_video = st.tabs(
    ["📊 Overview", "🔍 Search & Evidence", "📤 Upload & Analyze", "🎥 Video & Wrong-Side"]
)

# ── OVERVIEW TAB ──────────────────────────────────────────────────────────
with tab_overview:
    summary = api_get("/analytics/summary")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total violations recorded", summary.get("total_violations", 0))
        c2.metric("Total fines (₹)", f"{summary.get('total_fine_inr', 0):,}")
        c3.metric("Unique plates seen", summary.get("unique_plates", 0))
        with c4:
            st.write("")  # vertical alignment spacer
            if st.button("📄 Generate PDF Report"):
                try:
                    resp = requests.get(f"{API_BASE}/analytics/report", timeout=30)
                    resp.raise_for_status()
                    st.download_button(
                        "Download violation_report.pdf",
                        resp.content,
                        file_name="violation_report.pdf",
                        mime="application/pdf",
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not generate report: {e}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("By violation type")
            by_type = summary.get("by_type", {})
            if by_type:
                st.bar_chart(pd.Series(by_type, name="count"))
            else:
                st.info("No violations recorded yet — analyze a frame in the Upload tab to populate this.")
        with col_b:
            st.subheader("By risk level")
            by_risk = summary.get("by_risk", {})
            if by_risk:
                st.bar_chart(pd.Series(by_risk, name="count"))
            else:
                st.info("No violations recorded yet.")

    st.subheader("Daily trend")
    days = st.slider("Window (days)", 7, 90, 30)
    trend = api_get("/analytics/trend", days=days)
    if trend and trend.get("trend"):
        df = pd.DataFrame(trend["trend"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        st.line_chart(df[["count"]])
        st.caption("Daily fine total (₹)")
        st.line_chart(df[["fine"]])
    else:
        st.info("No trend data yet — once violations are recorded, daily counts will chart here.")

# ── SEARCH TAB ────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Search violation history")
    c1, c2, c3, c4 = st.columns(4)
    plate_q = c1.text_input("Plate number contains")
    date_from = c2.date_input("From", value=date.today() - timedelta(days=30))
    date_to = c3.date_input("To", value=date.today())
    vtype_q = c4.text_input("Violation type (exact)")

    if st.button("Search", type="primary"):
        results = api_get(
            "/violations/search",
            plate=plate_q or None,
            date_from=str(date_from),
            date_to=str(date_to),
            violation_type=vtype_q or None,
        )
        if results:
            st.session_state["search_results"] = results.get("results", [])

    rows = st.session_state.get("search_results", [])
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df[["challan_id", "timestamp", "violation_type", "plate_number", "confidence", "severity", "fine_inr", "risk_level"]],
            use_container_width=True,
        )

        st.subheader("Evidence viewer")
        challan_ids = sorted(set(df["challan_id"]))
        selected = st.selectbox("Select a challan to view evidence", challan_ids)
        if selected:
            img_col, meta_col = st.columns([2, 1])
            with img_col:
                try:
                    resp = requests.get(f"{API_BASE}/evidence/{selected}", timeout=10)
                    if resp.ok:
                        st.image(resp.content, caption=f"Evidence — {selected}", use_container_width=True)
                        st.download_button("Download evidence image", resp.content, file_name=f"{selected}.jpg", mime="image/jpeg")
                    else:
                        st.info("No evidence image stored for this record.")
                except requests.exceptions.RequestException:
                    st.error("Could not fetch evidence image from the API.")
            with meta_col:
                meta = api_get(f"/evidence/{selected}/metadata")
                if meta:
                    st.json(meta)
    else:
        st.caption("Run a search above to see results here.")

# ── UPLOAD TAB ────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a frame for analysis")
    uploaded = st.file_uploader("Traffic camera frame (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded and st.button("Analyze", type="primary"):
        with st.spinner("Running preprocess → detect → OCR → challan..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze/image",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                data = None

        if data:
            import base64
            if data.get("annotated_image"):
                st.image(base64.b64decode(data["annotated_image"]), use_container_width=True)
            st.metric("Processing time", f"{data['result']['proc_ms']} ms")
            st.metric("Violations found", data["result"]["violation_count"])
            if data.get("challan"):
                st.success(f"Challan generated: {data['challan']['challan_id']} — ₹{data['challan']['total_fine']:,}")
            else:
                st.info("No violation generated for this frame.")
            with st.expander("Full response JSON"):
                st.json(data)

# ── VIDEO / WRONG-SIDE TAB ────────────────────────────────────────────────
with tab_video:
    st.subheader("Wrong-side driving — video analysis")
    st.caption(
        "A single photo can't show direction of travel, so wrong-side driving is detected here by "
        "tracking each vehicle's heading across a short clip and comparing it to the direction traffic "
        "*should* be moving in this camera's view. That expected direction can't be inferred from pixels "
        "— it has to be set correctly below, or every result will be meaningless."
    )

    c1, c2, c3 = st.columns(3)
    expected_dir = c1.number_input(
        "Expected direction (degrees)", min_value=0, max_value=359, value=0, step=15,
        help="0 = left-to-right, 90 = top-to-bottom, 180 = right-to-left, 270 = bottom-to-top, "
             "as seen in the video frame. Set this to match real traffic flow for this camera.",
    )
    deviation_threshold = c2.slider(
        "Flag threshold (degrees off expected)", min_value=30, max_value=180, value=100, step=10,
        help="How far a vehicle's heading must deviate from the expected direction before it's flagged.",
    )
    frame_skip = c3.slider("Frame skip", min_value=1, max_value=15, value=5, help="Process every Nth frame — higher is faster but less precise.")

    uploaded_video = st.file_uploader("Traffic camera clip (MP4/AVI/MOV)", type=["mp4", "avi", "mov", "webm"])

    if uploaded_video and st.button("Analyze Video", type="primary"):
        with st.spinner("Tracking vehicle headings across the clip... this can take a while for longer videos"):
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze/video",
                    params={
                        "frame_skip": frame_skip,
                        "expected_direction_deg": expected_dir,
                        "deviation_threshold_deg": deviation_threshold,
                    },
                    files={"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)},
                    timeout=300,
                )
                resp.raise_for_status()
                video_data = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                video_data = None

        if video_data:
            m1, m2, m3 = st.columns(3)
            m1.metric("Frames processed", video_data["frames_processed"])
            m2.metric("Wrong-side events", len(video_data["wrong_side_events"]))
            m3.metric("Expected direction used", f"{video_data['expected_direction_deg']}°")

            events = video_data.get("wrong_side_events", [])
            if events:
                st.dataframe(pd.DataFrame(events), use_container_width=True)
                st.success(f"{len(events)} vehicle(s) flagged — check Search & Evidence tab, violation type 'wrong_side', for saved evidence frames.")
            else:
                st.info("No wrong-side events detected in this clip with the current calibration.")
            st.caption(video_data.get("note", ""))
