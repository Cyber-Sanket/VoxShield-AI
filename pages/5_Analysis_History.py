"""
VoxShield AI - Analysis History & Auditing Page
SQLite Database Records, Search & Filter, CSV Export, and Record Management.
Ghost Protocol | AI & Cyber Security
"""

import json
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Analysis History | VoxShield AI",
    page_icon="🛡️",
    layout="wide"
)

from config import MODE_MODEL, MODE_DEMO, LABEL_REAL, LABEL_FAKE
from modules.database import (
    init_db, get_records_dataframe, get_record_by_id,
    delete_record, clear_all_history, get_system_stats
)
from modules.ui_components import apply_custom_css, render_header, render_sidebar_info

init_db()
apply_custom_css()
render_sidebar_info(st.session_state.get("detector", None).model_status if "detector" in st.session_state else None)
render_header("Forensic Analysis History & Cryptographic Audit Log")

# Metric Summary
stats = get_system_stats()
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Records Stored", stats["total_analyzed"])
with col_m2:
    st.metric("Verified Real Audio", stats["real_count"])
with col_m3:
    st.metric("Flagged Synthetic Deepfakes", stats["fake_count"])
with col_m4:
    st.metric("Demo Analyses", stats["demo_count"])

st.markdown("---")

# Search and Filter Toolbar
st.markdown("### 🔍 Search & Filter Records")
col_s1, col_s2, col_s3 = st.columns([2, 1.2, 1.2])

with col_s1:
    search_query = st.text_input("Search by Filename, Analysis ID, or Model Name:", placeholder="e.g. sample_real or VX-2026...")

with col_s2:
    mode_filter = st.selectbox("Filter by Detection Mode:", ["All", MODE_MODEL, MODE_DEMO])

with col_s3:
    prediction_filter = st.selectbox("Filter by Prediction:", ["All", LABEL_REAL, LABEL_FAKE, "Inconclusive", "None / Unavailable"])

# Fetch Data
df = get_records_dataframe(
    search_query=search_query,
    mode_filter=mode_filter,
    prediction_filter=prediction_filter
)

st.markdown(f"**Found {len(df)} matching records**")

if df.empty:
    st.info("No records match the selected search and filter criteria.")
else:
    # Display interactive dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("Audit ID", width="medium"),
            "filename": st.column_config.TextColumn("Filename", width="medium"),
            "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "duration": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
            "detection_mode": st.column_config.TextColumn("Mode"),
            "prediction": st.column_config.TextColumn("Verdict"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
            "model_name": st.column_config.TextColumn("Model"),
            "sample_rate": st.column_config.NumberColumn("Rate (Hz)"),
            "channels": st.column_config.NumberColumn("Ch")
        }
    )

    # Action Toolbar: CSV Export and Record Details
    col_act1, col_act2 = st.columns([1.5, 1.5])
    
    with col_act1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered History to CSV",
            data=csv_data,
            file_name="voxshield_analysis_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_act2:
        # Inspect Record JSON
        selected_id = st.selectbox("Select Record ID to Inspect Full Acoustic Summary:", df["id"].tolist())
        if st.button("🔎 View Acoustic Feature Summary"):
            rec = get_record_by_id(selected_id)
            if rec and rec.get("feature_summary"):
                st.json(rec["feature_summary"])
            else:
                st.info("No extended acoustic feature summary stored for this record.")

st.markdown("---")

# Delete & Maintenance Section
st.markdown("### 🗑️ Audit Trail Maintenance")
col_del1, col_del2 = st.columns(2)

with col_del1:
    with st.expander("Delete Single Record"):
        if not df.empty:
            del_id = st.selectbox("Select ID to delete:", df["id"].tolist(), key="del_single_select")
            if st.button("⚠️ Delete Selected Record", type="secondary"):
                if delete_record(del_id):
                    st.success(f"Record '{del_id}' deleted successfully.")
                    st.rerun()
                else:
                    st.error("Failed to delete record.")
        else:
            st.caption("No records available to delete.")

with col_del2:
    with st.expander("Danger Zone: Clear All History"):
        st.warning("This action permanently deletes all analysis records from the SQLite database.")
        confirm = st.checkbox("I confirm I want to clear all forensic records.")
        if st.button("🚨 Clear Entire Database History", disabled=not confirm):
            cleared_count = clear_all_history()
            st.success(f"Cleared {cleared_count} records. Database is now empty.")
            st.rerun()
