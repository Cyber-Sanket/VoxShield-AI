"""
VoxShield AI - UI Components Module
Reusable presentation components, cybersecurity theme headers, and badges.
Ghost Protocol | AI & Cyber Security
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional

from config import (
    APP_NAME, APP_TAGLINE, TEAM_NAME, DOMAIN, VERSION,
    ASSETS_DIR, THEME
)
from modules.database import get_system_stats


def apply_custom_css():
    """Injects custom cyber dark CSS into Streamlit DOM."""
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline basic styles
        st.markdown(f"""
        <style>
        .stApp {{ background-color: {THEME['bg_dark']}; color: {THEME['text_main']}; }}
        </style>
        """, unsafe_allow_html=True)


def render_header(custom_subtitle: Optional[str] = None):
    """Renders the top cyber banner with official branding and system pulse."""
    subtitle = custom_subtitle or APP_TAGLINE
    st.markdown(f"""
    <div class="cyber-header">
        <div class="cyber-title">
            <span>🛡️</span>
            <span>{APP_NAME}</span>
        </div>
        <div class="cyber-tagline">{subtitle}</div>
        <div class="meta-badges">
            <span class="cyber-badge status-live">
                <span class="pulse-dot"></span> SYSTEM ACTIVE
            </span>
            <span class="cyber-badge">TEAM: {TEAM_NAME}</span>
            <span class="cyber-badge">DOMAIN: {DOMAIN}</span>
            <span class="cyber-badge">ENGINE v{VERSION}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_info(detector_status: Optional[Any] = None):
    """Renders common sidebar metadata and model status."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 10px 0; border-bottom: 1px solid rgba(30, 41, 59, 0.8); margin-bottom: 15px;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #00FFA3; font-family: 'Outfit', sans-serif;">
                🛡️ {APP_NAME}
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8;">AI Cyber Security Forensics</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"**Team:** {TEAM_NAME}")
        st.caption(f"**Domain:** {DOMAIN}")
        st.caption(f"**Target Sample Rate:** 16,000 Hz")

        st.markdown("---")
        st.markdown("#### 🔍 Detection Engine")
        
        if detector_status and detector_status.is_loaded:
            st.success(f"🟢 **Model Active**\n\n`{detector_status.model_name}`\n\n*Type:* {detector_status.model_type}")
        else:
            st.info("ℹ️ **Demo / Analysis Mode**\n\nNo trained weights in `models/`.\n*Automated classification disabled for academic integrity.*")

        st.markdown("---")
        st.markdown("#### ⚙️ Quick Actions")
        if st.button("🔄 Refresh Database & Engine", use_container_width=True):
            st.rerun()


def render_metric_cards(stats: Optional[Dict[str, int]] = None):
    """
    Renders the 4 live system counters.
    Counters strictly reflect the SQLite database, starting at zero.
    """
    if stats is None:
        stats = get_system_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="cyber-card">
            <div class="card-label">📁 Total Analyzed</div>
            <div class="card-value cyan">{stats.get('total_analyzed', 0)}</div>
            <div class="card-hint">Total audio recordings processed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="cyber-card">
            <div class="card-label">✅ Verified Real</div>
            <div class="card-value green">{stats.get('real_count', 0)}</div>
            <div class="card-hint">Model-classified authentic voices</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="cyber-card">
            <div class="card-label">⚠️ Synthetic / Fake</div>
            <div class="card-value red">{stats.get('fake_count', 0)}</div>
            <div class="card-hint">Model-detected synthetic voices</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="cyber-card">
            <div class="card-label">🔬 Demo Analyses</div>
            <div class="card-value blue">{stats.get('demo_count', 0)}</div>
            <div class="card-hint">Acoustic forensic inspections</div>
        </div>
        """, unsafe_allow_html=True)
