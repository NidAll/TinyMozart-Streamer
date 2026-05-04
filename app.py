from __future__ import annotations

import time

import streamlit as st

import streamer as streamer_module
import tinymozart_model as model_module


STREAMER_VERSION = getattr(streamer_module, "STREAMER_VERSION", "piano-renderer-v2")
StreamConfig = streamer_module.StreamConfig
TinyMozartStreamer = streamer_module.TinyMozartStreamer
QUALITY_PROFILES = getattr(
    model_module,
    "QUALITY_PROFILES",
    {
        "Fast": model_module.GenerationSettings(candidate_count=1),
        "Balanced": model_module.GenerationSettings(candidate_count=2),
        "High": model_module.GenerationSettings(candidate_count=4),
    },
)
settings_for_profile = getattr(
    model_module,
    "settings_for_profile",
    lambda profile: QUALITY_PROFILES.get(profile, QUALITY_PROFILES["Balanced"]),
)


st.set_page_config(page_title="TinyMozart Streamer", page_icon="🎹", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: #0f1115;
        color: #f2efe7;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        max-width: 760px;
        padding-top: 4.5rem;
    }
    .tm-shell {
        border: 1px solid rgba(242, 239, 231, 0.14);
        border-radius: 8px;
        padding: 28px;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.055), rgba(255,255,255,0.015)),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 28px);
    }
    .tm-title {
        font-family: Georgia, Cambria, serif;
        font-size: 42px;
        line-height: 1;
        margin: 0 0 8px;
        letter-spacing: 0;
    }
    .tm-subtitle {
        color: #bfb8a8;
        margin: 0 0 24px;
        font-size: 15px;
    }
    .tm-status {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 22px;
    }
    .tm-stat {
        border-top: 1px solid rgba(242, 239, 231, 0.18);
        padding-top: 10px;
    }
    .tm-label {
        color: #8f887c;
        font-size: 12px;
        text-transform: uppercase;
    }
    .tm-value {
        color: #f2efe7;
        font-size: 18px;
        margin-top: 2px;
    }
    .tm-wide {
        grid-column: 1 / -1;
    }
    .tm-metrics {
        color: #d8c48f;
        font-size: 15px;
        line-height: 1.4;
        margin-top: 2px;
        overflow-wrap: anywhere;
    }
    .stButton > button {
        height: 64px;
        width: 100%;
        border-radius: 6px;
        border: 1px solid #d8c48f;
        background: #d8c48f;
        color: #111;
        font-size: 20px;
        font-weight: 700;
    }
    .stButton > button:hover {
        border-color: #f0dfac;
        background: #f0dfac;
        color: #111;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


profile_names = list(QUALITY_PROFILES)
if "quality_mode" not in st.session_state:
    st.session_state.quality_mode = "Balanced"

existing_streamer = st.session_state.get("streamer")
existing_status = existing_streamer.snapshot() if existing_streamer else None
selected_mode = st.selectbox(
    "Quality mode",
    profile_names,
    index=profile_names.index(st.session_state.quality_mode),
    disabled=bool(existing_status and existing_status.running),
    label_visibility="collapsed",
)
st.session_state.quality_mode = selected_mode

streamer_key = f"{STREAMER_VERSION}:{selected_mode}"
if (
    "streamer" not in st.session_state
    or st.session_state.get("streamer_key") != streamer_key
):
    config = StreamConfig(
        settings=settings_for_profile(selected_mode),
        quality_mode=selected_mode,
    )
    st.session_state.streamer = TinyMozartStreamer(config)
    st.session_state.streamer_key = streamer_key

streamer: TinyMozartStreamer = st.session_state.streamer
status = streamer.snapshot()

st.markdown('<div class="tm-shell">', unsafe_allow_html=True)
st.markdown('<h1 class="tm-title">TinyMozart</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="tm-subtitle">Local unlimited piano generation from LH-Tech-AI/TinyMozart_v2_85M.</p>',
    unsafe_allow_html=True,
)

button_label = "Stop streaming" if status.running else "Start streaming"
if st.button(button_label, type="primary"):
    if status.running:
        streamer.stop()
    else:
        streamer.start()
    time.sleep(0.2)
    st.rerun()

status = streamer.snapshot()
runtime = "0:00"
if status.started_at and status.running:
    elapsed = int(time.time() - status.started_at)
    runtime = f"{elapsed // 60}:{elapsed % 60:02d}"

state = "Loading" if status.loading else ("Streaming" if status.running else "Stopped")
if status.error:
    state = "Error"

st.markdown(
    f"""
    <div class="tm-status">
      <div class="tm-stat"><div class="tm-label">State</div><div class="tm-value">{state}</div></div>
      <div class="tm-stat"><div class="tm-label">Device</div><div class="tm-value">{status.device}</div></div>
      <div class="tm-stat"><div class="tm-label">Runtime</div><div class="tm-value">{runtime}</div></div>
      <div class="tm-stat"><div class="tm-label">Pieces Ready</div><div class="tm-value">{status.chunks_generated}</div></div>
      <div class="tm-stat"><div class="tm-label">Pieces Played</div><div class="tm-value">{status.chunks_played}</div></div>
      <div class="tm-stat"><div class="tm-label">Status</div><div class="tm-value">{status.last_message}</div></div>
      <div class="tm-stat"><div class="tm-label">Best Score</div><div class="tm-value">{status.best_score:.1f}</div></div>
      <div class="tm-stat"><div class="tm-label">Rejected</div><div class="tm-value">{status.rejected_candidates}</div></div>
      <div class="tm-stat"><div class="tm-label">Mode</div><div class="tm-value">{selected_mode}</div></div>
      <div class="tm-stat"><div class="tm-label">Backend</div><div class="tm-value">{STREAMER_VERSION}</div></div>
      <div class="tm-stat tm-wide"><div class="tm-label">Quality Metrics</div><div class="tm-metrics">{status.last_metrics or "Waiting for candidate scores"}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

if status.error:
    st.error(status.error)

if status.running:
    time.sleep(1.0)
    st.rerun()
