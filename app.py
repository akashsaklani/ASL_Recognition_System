import streamlit as st
import cv2
import mediapipe as mp
import pickle
import json
import os
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from collections import deque, Counter
from datetime import datetime
import threading
import pyttsx3
import queue

st.set_page_config(layout="wide", page_title="ASL Vision Pro", page_icon="🤟")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Full Overhaul
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Clash+Display:wght@400;500;600;700&family=Satoshi:wght@300;400;500;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bricolage+Grotesque:wght@400;500;600;700;800&display=swap');

:root {
    --bg:        #060a14;
    --bg2:       #0c1120;
    --bg3:       #111828;
    --bg4:       #161f30;
    --border:    #1a2438;
    --border2:   #243050;
    --border3:   #2d3d66;
    --accent:    #3d7fff;
    --accent2:   #6b4fff;
    --green:     #00e5a0;
    --green2:    #00b87d;
    --pink:      #ff3d7f;
    --yellow:    #ffd166;
    --cyan:      #00d4ff;
    --text:      #dce4ff;
    --text2:     #a8b8e0;
    --muted:     #5a6b98;
    --muted2:    #3d4f78;

    --font-display: 'Bricolage Grotesque', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --font-body:    'Space Grotesk', sans-serif;

    --glow-accent: 0 0 24px rgba(61,127,255,0.35);
    --glow-green:  0 0 24px rgba(0,229,160,0.35);
    --glow-pink:   0 0 24px rgba(255,61,127,0.35);
    --shadow-card: 0 8px 40px rgba(0,0,0,0.6);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
        linear-gradient(rgba(61,127,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(61,127,255,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    animation: grid_shift 20s linear infinite;
}
@keyframes grid_shift {
    0%   { background-position: 0 0; }
    100% { background-position: 48px 48px; }
}

/* Ambient glow orbs */
.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 60% 40% at 20% 10%, rgba(61,127,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 35% at 80% 85%, rgba(107,79,255,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 60% 40%, rgba(0,229,160,0.03) 0%, transparent 50%);
    pointer-events: none;
}

#MainMenu, footer { visibility: hidden; }
.block-container { padding: 80px 1.5rem 2rem !important; position: relative; z-index: 1; }

/* ── SIDEBAR ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #04060f 0%, var(--bg) 100%) !important;
    border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

.sb-header {
    padding: 24px 18px 20px;
    background: linear-gradient(135deg, rgba(61,127,255,0.08), transparent);
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.sb-brand-row { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.sb-icon {
    width: 46px; height: 46px; border-radius: 14px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: var(--glow-accent), inset 0 1px 0 rgba(255,255,255,0.15);
}
.sb-title {
    font-family: var(--font-display); font-size: 14px; font-weight: 700;
    color: #fff; letter-spacing: 0.2px;
}
.sb-sub { font-size: 9px; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px; }
.sb-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 99px;
    font-family: var(--font-mono); font-size: 9px; font-weight: 600; letter-spacing: 1px;
    background: rgba(61,127,255,0.12); color: var(--accent);
    border: 1px solid rgba(61,127,255,0.28);
}
.sb-badge::before { content: '●'; font-size: 7px; color: var(--green); }

.sb-nav-label {
    font-family: var(--font-mono); font-size: 9px; font-weight: 600;
    color: var(--muted2); letter-spacing: 2px; text-transform: uppercase;
    padding: 12px 18px 6px;
}

div[data-testid="stButton"] > button {
    background: transparent !important;
    color: var(--text2) !important;
    border: none !important; border-radius: 10px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important; font-size: 13px !important;
    padding: 10px 16px !important; text-align: left !important;
    transition: all 0.2s ease !important; width: 100% !important;
    letter-spacing: 0.2px !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(61,127,255,0.1) !important;
    color: var(--text) !important;
    transform: translateX(3px) !important;
}

.sb-footer {
    padding: 16px 18px; border-top: 1px solid var(--border);
    margin-top: 12px;
}
.sb-footer-text { font-size: 10px; color: var(--muted); line-height: 1.8; }

/* ── TOPBAR ───────────────────────────────────────────────────────────────── */
.topbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    height: 58px;
    background: rgba(6,10,20,0.88);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    padding: 0 1.5rem 0 300px;
    justify-content: space-between;
}
.topbar-brand { display: flex; align-items: center; gap: 12px; }
.topbar-icon {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center; font-size: 17px;
    box-shadow: var(--glow-accent);
}
.topbar-name {
    font-family: var(--font-display); font-size: 15px; font-weight: 700; color: #fff;
}
.topbar-sub { font-size: 10px; color: var(--muted); letter-spacing: 0.5px; }
.topbar-route {
    font-family: var(--font-mono); font-size: 10px; color: var(--muted);
    letter-spacing: 1px;
    background: rgba(255,255,255,0.04); border: 1px solid var(--border2);
    padding: 4px 12px; border-radius: 99px;
}

/* ── CARDS ────────────────────────────────────────────────────────────────── */
.card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 18px; padding: 24px;
    position: relative; overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
    box-shadow: var(--shadow-card);
}
.card::before {
    content: ''; position: absolute; inset: 0; border-radius: 18px;
    background: linear-gradient(135deg, rgba(61,127,255,0.03), transparent 55%);
    pointer-events: none;
}
.card:hover { border-color: var(--border3); }
.card.at  { border-top: 2px solid var(--accent); }
.card.gt  { border-top: 2px solid var(--green); }
.card.pt  { border-top: 2px solid var(--pink); }
.card.yt  { border-top: 2px solid var(--yellow); }

.card-label {
    font-family: var(--font-mono); font-size: 9px; font-weight: 600;
    letter-spacing: 2.5px; text-transform: uppercase; color: var(--muted);
    margin-bottom: 18px; display: flex; align-items: center; gap: 10px;
}
.card-label::before {
    content: ''; width: 18px; height: 1px; background: var(--accent);
}

/* ── HOME FEATURE CARDS ───────────────────────────────────────────────────── */
.feat-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 18px; padding: 30px 24px 24px;
    position: relative; overflow: hidden; cursor: pointer;
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1), border-color 0.3s, box-shadow 0.3s;
    min-height: 200px;
    box-shadow: var(--shadow-card);
}
.feat-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--green));
    transform: scaleX(0); transform-origin: left;
    transition: transform 0.35s ease;
}
.feat-card:hover::after { transform: scaleX(1); }
.feat-card:hover {
    transform: translateY(-6px) scale(1.01);
    border-color: var(--border3);
    box-shadow: 0 24px 64px rgba(0,0,0,0.5), var(--glow-accent);
}
.feat-icon { font-size: 42px; margin-bottom: 18px; display: block; }
.feat-title {
    font-family: var(--font-display); font-size: 15px; font-weight: 700;
    margin-bottom: 10px; letter-spacing: 0.2px;
}
.feat-desc { color: var(--text2); font-size: 12px; line-height: 1.75; }

/* ── QUICK START ──────────────────────────────────────────────────────────── */
.qs-row {
    display: flex; gap: 16px; align-items: flex-start;
    padding: 14px 0; border-bottom: 1px solid var(--border);
}
.qs-row:last-child { border-bottom: none; }
.qs-num {
    width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0;
    background: rgba(61,127,255,0.1); color: var(--accent);
    font-family: var(--font-mono); font-size: 11px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    border: 1px solid rgba(61,127,255,0.2);
}
.qs-txt { font-size: 13px; color: var(--text2); line-height: 1.65; }
.qs-txt b { color: var(--text); }

/* ── LIVE PAGE ────────────────────────────────────────────────────────────── */
.status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: var(--font-mono); font-size: 10px; font-weight: 600;
    padding: 6px 18px; border-radius: 99px; letter-spacing: 0.5px;
    margin-bottom: 14px;
}
.sp-live    { background: rgba(0,229,160,0.1); color: var(--green);  border: 1px solid rgba(0,229,160,0.25); }
.sp-offline { background: rgba(255,61,127,0.1); color: var(--pink);   border: 1px solid rgba(255,61,127,0.25); }
.sp-sync    { background: rgba(0,212,255,0.1);  color: var(--cyan);   border: 1px solid rgba(0,212,255,0.25); }

.pulse { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.p-green { background: var(--green); animation: pulse_a 1.5s ease-in-out infinite; }
.p-red   { background: var(--pink);  animation: pulse_a 1.5s ease-in-out infinite; }
.p-cyan  { background: var(--cyan);  animation: pulse_a 1.5s ease-in-out infinite; }
@keyframes pulse_a {
    0%,100% { opacity:1; box-shadow:0 0 0 0 currentColor; }
    50%     { opacity:.5; box-shadow:0 0 0 5px transparent; }
}

/* Prediction ring */
.pred-wrap { display: flex; flex-direction: column; align-items: center; padding: 16px 0 8px; }
.pred-ring {
    width: 170px; height: 170px; border-radius: 50%;
    background: radial-gradient(circle at 38% 32%, rgba(61,127,255,0.15), rgba(4,6,14,0.95) 68%);
    border: 1.5px solid var(--border3);
    display: flex; align-items: center; justify-content: center;
    position: relative; margin-bottom: 18px;
    box-shadow: 0 0 80px rgba(61,127,255,0.08), inset 0 0 60px rgba(0,0,0,0.6);
}
.pred-ring::before {
    content: ''; position: absolute; inset: -4px; border-radius: 50%;
    background: conic-gradient(var(--accent) var(--conf,0deg), transparent var(--conf,0deg));
    z-index: -1; transition: all 0.5s ease;
}
.pred-ring::after {
    content: ''; position: absolute; inset: -2px; border-radius: 50%;
    background: var(--bg2); z-index: -1;
}
.pred-letter {
    font-family: var(--font-display); font-size: 92px; font-weight: 800;
    line-height: 1; color: #fff;
    text-shadow: 0 0 40px rgba(61,127,255,0.7), 0 0 80px rgba(61,127,255,0.25);
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    letter-spacing: -3px;
}
.pred-letter.empty { color: var(--muted); font-size: 60px; letter-spacing: 0; }
.pred-conf {
    font-family: var(--font-mono); font-size: 24px; font-weight: 700;
    color: var(--green); text-shadow: var(--glow-green);
}
.pred-conf-lbl { font-size: 10px; color: var(--muted); margin-top: 4px; letter-spacing: 1px; }
.conf-bar { width: 100%; height: 4px; background: var(--border); border-radius: 99px; margin-top: 18px; overflow: hidden; }
.conf-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--green));
    box-shadow: 0 0 12px rgba(61,127,255,0.5);
    transition: width 0.4s cubic-bezier(0.4,0,0.2,1);
}

/* Sentence box */
.sent-box {
    background: var(--bg); border: 1px solid var(--border3);
    border-radius: 14px; padding: 18px 22px; min-height: 72px;
    position: relative; overflow: hidden; margin-bottom: 16px;
}
.sent-box::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--green));
    border-radius: 3px 0 0 3px;
}
.sent-text {
    font-family: var(--font-mono); font-size: 17px; letter-spacing: 5px;
    color: var(--text); word-break: break-all; line-height: 1.65;
}
.cursor {
    display: inline-block; width: 2px; height: 20px;
    background: var(--accent); margin-left: 3px; vertical-align: middle;
    animation: cur_blink 1s step-end infinite;
}
@keyframes cur_blink { 50% { opacity: 0; } }

/* Sign trail */
.trail { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 8px 0; }
.trail-chip {
    font-family: var(--font-display); font-size: 14px; font-weight: 700;
    width: 38px; height: 38px; border-radius: 10px;
    background: rgba(61,127,255,0.1); color: var(--accent);
    border: 1px solid rgba(61,127,255,0.22);
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s;
}
.trail-chip:last-child {
    background: rgba(0,229,160,0.12); color: var(--green);
    border-color: rgba(0,229,160,0.28);
    box-shadow: var(--glow-green);
    animation: chip_pop 0.35s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes chip_pop { 0% { transform: scale(0.5); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
.trail-arrow { color: var(--border3); font-size: 14px; }

/* Sync badge */
.sync-badge {
    font-family: var(--font-mono); font-size: 9px; letter-spacing: 1px;
    color: var(--cyan); background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2); padding: 4px 12px; border-radius: 99px;
    display: inline-block; margin-bottom: 12px;
}

/* ── PAGE HERO ────────────────────────────────────────────────────────────── */
.page-hero { margin-bottom: 28px; }
.page-hero h1 {
    font-family: var(--font-display); font-size: 30px; font-weight: 800;
    background: linear-gradient(135deg, #fff 40%, var(--accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.page-hero p { color: var(--text2); font-size: 14px; }

/* ── INSTRUCTION CARDS ────────────────────────────────────────────────────── */
.instr-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 16px; padding: 22px; margin-bottom: 14px;
    transition: border-color 0.25s, transform 0.25s;
}
.instr-card:hover { border-color: var(--border3); transform: translateX(4px); }
.instr-icon {
    width: 44px; height: 44px; border-radius: 11px; flex-shrink: 0;
    background: rgba(61,127,255,0.1); border: 1px solid rgba(61,127,255,0.2);
    display: flex; align-items: center; justify-content: center; font-size: 21px;
}
.instr-title {
    font-family: var(--font-display); font-size: 14px; font-weight: 700;
    color: #fff; margin-bottom: 5px;
}
.instr-desc { font-size: 12px; color: var(--text2); line-height: 1.75; }

/* ── GESTURE GRID ─────────────────────────────────────────────────────────── */
.gest-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 13px; padding: 18px 10px; text-align: center;
    transition: all 0.22s; margin-bottom: 10px;
}
.gest-card:hover {
    border-color: rgba(61,127,255,0.45);
    background: rgba(61,127,255,0.06);
    transform: scale(1.05) translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), var(--glow-accent);
}
.gest-emoji { font-size: 30px; display: block; margin-bottom: 8px; }
.gest-letter {
    font-family: var(--font-display); font-size: 22px; font-weight: 800;
    color: var(--accent); display: block; margin-bottom: 4px;
}
.gest-desc { font-size: 10px; color: var(--muted); }

/* ── HISTORY ──────────────────────────────────────────────────────────────── */
.hist-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 13px 0; border-bottom: 1px solid var(--border);
}
.hist-row:last-child { border-bottom: none; }
.hist-chip {
    font-family: var(--font-display); font-size: 18px; font-weight: 800;
    width: 46px; height: 46px; border-radius: 11px;
    background: rgba(61,127,255,0.1); border: 1px solid rgba(61,127,255,0.22);
    display: flex; align-items: center; justify-content: center;
    color: #fff;
}
.hist-word { font-family: var(--font-mono); font-size: 12px; color: var(--text); }
.hist-time { font-family: var(--font-mono); font-size: 10px; color: var(--muted); }

.stat-box {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 16px; padding: 28px; text-align: center; margin-bottom: 14px;
}
.stat-num {
    font-family: var(--font-display); font-size: 60px; font-weight: 800; line-height: 1;
    margin-bottom: 8px;
}
.stat-lbl {
    font-family: var(--font-mono); font-size: 9px; color: var(--muted);
    letter-spacing: 2px; text-transform: uppercase;
}

/* ── SETTINGS SLIDERS ─────────────────────────────────────────────────────── */
div[data-testid="stSlider"] label { color: var(--text) !important; font-size: 13px !important; }
div[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }
div[data-testid="stSlider"] > div > div { background: var(--border) !important; }

.param-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 14px 0; border-bottom: 1px solid var(--border);
}
.param-row:last-child { border-bottom: none; }
.param-name { font-weight: 600; font-size: 13px; color: var(--text); margin-bottom: 4px; }
.param-desc { font-size: 11px; color: var(--muted); line-height: 1.6; max-width: 220px; }

/* ── ABOUT ────────────────────────────────────────────────────────────────── */
.about-hero {
    background: linear-gradient(135deg, rgba(61,127,255,0.09), rgba(107,79,255,0.09));
    border: 1px solid rgba(61,127,255,0.18); border-radius: 18px;
    padding: 36px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.about-hero::before {
    content: '🤟'; position: absolute; right: 28px; top: 50%;
    transform: translateY(-50%); font-size: 110px; opacity: 0.06;
}
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.tag {
    padding: 5px 14px; border-radius: 99px; font-size: 11px; font-weight: 600;
    border: 1px solid; font-family: var(--font-mono);
}

/* ── SCROLLBAR ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ── MODEL & MEDIAPIPE ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model    = load_model()
mp_hands = mp.solutions.hands

# ── SYNC FILE ──────────────────────────────────────────────────────────────────
SYNC_FILE = "sentence.json"

def read_sync():
    try:
        with open(SYNC_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"sentence": "", "last_word": "", "event": "", "timestamp": 0}

# ── TTS (Streamlit side for END event from app.py cam) ────────────────────────
tts_queue  = queue.Queue()
_tts_ready = False

def _tts_worker():
    eng = pyttsx3.init()

    def apply_cfg(cfg):
        eng.setProperty("rate", cfg.get("rate", 150))
        voices = eng.getProperty("voices")
        gender = cfg.get("gender", "male")
        matched = [v for v in voices
                   if gender in (v.gender or "").lower()
                   or gender in v.name.lower()]
        if not matched:
            matched = [voices[1]] if gender == "female" and len(voices) > 1 else [voices[0]]
        eng.setProperty("voice", matched[0].id)

    apply_cfg({"rate": 150, "gender": "male"})   # defaults

    while True:
        item = tts_queue.get()
        if item is None:
            break
        if isinstance(item, tuple):
            kind, payload = item
            if kind == "__CONFIG__":
                apply_cfg(payload)
            elif kind == "__SPEAK__":
                try:
                    eng.stop()
                except Exception:
                    pass
                eng.say(payload)
                eng.runAndWait()
        else:
            # plain string fallback
            try:
                eng.say(item)
                eng.runAndWait()
            except Exception:
                pass

try:
    threading.Thread(target=_tts_worker, daemon=True).start()
    _tts_ready = True
except Exception:
    pass

def speak_st(text):
    if not _tts_ready or not text.strip():
        return
    # push config then text as tuple messages
    tts_queue.put(("__CONFIG__", {
        "rate":   st.session_state.get("tts_rate",   150),
        "gender": st.session_state.get("tts_gender", "male"),
    }))
    tts_queue.put(("__SPEAK__", text.strip()))

# ── SESSION STATE ──────────────────────────────────────────────────────────────
defaults = {
    "page":        "Home",
    "history_log": [],
    "sentence":    "",
    "conf_thresh": 0.40,
    "buf_size":    15,
    "agree_ratio": 0.65,
    "cooldown":    1.2,
    "space_delay": 1.0,
    "use_pre_sync": False,   # toggle: read from pre.py sync file
    "tts_gender":  "male",   # "male" | "female"
    "tts_mode":    "end",    # "end"  | "letter"
    "tts_rate":    150,      # words per minute
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── ASL PROCESSOR ─────────────────────────────────────────────────────────────
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock           = threading.Lock()
        self.current_pred    = "-"
        self.current_conf    = 0.0
        self.current_sentence= st.session_state.get("sentence", "")
        self.current_history = []
        self.buffer          = deque(maxlen=st.session_state.get("buf_size", 15))
        self.last_added      = ""
        self.last_time       = time.time()
        self.no_hand_start   = None
        self.hand_detected   = False
        self.new_word_ready  = False
        self.last_word       = ""
        self.CONF_THRESH     = st.session_state.get("conf_thresh",  0.40)
        self.BUF_SIZE        = st.session_state.get("buf_size",     15)
        self.AGREE           = st.session_state.get("agree_ratio",  0.65)
        self.COOLDOWN        = st.session_state.get("cooldown",     1.2)
        self.SPACE_DELAY     = st.session_state.get("space_delay",  1.0)
        self._frame_skip     = 0
        self._last_result    = None
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def get_state(self):
        with self._lock:
            return dict(
                pred=self.current_pred, conf=self.current_conf,
                sentence=self.current_sentence, history=list(self.current_history),
                hand=self.hand_detected, new_word=self.new_word_ready,
                last_word=self.last_word,
            )

    def consume_new_word(self):
        with self._lock:
            w = self.last_word; self.new_word_ready = False; return w

    def add_space(self):
        with self._lock: self.current_sentence += " "

    def delete_last(self):
        with self._lock:
            if self.current_sentence: self.current_sentence = self.current_sentence[:-1]

    def clear_all(self):
        with self._lock:
            self.current_sentence = ""; self.current_history = []; self.last_added = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self._frame_skip += 1
        if self._frame_skip % 2 == 0:
            result = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self._last_result = result
        else:
            result = self._last_result

        if result and result.multi_hand_landmarks:
            with self._lock:
                self.no_hand_start = None
                self.hand_detected = True

            for hl in result.multi_hand_landmarks:
                data  = []
                wrist = hl.landmark[0]
                for lm in hl.landmark:
                    data.extend([lm.x - wrist.x, lm.y - wrist.y])

                try:
                    pred = model.predict([data])[0]
                    conf = float(max(model.predict_proba([data])[0]))
                except Exception:
                    pred, conf = "-", 0.0

                with self._lock:
                    self.current_pred = str(pred)
                    self.current_conf = conf

                    if conf >= self.CONF_THRESH:
                        self.buffer.append(pred)

                    if len(self.buffer) == self.BUF_SIZE:
                        mc, cnt = Counter(self.buffer).most_common(1)[0]
                        if cnt / self.BUF_SIZE >= self.AGREE:
                            if mc != self.last_added and time.time() - self.last_time > self.COOLDOWN:
                                if mc == "SPACE":
                                    self.current_sentence += " "
                                elif mc == "DEL":
                                    if self.current_sentence:
                                        self.current_sentence = self.current_sentence[:-1]
                                elif mc == "END":
                                    speak_st(self.current_sentence)
                                else:
                                    self.current_sentence += str(mc)

                                self.last_added = mc
                                self.last_time  = time.time()
                                self.current_history.append(str(mc))
                                self.current_history = self.current_history[-6:]
                                self.last_word       = str(mc)
                                self.new_word_ready  = True

                                # Letter-by-letter TTS
                                if mc not in ("SPACE","DEL","END"):
                                    if st.session_state.get("tts_mode","end") == "letter":
                                        speak_st(str(mc))

                color = (0, 229, 160) if conf >= self.CONF_THRESH else (91, 124, 255)
                cv2.putText(img, f"{pred}  {int(conf*100)}%",
                            (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                h, w = img.shape[:2]
                xs = [lm.x for lm in hl.landmark]; ys = [lm.y for lm in hl.landmark]
                x1 = max(0, int(min(xs)*w)-20); y1 = max(0, int(min(ys)*h)-20)
                x2 = min(w, int(max(xs)*w)+20); y2 = min(h, int(max(ys)*h)+20)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,229,160), 2)
        else:
            with self._lock:
                self.buffer.clear()
                self.current_pred  = "-"
                self.current_conf  = 0.0
                self.hand_detected = False
                if self.no_hand_start is None:
                    self.no_hand_start = time.time()
                elif time.time() - self.no_hand_start >= self.SPACE_DELAY:
                    if self.current_sentence and self.current_sentence[-1] != " ":
                        self.current_sentence += " "
                        self.last_added = " "
                        self.current_history.append("_")
                        self.current_history = self.current_history[-6:]
                    self.no_hand_start = None

            overlay = img.copy()
            cv2.rectangle(overlay, (8,8), (280,54), (5,8,25), -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            cv2.putText(img, "No hand detected", (14,36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (100,100,255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
pages = [("🏠","Home"),("📷","Live Recognition"),("📖","Instructions"),
         ("🤚","Gesture Guide"),("🕐","History"),("⚙️","Settings"),("ℹ️","About")]

with st.sidebar:
    st.markdown("""
    <div class="sb-header">
        <div class="sb-brand-row">
            <div class="sb-icon">🤟</div>
            <div>
                <div class="sb-title">ASL Vision Pro</div>
                <div class="sb-sub">Recognition System</div>
            </div>
        </div>
        <span class="sb-badge">v2.1 PREMIUM</span>
    </div>
    <div class="sb-nav-label">Navigation</div>
    """, unsafe_allow_html=True)

    for icon, label in pages:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown("""
    <div class="sb-footer">
        <div class="sb-footer-text">
            Built with MediaPipe + ML<br>
            <span style="color:var(--accent)">ASL Vision Pro © 2025</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════════════
pg = st.session_state.page
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-icon">🤟</div>
        <div>
            <div class="topbar-name">ASL Vision Pro</div>
            <div class="topbar-sub">American Sign Language Recognition</div>
        </div>
    </div>
    <span class="topbar-route">/{pg.upper().replace(' ','_')}</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if pg == "Home":
    st.markdown("""
    <div class="page-hero">
        <h1>Welcome to ASL Vision Pro 👋</h1>
        <p>Real-time American Sign Language detection powered by MediaPipe + Machine Learning</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("""
        <div class="feat-card">
            <span class="feat-icon">🎥</span>
            <div class="feat-title" style="color:var(--accent)">Live Detection</div>
            <div class="feat-desc">Real-time hand gesture recognition from your webcam with instant visual feedback and confidence scoring.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Live Detection", key="home_live", use_container_width=True):
            st.session_state.page = "Live Recognition"; st.rerun()

    with c2:
        st.markdown("""
        <div class="feat-card">
            <span class="feat-icon">🧠</span>
            <div class="feat-title" style="color:var(--green)">ML Powered</div>
            <div class="feat-desc">Confidence-buffered classifier with majority voting for flicker-free, stable sign output.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Learn How It Works", key="home_ml", use_container_width=True):
            st.session_state.page = "About"; st.rerun()

    with c3:
        st.markdown("""
        <div class="feat-card">
            <span class="feat-icon">📝</span>
            <div class="feat-title" style="color:var(--pink)">Text + Speech</div>
            <div class="feat-desc">Detected ASL signs assembled into readable sentences — with END gesture to speak them aloud.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("View History", key="home_history", use_container_width=True):
            st.session_state.page = "History"; st.rerun()

    st.markdown("""
    <div class="card" style="margin-top:10px">
        <div class="card-label">Quick Start Guide</div>
        <div class="qs-row"><div class="qs-num">01</div><div class="qs-txt">Click <b>Live Recognition</b> in the sidebar to open the camera view</div></div>
        <div class="qs-row"><div class="qs-num">02</div><div class="qs-txt">Allow camera access and click <b>START</b> to begin</div></div>
        <div class="qs-row"><div class="qs-num">03</div><div class="qs-txt">Show ASL hand signs — <b>hold each sign for 1–2 seconds</b></div></div>
        <div class="qs-row"><div class="qs-num">04</div><div class="qs-txt">Hide your hand for 1 second to insert a <b>space</b></div></div>
        <div class="qs-row"><div class="qs-num">05</div><div class="qs-txt">Use <b>END gesture</b> to speak the sentence aloud via TTS</div></div>
        <div class="qs-row"><div class="qs-num">06</div><div class="qs-txt">View your full session log in the <b>History</b> page</div></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "Live Recognition":
    # ── Sync mode toggle ──────────────────────────────────────────────────────
    use_sync = st.toggle(
        "🔄 Sync mode — read from pre.py (OpenCV window)",
        value=st.session_state.use_pre_sync,
        help="Enable if you're running pre.py separately and want the sentence to mirror here."
    )
    st.session_state.use_pre_sync = use_sync

    if use_sync:
        st.markdown('<span class="sync-badge">● SYNCING FROM pre.py / sentence.json</span>', unsafe_allow_html=True)

    col_vid, col_info = st.columns([55, 45], gap="large")

    with col_vid:
        st.markdown('<div class="card-label" style="margin-bottom:12px">🎥 Live Camera Feed</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="asl_live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ASLProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

    # ── State collection ──────────────────────────────────────────────────────
    state = {}
    if use_sync:
        # Read from sentence.json written by pre.py
        sync = read_sync()
        state = {
            "pred":      "—",
            "conf":      0.0,
            "sentence":  sync.get("sentence", ""),
            "history":   [],
            "hand":      False,
            "new_word":  False,
            "last_word": sync.get("last_word", ""),
        }
        # Log events from pre.py
        ev = sync.get("event", "")
        lw = sync.get("last_word", "")
        ts = sync.get("timestamp", 0)
        if lw and ev == "add":
            already = st.session_state.history_log[0]["word"] if st.session_state.history_log else None
            if lw != already:
                st.session_state.history_log.insert(0, {
                    "word": lw, "time": datetime.now().strftime("%I:%M:%S %p")
                })
                st.session_state.history_log = st.session_state.history_log[:100]
        st.session_state.sentence = state["sentence"]

    elif ctx and ctx.video_processor:
        try:
            state = ctx.video_processor.get_state()
            if state.get("new_word"):
                word = ctx.video_processor.consume_new_word()
                if word and word.strip():
                    st.session_state.history_log.insert(0, {
                        "word": word, "time": datetime.now().strftime("%I:%M:%S %p")
                    })
                    st.session_state.history_log = st.session_state.history_log[:100]
            st.session_state.sentence = state.get("sentence", "")
        except Exception:
            pass

    # ── Right panel ───────────────────────────────────────────────────────────
    with col_info:
        pred      = state.get("pred", "-")
        conf      = state.get("conf", 0.0)
        sentence  = state.get("sentence", st.session_state.sentence)
        hist      = state.get("history", [])
        hand_ok   = state.get("hand", False)

        conf_pct  = int(conf * 100)
        ring_deg  = f"{int(conf * 360)}deg"
        is_pred   = pred not in ["-","—","",None]
        pred_cls  = "pred-letter" if is_pred else "pred-letter empty"
        pred_disp = str(pred) if is_pred else "—"
        conf_color= "#00e5a0" if conf>=0.6 else ("#ffd166" if conf>=0.4 else "#ff3d7f")

        if use_sync:
            st.markdown('<span class="status-pill sp-sync"><span class="pulse p-cyan"></span>SYNCED FROM pre.py</span>', unsafe_allow_html=True)
        elif hand_ok:
            st.markdown('<span class="status-pill sp-live"><span class="pulse p-green"></span>HAND DETECTED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill sp-offline"><span class="pulse p-red"></span>NO HAND — Show your hand</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card gt" style="margin-bottom:14px">
            <div class="card-label">Current Prediction</div>
            <div class="pred-wrap">
                <div class="pred-ring" style="--conf:{ring_deg}">
                    <div class="{pred_cls}">{pred_disp}</div>
                </div>
                <div class="pred-conf" style="color:{conf_color}">{conf_pct}%</div>
                <div class="pred-conf-lbl">CONFIDENCE SCORE</div>
                <div class="conf-bar"><div class="conf-fill" style="width:{conf_pct}%"></div></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="card-label">Current Sentence</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sent-box">
            <div class="sent-text">{sentence if sentence else '<span style="color:var(--muted)">Waiting for signs...</span>'}
                <span class="cursor"></span>
            </div>
        </div>""", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("⎵ Space", key="sp", use_container_width=True):
                if ctx and ctx.video_processor:
                    try: ctx.video_processor.add_space()
                    except: pass
        with b2:
            if st.button("⌫ Delete", key="dl", use_container_width=True):
                if ctx and ctx.video_processor:
                    try: ctx.video_processor.delete_last()
                    except: pass
        with b3:
            if st.button("🔊 Speak", key="spk", use_container_width=True):
                if sentence.strip():
                    speak_st(sentence.strip())
        with b4:
            if st.button("🗑 Clear", key="cl", use_container_width=True):
                if ctx and ctx.video_processor:
                    try: ctx.video_processor.clear_all()
                    except: pass
                st.session_state.sentence = ""

        if hist:
            st.markdown('<div class="card-label" style="margin-top:14px">Recent Signs</div>', unsafe_allow_html=True)
            chips = ""
            for i, h in enumerate(hist):
                if i: chips += '<span class="trail-arrow">›</span>'
                chips += f'<div class="trail-chip">{h}</div>'
            st.markdown(f'<div class="trail">{chips}</div>', unsafe_allow_html=True)

    if ctx and ctx.state.playing:
        time.sleep(0.5)
        st.rerun()
    elif use_sync:
        time.sleep(0.6)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "Instructions":
    st.markdown('<div class="page-hero"><h1>📖 Instructions</h1><p>Follow these tips for the best detection accuracy.</p></div>', unsafe_allow_html=True)
    steps = [
        ("💡","Good Lighting",        "Ensure your hand is well-lit. Natural daylight or a lamp in front of you works best. Avoid backlighting."),
        ("👤","Hand Fully in Frame",   "Keep your entire hand visible. Partial or cropped hands reduce landmark detection accuracy significantly."),
        ("📏","Optimal Distance",      "Stay 1–2 feet from the camera. Too close distorts proportions; too far loses landmark precision."),
        ("✋","Hold Signs Steady",     "Hold each sign still for 1–2 seconds. 15 consistent frames are needed to confirm a detection."),
        ("🔄","Face Palm Forward",     "Point your palm toward the camera. Side angles may not match the model's training orientation."),
        ("☝️","One Hand Only",         "This system is trained for single-hand ASL. Keep the other hand out of frame to avoid interference."),
        ("🔊","END Gesture = Speak",   "Show the END gesture after completing a sentence to have it spoken aloud via text-to-speech."),
        ("⌫","DEL Gesture = Delete",  "Show the DEL gesture to remove the last character from the current sentence."),
    ]
    c_a, c_b = st.columns(2, gap="large")
    for i, (icon, title, desc) in enumerate(steps):
        with (c_a if i%2==0 else c_b):
            st.markdown(f"""
            <div class="instr-card">
                <div style="display:flex;gap:16px;align-items:flex-start">
                    <div class="instr-icon">{icon}</div>
                    <div>
                        <div class="instr-title">Step {i+1}: {title}</div>
                        <div class="instr-desc">{desc}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# GESTURE GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "Gesture Guide":
    st.markdown('<div class="page-hero"><h1>🤚 Gesture Guide</h1><p>Supported ASL signs — hold each sign steady for reliable detection.</p></div>', unsafe_allow_html=True)
    gestures = [
        ("1️⃣","1","Index up"),    ("2️⃣","2","Peace sign"),
        ("3️⃣","3","Three up"),    ("4️⃣","4","Four up"),
        ("🖐️","5","Open palm"),   ("🤙","6","Hang loose"),
        ("👌","A","Fist+thumb"),  ("🤜","B","Flat up"),
        ("🤌","C","Curved"),      ("🤏","D","Circle"),
        ("✊","E","Bent"),        ("🤞","F","Cross"),
        ("🤟","G","Gun point"),   ("🤘","H","Two side"),
        ("☝️","I","Pinky up"),   ("🫵","K","V-shape"),
        ("🫷","L","L-shape"),    ("✌️","V","Victory"),
        ("👋","W","Three wide"), ("🤙","Y","Pinky-thumb"),
        ("🤚","END","Speak all"), ("🖐","DEL","Delete last"),
    ]
    cols = st.columns(5, gap="small")
    for i, (emoji, label, desc) in enumerate(gestures):
        with cols[i % 5]:
            border = "border:1px solid rgba(0,229,160,0.35)" if label in ("END","DEL") else ""
            st.markdown(f"""
            <div class="gest-card" style="{border}">
                <span class="gest-emoji">{emoji}</span>
                <span class="gest-letter" style="{'color:var(--green)' if label in ('END','DEL') else ''}">{label}</span>
                <span class="gest-desc">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card at" style="margin-top:16px">
        <div class="card-label">Important Notes</div>
        <div style="color:var(--text2);font-size:13px;line-height:2.3">
            • <b style="color:#fff">J</b> and <b style="color:#fff">Z</b> are motion-based — not supported in this version<br>
            • <b style="color:var(--green)">END</b> gesture: speak the full sentence aloud via TTS<br>
            • <b style="color:var(--yellow)">DEL</b> gesture: remove the last character<br>
            • <b style="color:#fff">SPACE</b>: hide your hand for 1 second to insert a space<br>
            • Accuracy improves in good lighting with hand centred in frame
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "History":
    st.markdown('<div class="page-hero"><h1>🕐 Detection History</h1><p>All signs detected in this session.</p></div>', unsafe_allow_html=True)
    col_h, col_s = st.columns([3,1], gap="large")

    with col_h:
        if not st.session_state.history_log:
            st.markdown("""
            <div class="card" style="text-align:center;padding:60px 20px">
                <div style="font-size:56px;margin-bottom:16px">🕐</div>
                <div style="font-family:var(--font-display);font-size:16px;font-weight:700;color:#fff;margin-bottom:8px">No detections yet</div>
                <div style="font-size:13px;color:var(--muted)">Go to <b style="color:var(--accent)">Live Recognition</b> and start signing!</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for e in st.session_state.history_log:
                st.markdown(f"""
                <div class="hist-row">
                    <div style="display:flex;align-items:center;gap:14px">
                        <div class="hist-chip">{e['word']}</div>
                        <div class="hist-word">{e['word']}</div>
                    </div>
                    <div class="hist-time">{e['time']}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.history_log = []; st.rerun()

    with col_s:
        total  = len(st.session_state.history_log)
        unique = len(set(e["word"] for e in st.session_state.history_log))
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num" style="color:var(--accent)">{total}</div>
            <div class="stat-lbl">Total Signs</div>
        </div>
        <div class="stat-box">
            <div class="stat-num" style="color:var(--green)">{unique}</div>
            <div class="stat-lbl">Unique Signs</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "Settings":
    st.markdown('<div class="page-hero"><h1>⚙️ Settings</h1><p>Fine-tune detection and voice parameters.</p></div>', unsafe_allow_html=True)

    cs1, cs2 = st.columns(2, gap="large")

    # ── Detection params ──────────────────────────────────────────────────────
    with cs1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Detection Parameters</div>', unsafe_allow_html=True)
        conf_thresh = st.slider("Confidence Threshold",  0.10, 0.90, st.session_state.conf_thresh, 0.05)
        buf_size    = st.slider("Buffer Size (frames)",  5,    30,   st.session_state.buf_size,    1)
        agree_ratio = st.slider("Agreement Ratio",       0.40, 0.95, st.session_state.agree_ratio, 0.05)
        cooldown    = st.slider("Cooldown (seconds)",    0.5,  3.0,  st.session_state.cooldown,    0.1)
        space_delay = st.slider("Space Delay (seconds)", 0.5,  3.0,  st.session_state.space_delay, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("💾 Apply Detection Settings", use_container_width=True):
            st.session_state.conf_thresh  = conf_thresh
            st.session_state.buf_size     = buf_size
            st.session_state.agree_ratio  = agree_ratio
            st.session_state.cooldown     = cooldown
            st.session_state.space_delay  = space_delay
            st.success("✅ Detection settings saved! Restart the camera on Live page to apply.")

    with cs2:
        st.markdown("""
        <div class="card at">
            <div class="card-label">Parameter Guide</div>
            <div class="param-row"><div><div class="param-name">Confidence Threshold</div><div class="param-desc">Lower = more sensitive. Higher = more precise but slower.</div></div></div>
            <div class="param-row"><div><div class="param-name">Buffer Size</div><div class="param-desc">More frames = stable but slower response. Fewer = faster but shaky.</div></div></div>
            <div class="param-row"><div><div class="param-name">Agreement Ratio</div><div class="param-desc">% of buffer frames that must agree on the same sign.</div></div></div>
            <div class="param-row"><div><div class="param-name">Cooldown</div><div class="param-desc">Min gap (seconds) to prevent the same letter repeating too fast.</div></div></div>
            <div class="param-row"><div><div class="param-name">Space Delay</div><div class="param-desc">Hide hand for this many seconds to auto-insert a space.</div></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TTS / Voice settings ──────────────────────────────────────────────────
    tv1, tv2 = st.columns(2, gap="large")

    with tv1:
        st.markdown('<div class="card gt">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">🎙️ Voice / TTS Settings</div>', unsafe_allow_html=True)

        # Gender toggle
        gender_idx = 0 if st.session_state.tts_gender == "male" else 1
        tts_gender = st.radio(
            "Voice Gender",
            options=["male", "female"],
            index=gender_idx,
            horizontal=True,
            help="Switch between male and female TTS voice."
        )

        # Speak mode toggle
        mode_idx = 0 if st.session_state.tts_mode == "end" else 1
        tts_mode = st.radio(
            "Speak Mode",
            options=["end", "letter"],
            index=mode_idx,
            horizontal=True,
            help="END = speak full sentence on END gesture. LETTER = speak each letter as it's added."
        )

        # Speed slider
        tts_rate = st.slider(
            "Speech Speed (wpm)", 80, 260,
            st.session_state.tts_rate, 10,
            help="Words per minute. 150 = normal speed."
        )

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🎙️ Apply Voice Settings", use_container_width=True, key="apply_tts"):
            st.session_state.tts_gender = tts_gender
            st.session_state.tts_mode   = tts_mode
            st.session_state.tts_rate   = tts_rate
            # Test speak
            speak_st(f"Voice set to {tts_gender} at {tts_rate} words per minute.")
            st.success(f"✅ Voice: {tts_gender.upper()} | Mode: {tts_mode.upper()} | Rate: {tts_rate} wpm")

    with tv2:
        # Live TTS preview
        g_color = "#c87bff" if st.session_state.tts_gender == "female" else "#5bc8ff"
        m_color = "#00e5a0" if st.session_state.tts_mode == "end" else "#ffd166"
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Current Voice Config</div>
            <div class="param-row">
                <div>
                    <div class="param-name">Voice Gender</div>
                    <div class="param-desc">Selected voice type for TTS output</div>
                </div>
                <div style="font-family:var(--font-mono);font-size:14px;font-weight:700;color:{g_color}">
                    {st.session_state.tts_gender.upper()}
                </div>
            </div>
            <div class="param-row">
                <div>
                    <div class="param-name">Speak Mode</div>
                    <div class="param-desc">END = full sentence · LETTER = each letter aloud</div>
                </div>
                <div style="font-family:var(--font-mono);font-size:14px;font-weight:700;color:{m_color}">
                    {st.session_state.tts_mode.upper()}
                </div>
            </div>
            <div class="param-row">
                <div>
                    <div class="param-name">Speech Rate</div>
                    <div class="param-desc">Words per minute (80 slow → 260 fast)</div>
                </div>
                <div style="font-family:var(--font-mono);font-size:14px;font-weight:700;color:var(--text)">
                    {st.session_state.tts_rate} wpm
                </div>
            </div>
            <div class="param-row">
                <div>
                    <div class="param-name">pre.py Keyboard</div>
                    <div class="param-desc">G = gender toggle · M = mode toggle · +/- = speed</div>
                </div>
                <div style="font-family:var(--font-mono);font-size:11px;color:var(--yellow)">G / M / ± </div>
            </div>
        </div>

        <div class="card" style="margin-top:14px">
            <div class="card-label">Test Voice</div>
            <div style="font-size:12px;color:var(--text2);line-height:1.8">
                Click <b style="color:var(--green)">Apply Voice Settings</b> to hear a preview.<br>
                Or use the <b style="color:var(--accent)">🔊 Speak</b> button on the Live page.<br>
                In pre.py: press <b style="color:var(--yellow)">[G]</b> to toggle gender,
                <b style="color:var(--yellow)">[M]</b> to toggle mode,
                <b style="color:var(--yellow)">[+/-]</b> to change speed.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "About":
    st.markdown('<div class="page-hero"><h1>ℹ️ About</h1><p>Technology stack and model details.</p></div>', unsafe_allow_html=True)
    ca1, ca2 = st.columns([2,1], gap="large")
    with ca1:
        st.markdown("""
        <div class="about-hero">
            <div style="font-family:var(--font-display);font-size:20px;font-weight:800;color:#fff;margin-bottom:14px">
                ASL Vision Pro — Sign Language Recognition
            </div>
            <div style="color:var(--text2);font-size:13px;line-height:2.1">
                This system uses <b style="color:#fff">Machine Learning</b> and <b style="color:#fff">Computer Vision</b>
                to recognize American Sign Language (ASL) gestures in real-time and convert them to text.<br><br>
                <b style="color:var(--accent)">MediaPipe</b> extracts 21 hand landmarks per frame. Coordinates are normalized
                relative to the wrist, then fed into a trained classifier. A 15-frame confidence buffer with majority voting
                ensures stable, flicker-free output.<br><br>
                The <b style="color:var(--green)">END gesture</b> speaks the assembled sentence aloud via TTS.
                The <b style="color:var(--yellow)">DEL gesture</b> removes the last character.
                Sentences sync from the OpenCV window to Streamlit via <code style="color:var(--cyan)">sentence.json</code>.<br><br>
                Built to bridge communication gaps for the hearing-impaired community.
            </div>
            <div class="tag-row">
                <span class="tag" style="color:#3b82f6;border-color:#3b82f640;background:#3b82f610">Python</span>
                <span class="tag" style="color:#f59e0b;border-color:#f59e0b40;background:#f59e0b10">OpenCV</span>
                <span class="tag" style="color:#ef4444;border-color:#ef444440;background:#ef444410">Scikit-learn</span>
                <span class="tag" style="color:#a78bfa;border-color:#a78bfa40;background:#a78bfa10">MediaPipe</span>
                <span class="tag" style="color:#f97316;border-color:#f9731640;background:#f9731610">Streamlit</span>
                <span class="tag" style="color:#06b6d4;border-color:#06b6d440;background:#06b6d410">WebRTC</span>
                <span class="tag" style="color:#00e5a0;border-color:#00e5a040;background:#00e5a010">pyttsx3</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with ca2:
        st.markdown("""
        <div class="card at">
            <div class="card-label">Model Info</div>
            <div class="param-row"><div><div class="param-name">Input Features</div><div class="param-desc">42 (21 landmarks × x, y)</div></div></div>
            <div class="param-row"><div><div class="param-name">Normalization</div><div class="param-desc">Wrist-relative coordinates</div></div></div>
            <div class="param-row"><div><div class="param-name">Classifier</div><div class="param-desc">Random Forest / SVM</div></div></div>
            <div class="param-row"><div><div class="param-name">Output Classes</div><div class="param-desc">A–Z, 0–9, SPACE, END, DEL</div></div></div>
            <div class="param-row"><div><div class="param-name">Buffer Logic</div><div class="param-desc">15-frame majority vote</div></div></div>
            <div class="param-row"><div><div class="param-name">Min Confidence</div><div class="param-desc">40% (adjustable)</div></div></div>
            <div class="param-row"><div><div class="param-name">Sync Method</div><div class="param-desc">sentence.json (pre.py ↔ app.py)</div></div></div>
        </div>""", unsafe_allow_html=True)