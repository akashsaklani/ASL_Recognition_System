import streamlit as st
import cv2
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from collections import deque, Counter
import time
from datetime import datetime
import threading

st.set_page_config(layout="wide", page_title="ASL Vision Pro", page_icon="🤟")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0f1f;
    --bg2:      #11172a;
    --bg3:      #1a2035;
    --border:   #1f2745;
    --border2:  #2a3560;
    --accent:   #4f7cff;
    --accent2:  #7c5cff;
    --green:    #00d4a0;
    --pink:     #ff4f87;
    --text:     #e6e9ff;
    --muted:    #7a86b6;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
        linear-gradient(rgba(91,124,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(91,124,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
}

#MainMenu, footer{ visibility: hidden; }
.block-container {
    padding: 80px 1.5rem 2rem !important;
}

.topbar {
    background: rgba(10,15,31,0.85);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    box-shadow: none;
}
.topbar-brand {
    display: flex; align-items: center; gap: 14px;
}
.topbar-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 19px;
    box-shadow: 0 0 20px rgba(91,124,255,0.4);
}
.topbar-name {
    font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 800;
    color: #fff; letter-spacing: 0.5px;
}
.topbar-sub { font-size: 10px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
.topbar-status {
    display: flex; align-items: center; gap: 24px;
}
.status-pill {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; font-weight: 600; padding: 6px 16px;
    border-radius: 99px; letter-spacing: 0.5px;
}
.status-pill.live { background: rgba(0,229,160,0.1); color: var(--green); border: 1px solid rgba(0,229,160,0.25); }
.status-pill.offline { background: rgba(255,77,148,0.1); color: var(--pink); border: 1px solid rgba(255,77,148,0.25); }
.pulse { width: 7px; height: 7px; border-radius: 50%; }
.pulse.green { background: var(--green); animation: pulse_anim 1.5s ease-in-out infinite; }
.pulse.red   { background: var(--pink);  animation: pulse_anim 1.5s ease-in-out infinite; }
@keyframes pulse_anim {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 currentColor; }
    50% { opacity: 0.5; box-shadow: 0 0 0 4px transparent; }
}
.page-indicator {
    font-family: 'Space Mono', monospace;
    font-size: 10px; color: var(--muted); letter-spacing: 1px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #05071a 0%, var(--bg) 100%) !important;
    border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
    overflow-y: auto !important;
    height: auto !important;
}

.sb-header {
    padding: 28px 20px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
    background: linear-gradient(135deg, rgba(91,124,255,0.05), transparent);
}
.sb-logo-row { display: flex; align-items: center; gap: 13px; margin-bottom: 18px; }
.sb-logo-box {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: 0 4px 20px rgba(91,124,255,0.35);
}
.sb-logo-title { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 800; color: #fff; }
.sb-logo-sub { font-size: 9px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 2px; }
.sb-version-badge {
    display: inline-block; padding: 3px 10px; border-radius: 99px;
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    background: rgba(91,124,255,0.15); color: var(--accent); border: 1px solid rgba(91,124,255,0.3);
}

.sb-section-label {
    font-size: 9px; font-weight: 700; color: var(--muted); letter-spacing: 2px;
    text-transform: uppercase; padding: 8px 20px 4px;
}

div[data-testid="stButton"] > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 13px !important;
    padding: 10px 16px !important;
    text-align: left !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(91,124,255,0.1) !important;
    color: var(--text) !important;
}

.sb-footer {
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    background: rgba(3,5,13,0.8);
    margin-top: 12px;
}
.sb-footer-text { font-size: 10px; color: var(--muted); line-height: 1.6; }

.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.card::before {
    content: '';
    position: absolute; inset: 0; border-radius: 16px;
    background: linear-gradient(135deg, rgba(91,124,255,0.03), transparent 60%);
    pointer-events: none;
}
.card:hover { border-color: var(--border2); }
.card.accent-top { border-top: 2px solid var(--accent); }
.card.green-top  { border-top: 2px solid var(--green); }
.card.pink-top   { border-top: 2px solid var(--pink); }

.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
}
.card-label::before {
    content: ''; width: 16px; height: 1px;
    background: var(--accent); display: inline-block;
}

.feature-card-wrap {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 16px; padding: 28px 22px;
    transition: all 0.3s; position: relative; overflow: hidden;
    min-height: 190px; cursor: pointer;
}
.feature-card-wrap::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--green));
    transform: scaleX(0); transform-origin: left;
    transition: transform 0.3s;
}
.feature-card-wrap:hover::before { transform: scaleX(1); }
.feature-card-wrap:hover {
    border-color: var(--border2);
    transform: translateY(-3px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}
.feature-icon { font-size: 44px; margin-bottom: 16px; display: block; }
.feature-title {
    font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 700;
    color: #fff; margin-bottom: 8px; letter-spacing: 0.3px;
}
.feature-desc { color: var(--muted); font-size: 12px; line-height: 1.7; }

.pred-container {
    display: flex; flex-direction: column; align-items: center;
    padding: 20px 0 10px;
    position: relative;
}
.pred-ring {
    width: 160px; height: 160px; border-radius: 50%;
    background: radial-gradient(circle at 40% 35%, rgba(91,124,255,0.12), rgba(3,5,13,0.9) 70%);
    border: 2px solid var(--border2);
    display: flex; align-items: center; justify-content: center;
    position: relative; margin-bottom: 16px;
    box-shadow: 0 0 60px rgba(91,124,255,0.1), inset 0 0 40px rgba(0,0,0,0.5);
}
.pred-ring::before {
    content: '';
    position: absolute; inset: -4px; border-radius: 50%;
    background: conic-gradient(var(--accent) var(--conf), transparent var(--conf));
    z-index: -1;
    transition: all 0.5s ease;
}
.pred-ring::after {
    content: '';
    position: absolute; inset: -2px; border-radius: 50%;
    background: var(--bg2);
    z-index: -1;
}
.pred-letter {
    font-family: 'Syne', sans-serif;
    font-size: 88px; font-weight: 800; line-height: 1;
    color: #fff;
    text-shadow:
        0 0 40px rgba(91,124,255,0.6),
        0 0 80px rgba(91,124,255,0.2);
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    letter-spacing: -2px;
}
.pred-letter.no-pred { color: var(--muted); font-size: 60px; }
.pred-conf-badge {
    font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700;
    color: var(--green);
    text-shadow: 0 0 20px rgba(0,229,160,0.4);
}
.pred-conf-label { font-size: 11px; color: var(--muted); margin-top: 4px; }
.conf-track {
    width: 100%; height: 4px; background: var(--border);
    border-radius: 99px; margin-top: 16px; overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--green));
    transition: width 0.4s cubic-bezier(0.4,0,0.2,1);
    box-shadow: 0 0 10px rgba(91,124,255,0.5);
}

.sentence-outer {
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 16px 20px;
    min-height: 68px;
    position: relative;
    overflow: hidden;
}
.sentence-outer::before {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--green));
    border-radius: 3px 0 0 3px;
}
.sentence-text {
    font-family: 'Space Mono', monospace;
    font-size: 18px; letter-spacing: 6px; color: var(--text);
    word-break: break-all; line-height: 1.6;
}
.sentence-cursor {
    display: inline-block; width: 2px; height: 22px;
    background: var(--accent); margin-left: 2px; vertical-align: middle;
    animation: blink_cur 1s step-end infinite;
}
@keyframes blink_cur { 50% { opacity: 0; } }

.page-hero { margin-bottom: 28px; }
.page-hero h1 {
    font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800;
    color: #fff; margin-bottom: 6px;
    background: linear-gradient(135deg, #fff 40%, var(--accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-hero p { color: var(--muted); font-size: 14px; }

.qs-step {
    display: flex; align-items: flex-start; gap: 16px;
    padding: 14px 0; border-bottom: 1px solid var(--border);
}
.qs-step:last-child { border-bottom: none; }
.qs-num {
    width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
    background: rgba(91,124,255,0.12); color: var(--accent);
    font-family: 'Space Mono', monospace; font-size: 12px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    border: 1px solid rgba(91,124,255,0.2);
}
.qs-content { font-size: 13px; color: var(--muted); line-height: 1.6; }
.qs-content b { color: var(--text); }

.stat-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 14px; padding: 24px; text-align: center;
    margin-bottom: 14px;
}
.stat-num {
    font-family: 'Syne', sans-serif; font-size: 56px; font-weight: 800;
    line-height: 1; margin-bottom: 8px;
}
.stat-label {
    font-family: 'Space Mono', monospace; font-size: 9px;
    color: var(--muted); letter-spacing: 2px; text-transform: uppercase;
}

.hist-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 0; border-bottom: 1px solid var(--border);
}
.hist-item:last-child { border-bottom: none; }
.hist-sign {
    font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 800;
    color: #fff; width: 48px; height: 48px; border-radius: 12px;
    background: rgba(91,124,255,0.1); border: 1px solid rgba(91,124,255,0.2);
    display: flex; align-items: center; justify-content: center;
}
.hist-time { font-family: 'Space Mono', monospace; font-size: 10px; color: var(--muted); }

.g-grid-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 10px; text-align: center;
    transition: all 0.2s; cursor: default; margin-bottom: 10px;
}
.g-grid-card:hover {
    border-color: rgba(91,124,255,0.4);
    background: rgba(91,124,255,0.06);
    transform: scale(1.04);
}
.g-emoji-big { font-size: 32px; display: block; margin-bottom: 8px; }
.g-letter {
    font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800;
    color: var(--accent); display: block; margin-bottom: 4px;
}
.g-desc { font-size: 10px; color: var(--muted); }

div[data-testid="stSlider"] label { color: var(--text) !important; font-size: 13px !important; }
div[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }
div[data-testid="stSlider"] > div > div { background: var(--border) !important; }

.param-item {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 14px 0; border-bottom: 1px solid var(--border);
}
.param-item:last-child { border-bottom: none; }
.param-name { font-weight: 600; font-size: 13px; color: var(--text); margin-bottom: 4px; }
.param-desc { font-size: 11px; color: var(--muted); line-height: 1.5; max-width: 220px; }

.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.tag {
    padding: 5px 14px; border-radius: 99px; font-size: 11px; font-weight: 600;
    border: 1px solid; font-family: 'Space Mono', monospace;
}

.about-hero {
    background: linear-gradient(135deg, rgba(91,124,255,0.1), rgba(123,94,167,0.1));
    border: 1px solid rgba(91,124,255,0.2); border-radius: 16px;
    padding: 36px; margin-bottom: 20px;
    position: relative; overflow: hidden;
}
.about-hero::before {
    content: '🤟';
    position: absolute; right: 30px; top: 50%; transform: translateY(-50%);
    font-size: 100px; opacity: 0.08;
}

.sign-trail {
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 10px 0;
}
.sign-chip {
    font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700;
    width: 38px; height: 38px; border-radius: 10px;
    background: rgba(91,124,255,0.12); color: var(--accent);
    border: 1px solid rgba(91,124,255,0.25);
    display: flex; align-items: center; justify-content: center;
}
.sign-chip:last-child {
    background: rgba(0,229,160,0.15); color: var(--green);
    border-color: rgba(0,229,160,0.3);
    box-shadow: 0 0 15px rgba(0,229,160,0.15);
}
.sign-arrow { color: var(--border2); font-size: 14px; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ── MODEL LOAD ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()
mp_hands = mp.solutions.hands

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "page": "Home",
    "history_log": [],
    "sentence": "",
    "conf_thresh": 0.40,
    "buf_size": 15,
    "agree_ratio": 0.65,
    "cooldown": 1.2,
    "space_delay": 1.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── ASL PROCESSOR ─────────────────────────────────────────────────────────────
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock            = threading.Lock()
        self.current_pred     = "-"
        self.current_conf     = 0.0
        self.current_sentence = st.session_state.get("sentence", "")
        self.current_history  = []
        self.buffer           = deque(maxlen=st.session_state.get("buf_size", 15))
        self.last_added       = ""
        self.last_time        = time.time()
        self.no_hand_start    = None
        self.hand_detected    = False
        self.new_word_ready   = False
        self.last_word        = ""
        self.CONFIDENCE_THRESHOLD = st.session_state.get("conf_thresh",  0.40)
        self.BUFFER_SIZE          = st.session_state.get("buf_size",     15)
        self.AGREEMENT_RATIO      = st.session_state.get("agree_ratio",  0.65)
        self.COOLDOWN             = st.session_state.get("cooldown",     1.2)
        self.SPACE_DELAY          = st.session_state.get("space_delay",  1.0)
        self.frame_skip           = 0
        self.last_result          = None
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_state(self):
        with self._lock:
            return {
                "pred":      self.current_pred,
                "conf":      self.current_conf,
                "sentence":  self.current_sentence,
                "history":   list(self.current_history),
                "hand":      self.hand_detected,
                "new_word":  self.new_word_ready,
                "last_word": self.last_word,
            }

    def consume_new_word(self):
        with self._lock:
            word = self.last_word
            self.new_word_ready = False
            return word

    def add_space(self):
        with self._lock:
            self.current_sentence += " "

    def delete_last(self):
        with self._lock:
            if self.current_sentence:
                self.current_sentence = self.current_sentence[:-1]

    def clear_all(self):
        with self._lock:
            self.current_sentence = ""
            self.current_history  = []
            self.last_added       = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        self.frame_skip += 1
        if self.frame_skip % 2 == 0:
            frame_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result       = self.hands.process(frame_rgb)
            self.last_result = result
        else:
            result = self.last_result

        if result and result.multi_hand_landmarks:
            with self._lock:
                self.no_hand_start = None
                self.hand_detected = True

            for hand_landmarks in result.multi_hand_landmarks:
                data  = []
                wrist = hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    data.append(lm.x - wrist.x)
                    data.append(lm.y - wrist.y)

                try:
                    prediction = model.predict([data])[0]
                    confidence = float(max(model.predict_proba([data])[0]))
                except Exception:
                    prediction, confidence = "-", 0.0

                with self._lock:
                    self.current_pred = str(prediction)
                    self.current_conf = confidence

                    if confidence >= self.CONFIDENCE_THRESHOLD:
                        self.buffer.append(prediction)

                    if len(self.buffer) == self.BUFFER_SIZE:
                        most_common, count = Counter(self.buffer).most_common(1)[0]
                        agreement = count / self.BUFFER_SIZE
                        if agreement >= self.AGREEMENT_RATIO:
                            if most_common != self.last_added:
                                if time.time() - self.last_time > self.COOLDOWN:
                                    if most_common == "SPACE":
                                        self.current_sentence += " "
                                    else:
                                        self.current_sentence += str(most_common)
                                    self.last_added = most_common
                                    self.last_time  = time.time()
                                    self.current_history.append(str(most_common))
                                    self.current_history = self.current_history[-5:]
                                    self.last_word       = str(most_common)
                                    self.new_word_ready  = True

                color = (0, 229, 160) if confidence >= self.CONFIDENCE_THRESHOLD else (100, 100, 255)
                cv2.putText(img, f"{prediction}  {int(confidence*100)}%",
                            (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                h, w = img.shape[:2]
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x1 = max(0, int(min(xs)*w) - 20)
                y1 = max(0, int(min(ys)*h) - 20)
                x2 = min(w, int(max(xs)*w) + 20)
                y2 = min(h, int(max(ys)*h) + 20)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 229, 160), 2)
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
                        self.current_history = self.current_history[-5:]
                    self.no_hand_start = None

            overlay = img.copy()
            cv2.rectangle(overlay, (8, 8), (280, 54), (5, 8, 25), -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            cv2.putText(img, "No hand detected", (14, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 100, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
pages = [
    ("🏠",  "Home"),
    ("📷",  "Live Recognition"),
    ("📖",  "Instructions"),
    ("🤚",  "Gesture Guide"),
    ("🕐",  "History"),
    ("⚙️",  "Settings"),
    ("ℹ️",  "About"),
]

with st.sidebar:
    st.markdown("""
    <div class="sb-header">
        <div class="sb-logo-row">
            <div class="sb-logo-box">🤟</div>
            <div>
                <div class="sb-logo-title">ASL Vision Pro</div>
                <div class="sb-logo-sub">Recognition System</div>
            </div>
        </div>
        <span class="sb-version-badge">v2.0 PREMIUM</span>
    </div>
    <div class="sb-section-label">Navigation</div>
    """, unsafe_allow_html=True)

    for icon, label in pages:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown("""
    <div class="sb-footer">
        <div class="sb-footer-text">
            Built with MediaPipe + ML<br>
            <span style="color:#5b7cff">ASL Vision Pro © 2025</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── TOP BAR ───────────────────────────────────────────────────────────────────
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
    <div class="topbar-status">
        <div class="page-indicator">/{pg.upper().replace(" ", "_")}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if pg == "Home":
    st.markdown("""
    <div class="page-hero">
        <h1>Welcome to ASL Vision Pro 👋</h1>
        <p>Real-time American Sign Language detection powered by MediaPipe + Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
        <div class="feature-card-wrap" style="border-top:2px solid #5b7cff">
            <span class="feature-icon">🎥</span>
            <div class="feature-title" style="color:#5b7cff">Live Detection</div>
            <div class="feature-desc">Real-time hand gesture recognition from your webcam with instant visual feedback.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Live Detection", key="home_live", use_container_width=True):
            st.session_state.page = "Live Recognition"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="feature-card-wrap" style="border-top:2px solid #00e5a0">
            <span class="feature-icon">🧠</span>
            <div class="feature-title" style="color:#00e5a0">ML Powered</div>
            <div class="feature-desc">Confidence-buffered classifier with majority voting for flicker-free outputs.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Learn How It Works", key="home_ml", use_container_width=True):
            st.session_state.page = "About"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="feature-card-wrap" style="border-top:2px solid #ff4d94">
            <span class="feature-icon">📝</span>
            <div class="feature-title" style="color:#ff4d94">Text Output</div>
            <div class="feature-desc">Detected ASL signs seamlessly assembled into readable sentences in real-time.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("View History", key="home_history", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()

    st.markdown("""
    <div class="card" style="margin-top:8px">
        <div class="card-label">Quick Start Guide</div>
        <div class="qs-step">
            <div class="qs-num">01</div>
            <div class="qs-content">Click <b>Live Recognition</b> in the sidebar to open the camera view</div>
        </div>
        <div class="qs-step">
            <div class="qs-num">02</div>
            <div class="qs-content">Allow camera access and click <b>START</b> to begin</div>
        </div>
        <div class="qs-step">
            <div class="qs-num">03</div>
            <div class="qs-content">Show ASL hand signs clearly — <b>hold each sign for 1–2 seconds</b></div>
        </div>
        <div class="qs-step">
            <div class="qs-num">04</div>
            <div class="qs-content">Hide your hand for 1 second to insert a <b>space</b> between letters</div>
        </div>
        <div class="qs-step">
            <div class="qs-num">05</div>
            <div class="qs-content">View your full session log in the <b>History</b> page</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE RECOGNITION  ←  FIXED
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "Live Recognition":
    col_vid, col_info = st.columns([55, 45], gap="large")

    with col_vid:
        st.markdown('<div class="card-label" style="margin-bottom:12px">🎥 Live Camera Feed</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="asl_live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ASLProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

    # ── FIX: State polling happens OUTSIDE col_vid ──────────────────────────
    state = {}
    if ctx and ctx.video_processor:
        try:
            state = ctx.video_processor.get_state()
            if state.get("new_word"):
                word = ctx.video_processor.consume_new_word()
                if word and word.strip():
                    st.session_state.history_log.insert(0, {
                        "word": word,
                        "time": datetime.now().strftime("%I:%M:%S %p")
                    })
                    st.session_state.history_log = st.session_state.history_log[:100]
            st.session_state.sentence = state.get("sentence", "")
        except Exception:
            pass

    # ── Right panel ──────────────────────────────────────────────────────────
    with col_info:
        pred     = state.get("pred", "-")
        conf     = state.get("conf", 0.0)
        sentence = state.get("sentence", st.session_state.sentence)
        hist     = state.get("history", [])
        hand_ok  = state.get("hand", False)

        conf_pct   = int(conf * 100)
        ring_deg   = f"{int(conf * 360)}deg"
        is_pred    = pred not in ["-", "", None]
        pred_cls   = "pred-letter" if is_pred else "pred-letter no-pred"
        pred_disp  = str(pred) if is_pred else "—"
        conf_color = "#00e5a0" if conf >= 0.6 else ("#f5c842" if conf >= 0.4 else "#ff4d94")

        badge_cls = "live" if hand_ok else "offline"
        badge_dot = "green" if hand_ok else "red"
        badge_txt = "Hand Detected" if hand_ok else "No Hand — Show your hand"

        st.markdown(f"""
        <div style="margin-bottom:12px">
            <span class="status-pill {badge_cls}">
                <span class="pulse {badge_dot}"></span>{badge_txt}
            </span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card green-top" style="margin-bottom:14px">
            <div class="card-label">Current Prediction</div>
            <div class="pred-container">
                <div class="pred-ring" style="--conf:{ring_deg}">
                    <div class="{pred_cls}">{pred_disp}</div>
                </div>
                <div class="pred-conf-badge" style="color:{conf_color}">{conf_pct}%</div>
                <div class="pred-conf-label">Confidence Score</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{conf_pct}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-label">Current Sentence</div>', unsafe_allow_html=True)
        disp = sentence if sentence else ""
        st.markdown(f"""
        <div class="sentence-outer" style="margin-bottom:14px">
            <div class="sentence-text">{disp if disp else '<span style="color:var(--muted)">Waiting...</span>'}
                <span class="sentence-cursor"></span>
            </div>
        </div>""", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("⎵  Space", key="sp", use_container_width=True):
                if ctx and ctx.video_processor:
                    try: ctx.video_processor.add_space()
                    except: pass
        with b2:
            if st.button("⌫  Delete", key="dl", use_container_width=True):
                if ctx and ctx.video_processor:
                    try: ctx.video_processor.delete_last()
                    except: pass
        with b3:
            if st.button("🗑  Clear", key="cl", use_container_width=True):
                if ctx and ctx.video_processor:
                    try: ctx.video_processor.clear_all()
                    except: pass
                st.session_state.sentence = ""

        if hist:
            st.markdown('<div class="card-label" style="margin-top:14px">Recent Signs</div>', unsafe_allow_html=True)
            chips_html = ""
            for i, h in enumerate(hist):
                if i > 0:
                    chips_html += '<span class="sign-arrow">›</span>'
                chips_html += f'<div class="sign-chip">{h}</div>'
            st.markdown(f'<div class="sign-trail">{chips_html}</div>', unsafe_allow_html=True)

    # ── KEY FIX: Auto-rerun every 0.5s while camera is active ───────────────
    if ctx and ctx.state.playing:
        time.sleep(0.5)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "Instructions":
    st.markdown('<div class="page-hero"><h1>📖 Instructions</h1><p>Follow these tips for the best detection accuracy.</p></div>', unsafe_allow_html=True)

    steps = [
        ("💡", "Good Lighting",       "Ensure your hand is well-lit. Natural daylight or a lamp in front of you works best. Avoid backlighting."),
        ("👤", "Hand Fully in Frame",  "Keep your entire hand visible. Partial or cropped hands reduce landmark detection accuracy significantly."),
        ("📏", "Optimal Distance",     "Stay 1–2 feet from the camera. Too close distorts proportions; too far loses landmark precision."),
        ("✋", "Hold Signs Steady",    "Hold each sign still for 1–2 seconds. 15 consistent frames are needed to confirm a detection."),
        ("🔄", "Face Palm Forward",    "Point your palm toward the camera. Side angles may not match the model's training orientation."),
        ("☝️", "One Hand Only",        "This system is trained for single-hand ASL. Keep the other hand out of frame to avoid interference."),
    ]
    col_a, col_b = st.columns(2, gap="large")
    for i, (icon, title, desc) in enumerate(steps):
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;gap:16px;align-items:flex-start">
                    <div style="width:42px;height:42px;border-radius:10px;background:rgba(91,124,255,0.1);border:1px solid rgba(91,124,255,0.2);display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0">{icon}</div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:14px;color:#fff;margin-bottom:6px">Step {i+1}: {title}</div>
                        <div style="font-size:12px;color:var(--muted);line-height:1.7">{desc}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GESTURE GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
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
    ]
    cols = st.columns(5, gap="small")
    for i, (emoji, label, desc) in enumerate(gestures):
        with cols[i % 5]:
            st.markdown(f"""
            <div class="g-grid-card">
                <span class="g-emoji-big">{emoji}</span>
                <span class="g-letter">{label}</span>
                <span class="g-desc">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card accent-top" style="margin-top:16px">
        <div class="card-label">Important Notes</div>
        <div style="color:var(--muted);font-size:13px;line-height:2.2">
            • <b style="color:#fff">J</b> and <b style="color:#fff">Z</b> are motion-based — not supported in this version<br>
            • <b style="color:#fff">SPACE</b>: Hide your hand for 1 second to insert a space<br>
            • Accuracy improves in good lighting with hand centred in frame
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "History":
    st.markdown('<div class="page-hero"><h1>🕐 Detection History</h1><p>All signs detected in this session.</p></div>', unsafe_allow_html=True)

    col_h, col_stats = st.columns([3, 1], gap="large")

    with col_h:
        if not st.session_state.history_log:
            st.markdown("""
            <div class="card" style="text-align:center;padding:60px 20px">
                <div style="font-size:56px;margin-bottom:16px">🕐</div>
                <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#fff;margin-bottom:8px">No detections yet</div>
                <div style="font-size:13px;color:var(--muted)">Go to <b style="color:var(--accent)">Live Recognition</b> and start signing!</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for entry in st.session_state.history_log:
                st.markdown(f"""
                <div class="hist-item">
                    <div style="display:flex;align-items:center;gap:14px">
                        <div class="hist-sign">{entry['word']}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:12px;color:var(--text)">{entry['word']}</div>
                    </div>
                    <div class="hist-time">{entry['time']}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.history_log = []
                st.rerun()

    with col_stats:
        total  = len(st.session_state.history_log)
        unique = len(set(e["word"] for e in st.session_state.history_log))
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-num" style="color:var(--accent)">{total}</div>
            <div class="stat-label">Total Signs</div>
        </div>
        <div class="stat-card">
            <div class="stat-num" style="color:var(--green)">{unique}</div>
            <div class="stat-label">Unique Signs</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "Settings":
    st.markdown('<div class="page-hero"><h1>⚙️ Settings</h1><p>Fine-tune detection parameters for your environment.</p></div>', unsafe_allow_html=True)

    cs1, cs2 = st.columns(2, gap="large")
    with cs1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Detection Parameters</div>', unsafe_allow_html=True)
        conf_thresh = st.slider("Confidence Threshold",  0.10, 0.90, st.session_state.conf_thresh, 0.05)
        buf_size    = st.slider("Buffer Size (frames)",  5,    30,   st.session_state.buf_size,    1)
        agree_ratio = st.slider("Agreement Ratio",       0.40, 0.95, st.session_state.agree_ratio, 0.05)
        cooldown    = st.slider("Cooldown (seconds)",    0.5,  3.0,  st.session_state.cooldown,    0.1)
        space_delay = st.slider("Space Delay (seconds)", 0.5,  3.0,  st.session_state.space_delay, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("💾  Apply Settings", use_container_width=True):
            st.session_state.conf_thresh  = conf_thresh
            st.session_state.buf_size     = buf_size
            st.session_state.agree_ratio  = agree_ratio
            st.session_state.cooldown     = cooldown
            st.session_state.space_delay  = space_delay
            st.success("✅ Settings saved! Restart the camera on Live page to apply.")

    with cs2:
        st.markdown("""
        <div class="card">
            <div class="card-label">Parameter Guide</div>
            <div class="param-item">
                <div><div class="param-name">Confidence Threshold</div><div class="param-desc">Lower = more sensitive. Higher = more precise but slower.</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Buffer Size</div><div class="param-desc">More frames = stable but slower response. Fewer = faster but shaky.</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Agreement Ratio</div><div class="param-desc">% of buffer frames that must agree on the same sign.</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Cooldown</div><div class="param-desc">Min gap (seconds) to prevent the same letter repeating too fast.</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Space Delay</div><div class="param-desc">Hide hand for this many seconds to auto-insert a space.</div></div>
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif pg == "About":
    st.markdown('<div class="page-hero"><h1>ℹ️ About</h1><p>Technology stack and model details.</p></div>', unsafe_allow_html=True)

    ca1, ca2 = st.columns([2, 1], gap="large")
    with ca1:
        st.markdown("""
        <div class="about-hero">
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:#fff;margin-bottom:14px">
                ASL Vision Pro — Sign Language Recognition
            </div>
            <div style="color:var(--muted);font-size:13px;line-height:2">
                This system uses <b style="color:#fff">Machine Learning</b> and <b style="color:#fff">Computer Vision</b>
                to recognize American Sign Language (ASL) gestures in real-time and convert them to text.<br><br>
                <b style="color:var(--accent)">MediaPipe</b> extracts 21 hand landmarks per frame. Coordinates are normalized
                relative to the wrist, then fed into a trained classifier. A 15-frame confidence buffer with majority voting
                ensures stable, flicker-free output.<br><br>
                Built to bridge communication gaps for the hearing-impaired community.
            </div>
            <div class="tag-row">
                <span class="tag" style="color:#3b82f6;border-color:#3b82f640;background:#3b82f610">Python</span>
                <span class="tag" style="color:#f59e0b;border-color:#f59e0b40;background:#f59e0b10">OpenCV</span>
                <span class="tag" style="color:#ef4444;border-color:#ef444440;background:#ef444410">Scikit-learn</span>
                <span class="tag" style="color:#a78bfa;border-color:#a78bfa40;background:#a78bfa10">MediaPipe</span>
                <span class="tag" style="color:#f97316;border-color:#f9731640;background:#f9731610">Streamlit</span>
                <span class="tag" style="color:#06b6d4;border-color:#06b6d440;background:#06b6d410">WebRTC</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with ca2:
        st.markdown("""
        <div class="card accent-top">
            <div class="card-label">Model Info</div>
            <div class="param-item">
                <div><div class="param-name">Input Features</div><div class="param-desc">42 (21 landmarks × x, y)</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Normalization</div><div class="param-desc">Wrist-relative coordinates</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Classifier</div><div class="param-desc">Random Forest / SVM</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Output Classes</div><div class="param-desc">A–Z, 0–9, SPACE</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Buffer Logic</div><div class="param-desc">15-frame majority vote</div></div>
            </div>
            <div class="param-item">
                <div><div class="param-name">Min Confidence</div><div class="param-desc">40% (adjustable)</div></div>
            </div>
        </div>""", unsafe_allow_html=True)