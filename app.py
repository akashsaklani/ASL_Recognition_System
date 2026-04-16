import streamlit as st
import cv2
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from collections import deque, Counter
import time

st.set_page_config(layout="wide")

st.markdown("""
<style>
.card {
    background: linear-gradient(145deg,#1a1f2e,#0f172a);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0 0 15px rgba(108,99,255,0.2);
}
.card-title { color: #aaa; font-size: 14px; margin-bottom: 6px; }
.stProgress > div > div { background-color: #6C63FF; }
</style>
""", unsafe_allow_html=True)

model = pickle.load(open("model.pkl", "rb"))
mp_hands = mp.solutions.hands

class ASLProcessor(VideoProcessorBase):

    # ── Tune these 3 values to control stability ──────────────────────────────
    CONFIDENCE_THRESHOLD = 0.40   # ignore predictions below 40% confidence
    BUFFER_SIZE          = 15     # how many frames to collect before deciding
    AGREEMENT_RATIO      = 0.65  # 65% of buffer must agree on the same sign
    COOLDOWN             = 1.2    # seconds before the same letter can repeat
    SPACE_DELAY          = 1.0    # seconds of no hand → insert space
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self):
        self.current_pred    = "-"
        self.current_conf    = 0.0
        self.current_sentence = ""
        self.current_history  = []
        self.buffer           = deque(maxlen=self.BUFFER_SIZE)
        self.last_added       = ""
        self.last_time        = time.time()
        self.no_hand_start    = None          # tracks when hand disappeared
        self.hands            = mp_hands.Hands(max_num_hands=1)

    def recv(self, frame):
        img       = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result    = self.hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            self.no_hand_start = None           # hand is visible, reset space timer

            for hand_landmarks in result.multi_hand_landmarks:
                data  = []
                wrist = hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    data.append(lm.x - wrist.x)
                    data.append(lm.y - wrist.y)

                prediction = model.predict([data])[0]
                confidence = float(max(model.predict_proba([data])[0]))

                self.current_pred = prediction
                self.current_conf = confidence

                # ✅ FIX 1: Only add to buffer if confidence is high enough
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    self.buffer.append(prediction)

                # ✅ FIX 2: Require strong agreement across the full buffer
                if len(self.buffer) == self.BUFFER_SIZE:
                    most_common, count = Counter(self.buffer).most_common(1)[0]
                    agreement = count / self.BUFFER_SIZE

                    if agreement >= self.AGREEMENT_RATIO:
                        if most_common != self.last_added:
                            if time.time() - self.last_time > self.COOLDOWN:
                                if most_common == "SPACE":
                                    self.current_sentence += " "
                                else:
                                    self.current_sentence += most_common
                                self.last_added        = most_common
                                self.last_time         = time.time()
                                self.current_history.append(most_common)
                                self.current_history = self.current_history[-5:]

                # Show prediction + confidence on frame
                color = (0, 255, 0) if confidence >= self.CONFIDENCE_THRESHOLD else (0, 0, 255)
                cv2.putText(img, f"{prediction} {int(confidence*100)}%",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        else:
            self.buffer.clear()
            self.current_pred = "-"
            self.current_conf = 0.0

            # ✅ FIX 3: Hide hand for SPACE_DELAY seconds → add a space
            if self.no_hand_start is None:
                self.no_hand_start = time.time()

            elif time.time() - self.no_hand_start >= self.SPACE_DELAY:
                # Only add one space (don't keep adding)
                if self.current_sentence and self.current_sentence[-1] != " ":
                    self.current_sentence += " "
                    self.last_added = " "
                    self.current_history.append(" ")
                    self.current_history = self.current_history[-5:]
                self.no_hand_start = None   # reset so next hide triggers fresh

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── UI ────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("### 🎥 Live Detection")
    ctx = webrtc_streamer(
        key="asl",
        video_processor_factory=ASLProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if ctx and ctx.video_processor:
    proc     = ctx.video_processor
    pred     = proc.current_pred
    conf     = proc.current_conf
    sentence = proc.current_sentence
    history  = list(proc.current_history)
else:
    pred, conf, sentence, history = "-", 0.0, "", []

with col2:

    st.markdown('<div class="card"><div class="card-title">Detected Sign</div></div>', unsafe_allow_html=True)
    st.markdown(f"### {pred}")
    st.progress(conf)
    st.caption(f"Confidence: {round(conf * 100)}%")

    st.markdown('<div class="card"><div class="card-title">Sentence</div></div>', unsafe_allow_html=True)
    st.write(sentence if sentence else "Waiting...")

    st.markdown('<div class="card"><div class="card-title">History</div></div>', unsafe_allow_html=True)
    st.write(", ".join(history) if history else "Waiting...")

    if st.button("🗑️ Clear"):
        if ctx and ctx.video_processor:
            ctx.video_processor.current_sentence = ""
            ctx.video_processor.current_history  = []
            ctx.video_processor.last_added       = ""

time.sleep(0.5)
st.rerun()