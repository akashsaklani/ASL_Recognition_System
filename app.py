import streamlit as st
import cv2
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ---------- CONFIG ----------
st.set_page_config(layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>

.card {
    background: linear-gradient(145deg,#1a1f2e,#0f172a);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0 0 15px rgba(108,99,255,0.2);
}

.card-title {
    color: #aaa;
    font-size: 14px;
}

.big-letter {
    font-size: 50px;
    color: #6C63FF;
    font-weight: bold;
}

.sentence-box {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
    color: white;
}

.stProgress > div > div {
    background-color: #6C63FF;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# ---------- SESSION ----------
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "last_pred" not in st.session_state:
    st.session_state.last_pred = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

# ---------- PROCESSOR ----------
class ASLProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.current_prediction = ""

    def recv(self, frame):
        state = st.session_state

        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                data = []
                wrist = hand_landmarks.landmark[0]

                for lm in hand_landmarks.landmark:
                    data.append(lm.x - wrist.x)
                    data.append(lm.y - wrist.y)

                prediction = model.predict([data])[0]
                state.confidence = max(model.predict_proba([data])[0])

                # stability logic
                if prediction == self.current_prediction:
                    self.frame_count += 1
                else:
                    self.current_prediction = prediction
                    self.frame_count = 0

                if self.frame_count >= 2:
                    if prediction != state.last_pred:
                        state.sentence += prediction
                        state.last_pred = prediction

                        state.history.append(prediction)
                        state.history = state.history[-5:]

                cv2.putText(img, prediction, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:
            self.frame_count = 0
            self.current_prediction = ""

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- UI ----------

col1, col2 = st.columns([4,1])

with col1:
    st.markdown("### 🎥 Live Detection")

    webrtc_streamer(
        key="asl",
        video_processor_factory=ASLProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:

    # 🔥 DETECTED SIGN CARD
    st.markdown("""
    <div class="card">
        <div class="card-title">Detected Sign</div>
    </div>
    """, unsafe_allow_html=True)
    st.write(st.session_state.get('last_pred', '-'))

    # 🔥 CONFIDENCE BAR
    st.progress(st.session_state.get('confidence', 0.0))

    # 🔥 SENTENCE CARD
    st.markdown("""
    <div class="card">
        <div class="card-title">Sentence</div>
    </div>
    """, unsafe_allow_html=True)
    st.write(st.session_state.get('sentence', 'Waiting...'))

    # 🔥 HISTORY CARD
    st.markdown("""
    <div class="card">
        <div class="card-title">History</div>
    </div>
    """, unsafe_allow_html=True)
    history_text = ", ".join(st.session_state.get('history', [])) or "Waiting..."
    st.write(history_text)

    # 🔥 BUTTON
    if st.button("Clear Sentence"):
        st.session_state.sentence = ""
        st.session_state.last_pred = ""
        st.session_state.history = []

st.empty()