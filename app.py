import streamlit as st
import cv2
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Load model
model = pickle.load(open("model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sentence = ""
last_prediction = ""
frame_count = 0
current_prediction = ""
threshold = 6

class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.sentence = ""
        self.last_prediction = ""
        self.frame_count = 0
        self.current_prediction = ""
        self.space_lock = False
        self.skip_frames = 2
        self.frame_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        self.frame_counter += 1
        if self.frame_counter % self.skip_frames != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

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

                if prediction == self.current_prediction:
                    self.frame_count += 1
                else:
                    self.current_prediction = prediction
                    self.frame_count = 0

                if self.frame_count >= threshold:

                    if prediction == "SPACE" and not self.space_lock:
                        self.sentence += " "
                        self.space_lock = True
                        self.last_prediction = "SPACE"

                    elif prediction != "SPACE" and prediction != self.last_prediction:
                        self.sentence += prediction
                        self.last_prediction = prediction
                        self.space_lock = False

                cv2.putText(img, f"{prediction}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(img, f"Sentence: {self.sentence}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("ASL Recognition (Web Version)")

webrtc_streamer(
    key="asl",
    video_processor_factory=ASLProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
