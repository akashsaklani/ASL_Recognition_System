import cv2
import mediapipe as mp
import pickle
import pyttsx3
import queue
import threading
import time

# Load model
model = pickle.load(open("model.pkl", "rb"))
speech_queue = queue.Queue()
def speech_worker():
    local_engine = pyttsx3.init()
    local_engine.setProperty('rate', 150)

    while True:
        text = speech_queue.get()
        if text is None:
            break
        local_engine.stop()
        local_engine.say(text)
        local_engine.runAndWait()
        time.sleep(0.3)  # 🔥 small delay to prevent rapid speech
threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

prev_spoken_word = ""

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sentence = ""
last_prediction = ""
frame_count = 0
current_prediction = ""
threshold = 12
no_hand_frames = 0
reset_threshold = 10
space_lock = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        no_hand_frames = 0
        for hand_landmarks in result.multi_hand_landmarks:
            data = []
            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                x_rel = lm.x - wrist.x
                y_rel = lm.y - wrist.y
                data.append(x_rel)
                data.append(y_rel)

            prediction = model.predict([data])[0]

            if prediction == current_prediction:
                frame_count += 1
            else:
                current_prediction = prediction
                frame_count = 0

            if frame_count >= threshold:
                
                if prediction == "SPACE" and not space_lock:

                    words = sentence.strip().split(" ")
                    if len(words) > 0:

                        last_word = words[-1]

                        if last_word != "" and last_word != prev_spoken_word:
                            print("Speaking:", last_word)
                            speak(last_word)
                            prev_spoken_word = last_word

                    sentence += " "

                    space_lock = True
                    time.sleep(0.5)  # 🔥 small delay to prevent multiple spaces
                    last_prediction = "SPACE"

                    frame_count = 0
                    current_prediction = ""

                elif prediction != "SPACE" and prediction != last_prediction:
                    sentence += prediction
                    last_prediction = prediction

                    space_lock = False   # 🔥 unlock

            cv2.putText(frame, f"{prediction}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Sentence: {sentence}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        no_hand_frames += 1

        if no_hand_frames > reset_threshold:
            last_prediction = ""
            current_prediction = ""
            frame_count = 0

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()