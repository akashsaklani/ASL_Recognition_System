import cv2
import mediapipe as mp
import pickle
import pyttsx3

# Load model
model = pickle.load(open("model.pkl", "rb"))

engine = pyttsx3.init()
engine.setProperty('rate', 150)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sentence = ""
last_prediction = ""
current_word = ""
frame_count = 0
current_prediction = ""
threshold = 15   # jitna bada, utna slow/accurate
no_hand_frames = 0
reset_threshold = 10
space_hold = 0
space_threshold = 10

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

            if prediction == "SPACE":
                space_hold += 1
            else:
                space_hold = 0

            if prediction == current_prediction:
                frame_count += 1
            else:
                current_prediction = prediction
                frame_count = 0

            if frame_count == threshold:
                if prediction != last_prediction:
                    if prediction == "SPACE":
                        sentence += " "
                    else:
                        sentence += prediction
                        current_word += prediction
                    last_prediction = prediction

            if space_hold == space_threshold:
                if current_word != "":
                    engine.say(current_word)
                    engine.runAndWait()
                    current_word = ""

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