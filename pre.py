import cv2
import mediapipe as mp
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            data = []
            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                x_rel = lm.x - wrist.x
                y_rel = lm.y - wrist.y
                data.append(x_rel)
                data.append(y_rel)

            prediction = model.predict([data])[0]

            cv2.putText(frame, f"{prediction}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()