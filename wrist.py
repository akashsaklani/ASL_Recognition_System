import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

file = open("dataset.csv", "a", newline="")
writer = csv.writer(file)

label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    data = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                x_rel = lm.x - wrist.x
                y_rel = lm.y - wrist.y
                data.append(x_rel)
                data.append(y_rel)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show label
    cv2.putText(frame, f"Label: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1)

    # -------- KEY CONTROLS -------- #

    # ESC → Exit
    if key == 27:
        break

    # ENTER → Save data
    elif key == 13:
        if data and label != "":
            row = data + [label]
            writer.writerow(row)
            print("Saved:", label)

    # SPACE → Space gesture
    elif key == 32:
        label = "SPACE"

    # a–z → Labels (including s and q)
    elif 97 <= key <= 122:
        label = chr(key).upper()

# ------------------------------- #

cap.release()
file.close()
cv2.destroyAllWindows()