import cv2
import mediapipe as mp
import pickle
import pyttsx3
import queue
import threading
import time
import json

# ── MODEL LOAD ────────────────────────────────────────────────────────────────
model = pickle.load(open("model.pkl", "rb"))

# ── TTS CONFIG (runtime-changeable) ──────────────────────────────────────────
tts_config = {
    "rate":   150,      # wpm
    "gender": "male",   # "male" | "female"
    "mode":   "end",    # "end"  | "letter"
}

# ── SPEECH ENGINE ─────────────────────────────────────────────────────────────
speech_queue = queue.Queue()

def _pick_voice(engine, gender):
    """Return best matching voice id for requested gender."""
    voices = engine.getProperty("voices")
    matched = [v for v in voices
               if gender.lower() in (v.gender or "").lower()
               or gender.lower() in v.name.lower()]
    if matched:
        return matched[0].id
    # Windows fallback: index 0 ≈ male, index 1 ≈ female
    if gender == "female" and len(voices) > 1:
        return voices[1].id
    return voices[0].id

def speech_worker():
    engine = pyttsx3.init()

    def apply():
        engine.setProperty("rate",  tts_config["rate"])
        engine.setProperty("voice", _pick_voice(engine, tts_config["gender"]))

    apply()

    while True:
        item = speech_queue.get()
        if item is None:
            break
        if item == "__RELOAD__":
            apply()
            continue
        try:
            engine.stop()
        except Exception:
            pass
        apply()
        engine.say(item)
        engine.runAndWait()
        time.sleep(0.25)

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    if not text.strip():
        return
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break
    speech_queue.put(text.strip())

def reload_voice():
    speech_queue.put("__RELOAD__")

# ── SENTENCE SYNC ─────────────────────────────────────────────────────────────
SYNC_FILE = "sentence.json"

def write_sync(sentence, last_word="", event=""):
    try:
        with open(SYNC_FILE, "w") as f:
            json.dump({
                "sentence":   sentence,
                "last_word":  last_word,
                "event":      event,
                "tts_mode":   tts_config["mode"],
                "tts_gender": tts_config["gender"],
                "tts_rate":   tts_config["rate"],
                "timestamp":  time.time(),
            }, f)
    except Exception:
        pass

write_sync("", "", "init")

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)
mp_draw    = mp.solutions.drawing_utils
HAND_STYLE = mp_draw.DrawingSpec(color=(0, 229, 160), thickness=2, circle_radius=3)
CONN_STYLE = mp_draw.DrawingSpec(color=(91, 124, 255), thickness=2)

# ── STATE ─────────────────────────────────────────────────────────────────────
sentence           = ""
last_prediction    = ""
current_prediction = ""
frame_count        = 0
threshold          = 12
no_hand_frames     = 0
reset_threshold    = 15
space_lock         = False
back_lock          = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── OVERLAY HELPERS ───────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_progress_bar(img, x, y, w, h, ratio, fg=(0,229,160), bg=(30,35,60)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    filled = int(w * max(0, min(1, ratio)))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x+filled, y+h), fg, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (60,70,100), 1)

def put_text(img, text, x, y, scale=0.6, color=(230,235,255), thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result     = hands.process(frame_rgb)

    prediction  = "-"
    confidence  = 0.0
    hand_active = False

    if result.multi_hand_landmarks:
        no_hand_frames = 0
        hand_active    = True
        space_lock     = False

        for hand_landmarks in result.multi_hand_landmarks:
            data  = []
            wrist = hand_landmarks.landmark[0]
            for lm in hand_landmarks.landmark:
                data.append(lm.x - wrist.x)
                data.append(lm.y - wrist.y)

            prediction = str(model.predict([data])[0])
            proba      = model.predict_proba([data])[0]
            confidence = float(max(proba))

            if prediction == current_prediction:
                frame_count += 1
            else:
                current_prediction = prediction
                frame_count        = 0

            if frame_count >= threshold:

                # END → speak full sentence
                if prediction == "END":
                    if sentence.strip():
                        speak(sentence.strip())
                        write_sync(sentence, "", "end")
                    frame_count        = 0
                    current_prediction = ""

                # SPACE
                elif prediction == "SPACE" and not space_lock:
                    sentence       += " "
                    space_lock      = True
                    last_prediction = "SPACE"
                    frame_count     = 0
                    current_prediction = ""
                    write_sync(sentence, " ", "space")

                # DEL → backspace
                elif prediction == "DEL" and not back_lock:
                    if sentence:
                        sentence  = sentence[:-1]
                        back_lock = True
                        write_sync(sentence, "", "backspace")
                    frame_count        = 0
                    current_prediction = ""

                # Normal letter
                elif prediction not in ("SPACE","END","DEL") and prediction != last_prediction:
                    sentence       += prediction
                    last_prediction = prediction
                    back_lock       = False
                    space_lock      = False
                    write_sync(sentence, prediction, "add")

                    # Letter-by-letter mode → speak each letter as added
                    if tts_config["mode"] == "letter":
                        speak(prediction)

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS,
                                   HAND_STYLE, CONN_STYLE)

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            bx1 = max(0, int(min(xs)*w)-20);  by1 = max(0, int(min(ys)*h)-20)
            bx2 = min(w, int(max(xs)*w)+20);  by2 = min(h, int(max(ys)*h)+20)
            cv2.rectangle(frame, (bx1,by1), (bx2,by2),
                          (0,229,160) if confidence>=0.5 else (91,124,255), 2)

    else:
        no_hand_frames += 1
        if no_hand_frames > reset_threshold:
            last_prediction    = ""
            current_prediction = ""
            frame_count        = 0
            back_lock          = False
            if no_hand_frames == reset_threshold + 1:
                if sentence and sentence[-1] != " ":
                    sentence  += " "
                    space_lock = True
                    write_sync(sentence, " ", "space")

    # ── HUD ───────────────────────────────────────────────────────────────────

    # ── Top-left: prediction + bars ──────────────────────────────────────────
    draw_rounded_rect(frame, 10, 10, 320, 128, 12, (8,12,30))
    pred_color = (0,229,160) if confidence>=0.5 else (91,124,255)
    put_text(frame, "PREDICTION", 24, 36, 0.45, (100,110,160))
    put_text(frame, prediction if hand_active else "-", 24, 84, 1.6, pred_color, 2)
    conf_pct = int(confidence*100)
    put_text(frame, f"CONF  {conf_pct}%", 134, 56, 0.5, (200,210,255))
    draw_progress_bar(frame, 134, 65, 166, 8, confidence,
                      fg=(0,229,160) if confidence>=0.5 else (255,77,148))
    stab = frame_count / threshold
    put_text(frame, f"STAB  {int(stab*100)}%", 134, 98, 0.5, (170,180,220))
    draw_progress_bar(frame, 134, 107, 166, 6, stab, fg=(91,124,255))

    # ── Top-right: hand status ────────────────────────────────────────────────
    status_txt   = "HAND DETECTED" if hand_active else "NO HAND"
    status_color = (0,229,160) if hand_active else (255,77,148)
    pill_x = w - 210
    draw_rounded_rect(frame, pill_x, 12, w-12, 42, 10, (8,12,30))
    cv2.circle(frame, (pill_x+18, 27), 5, status_color, -1)
    put_text(frame, status_txt, pill_x+30, 32, 0.45, status_color)

    # ── TTS config panel (top-right) ─────────────────────────────────────────
    gender_color = (200,140,255) if tts_config["gender"]=="female" else (100,200,255)
    mode_color   = (0,229,160)   if tts_config["mode"]=="end"      else (255,210,60)
    draw_rounded_rect(frame, pill_x, 50, w-12, 152, 8, (8,12,30))
    put_text(frame, "TTS SETTINGS", pill_x+10, 70, 0.40, (80,90,130))
    # Gender row
    put_text(frame, "VOICE",  pill_x+10, 94,  0.42, (120,130,170))
    put_text(frame, tts_config["gender"].upper(), pill_x+72, 94, 0.44, gender_color, 1)
    put_text(frame, "[G]", w-58, 94, 0.38, (255,210,60))
    # Mode row
    put_text(frame, "MODE",   pill_x+10, 116, 0.42, (120,130,170))
    put_text(frame, tts_config["mode"].upper(),   pill_x+72, 116, 0.44, mode_color, 1)
    put_text(frame, "[M]", w-58, 116, 0.38, (255,210,60))
    # Rate row
    put_text(frame, "RATE",   pill_x+10, 138, 0.42, (120,130,170))
    put_text(frame, f"{tts_config['rate']} wpm",  pill_x+72, 138, 0.44, (200,215,255))
    put_text(frame, "[+-]", w-62, 138, 0.38, (255,210,60))

    # ── Gesture hints (right, below TTS panel) ────────────────────────────────
    hints = [("END","speak sentence"),("SPACE","insert space"),("DEL","backspace")]
    draw_rounded_rect(frame, pill_x, 160, w-12, 254, 8, (8,12,30))
    put_text(frame, "GESTURES", pill_x+10, 178, 0.38, (80,90,130))
    for i,(g,desc) in enumerate(hints):
        gy = 200 + i*22
        put_text(frame, g,    pill_x+10, gy, 0.42, (91,124,255))
        put_text(frame, desc, pill_x+72, gy, 0.40, (140,155,200))

    # ── Bottom sentence panel ────────────────────────────────────────────────
    sent_display = (sentence[-52:] if len(sentence)>52 else sentence) + "|"
    draw_rounded_rect(frame, 10, h-70, w-12, h-10, 12, (8,12,30))
    put_text(frame, "SENTENCE", 24, h-48, 0.40, (100,110,160))
    put_text(frame, sent_display if sentence else "Show a sign to begin...",
             24, h-24, 0.65,
             (230,235,255) if sentence else (80,90,130), 1)

    # Mode indicator badge (bottom-left above sentence)
    mode_badge = f"MODE: {'LETTER-BY-LETTER' if tts_config['mode']=='letter' else 'END-ONLY'}  |  VOICE: {tts_config['gender'].upper()}"
    draw_rounded_rect(frame, 10, h-98, 480, h-76, 8, (8,12,30), alpha=0.5)
    put_text(frame, mode_badge, 20, h-80, 0.38, mode_color)

    cv2.imshow("ASL Vision Pro — OpenCV", frame)

    # ── KEYBOARD CONTROLS ─────────────────────────────────────────────────────
    key = cv2.waitKey(1)
    if   key == 27:                        # ESC → quit
        break
    elif key == ord('c'):                  # C → clear sentence
        sentence = ""; last_prediction = ""
        write_sync("", "", "clear")
    elif key in (8, 127):                  # Backspace → delete last char
        if sentence:
            sentence = sentence[:-1]
            write_sync(sentence, "", "backspace")
    elif key == ord('g'):                  # G → toggle gender
        tts_config["gender"] = "female" if tts_config["gender"]=="male" else "male"
        reload_voice()
        write_sync(sentence, "", "config")
    elif key == ord('m'):                  # M → toggle speak mode
        tts_config["mode"] = "letter" if tts_config["mode"]=="end" else "end"
        write_sync(sentence, "", "config")
    elif key in (ord('+'), ord('=')):      # + → speed up
        tts_config["rate"] = min(260, tts_config["rate"] + 10)
        reload_voice()
    elif key == ord('-'):                  # - → slow down
        tts_config["rate"] = max(80,  tts_config["rate"] - 10)
        reload_voice()

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)