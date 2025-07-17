import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

def detect_sign_language():
    print("[INFO] Loading model...")
    try:
        with open('model.p', 'rb') as f:
            model_dict = pickle.load(f)
        model = model_dict['model']
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        return f"[ERROR] Could not load model: {e}"

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

    labels_dict = {
        0: 'LA', 1: 'LB', 2: 'LC', 3: 'LD', 4: 'LE', 5: 'LF', 6: 'LG',
        7: 'RA', 8: 'RB', 9: 'RC', 10: 'RD', 11: 'RE', 12: 'RF', 13: 'RG'
    }

    predicted_labels = deque(maxlen=10)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return "[ERROR] Could not open webcam."

    print("[INFO] Starting video stream. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data_aux, x_, y_ = [], [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                except Exception as e:
                    predicted_character = "?"

                predicted_labels.append(predicted_character)

                #  Write the latest prediction to a file
                with open("latest_prediction.txt", "w") as f:
                    f.write(predicted_character)

                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10
                x2 = int(max(x_) * frame.shape[1]) + 10
                y2 = int(max(y_) * frame.shape[0]) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        current_text = "Current: " + (predicted_labels[-1] if predicted_labels else "None")
        cv2.putText(frame, current_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Sign Language Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)
    return "Captured Predictions: " + " ".join(predicted_labels)

#  Fix: use correct __name__ syntax and actually call the function
if __name__ == "__main__":
    result = detect_sign_language()
    print(result)
