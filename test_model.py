import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model('sign_language_model.h5')

labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[26] = 'Blank'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (512, 513))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            frame_resized_expanded = np.expand_dims(frame_resized, axis=0)

            predictions = model.predict(frame_resized_expanded)
            predicted_class = np.argmax(predictions)
            predicted_character = labels_dict[predicted_class]

            print(f"Predicted Character: {predicted_character}")  
            h, w, _ = frame.shape
            x1 = int(hand_landmarks.landmark[0].x * w)
            y1 = int(hand_landmarks.landmark[0].y * h)
            x2 = int(hand_landmarks.landmark[9].x * w)
            y2 = int(hand_landmarks.landmark[9].y * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Real-Time Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
