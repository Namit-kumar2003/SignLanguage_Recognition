import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands


class HandDetector:
    def __init__(self, max_hands=1):
        self.hands = mp_hands.Hands(
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
       
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        h, w, _ = frame.shape

        # Compute bounding box
        x_vals, y_vals = [], []
        for lm in result.multi_hand_landmarks[0].landmark:
            x_vals.append(int(lm.x * w))
            y_vals.append(int(lm.y * h))

        x1, x2 = max(min(x_vals) - 30, 0), min(max(x_vals) + 30, w)
        y1, y2 = max(min(y_vals) - 30, 0), min(max(y_vals) + 30, h)

        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
