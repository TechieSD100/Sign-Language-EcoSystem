import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# 21 landmarks per hand, 2 coordinates (x, y) per landmark, for up to 2 hands
num_landmarks = 21 * 2 * 2

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = [0] * num_landmarks  # Initialize with zeros
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx > 1:  # We handle only up to 2 hands (left and right)
                    break
                start_index = idx * 42  # 42 = 21 landmarks * 2 coordinates
                for i in range(21):
                    data_aux[start_index + i * 2] = hand_landmarks.landmark[i].x
                    data_aux[start_index + i * 2 + 1] = hand_landmarks.landmark[i].y

            data.append(data_aux)
            labels.append(dir_)

# Save the dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
