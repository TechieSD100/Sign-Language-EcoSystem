from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import HandTrackingModule as htm

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'L', 1: 'V', 2: 'I', 3: 'T', 4: 'G'}  # Modify this according to your classes

# Capture video
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Global variable to store the current prediction and finger count
current_prediction = None
totalFingers = 0

# Load overlay images for numbers
folderPath = "FingerImages"
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in os.listdir(folderPath) if imPath.endswith('.png') or imPath.endswith('.jpg')]

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.75)

def generate_frames():
    global current_prediction
    global totalFingers

    while True:
        success, frame = cap.read()
        if not success:
            break

        # MediaPipe hands detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        data_aux = [0] * 84  # 21 landmarks * 2 coordinates * 2 hands
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx > 1:  # We handle only up to 2 hands
                    break
                start_index = idx * 42  # 42 = 21 landmarks * 2 coordinates
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux[start_index + i * 2] = x
                    data_aux[start_index + i * 2 + 1] = y
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * frame.shape[1]) - 10
            y1 = int(min(y_) * frame.shape[0]) - 10
            x2 = int(max(x_) * frame.shape[1]) + 10
            y2 = int(max(y_) * frame.shape[0]) + 10

            # Predict the character
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            current_prediction = predicted_character

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Hand tracking module for finger counting
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)

        if len(lmList) != 0:
            try:
                fingers = detector.fingersUp(lmList)
                totalFingers = fingers.count(1)

                # Ensure totalFingers is within range of overlayList
                if 0 <= totalFingers <= len(overlayList):
                    h, w, c = overlayList[totalFingers - 1].shape
                    # frame[0:h, 0:w] = overlayList[totalFingers - 1]
            except Exception as e:
                print(f"Error processing hand landmarks: {e}")

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_finger_count', methods=['GET'])
def get_finger_count():
    return jsonify(totalFingers=totalFingers)

@app.route('/prediction')
def prediction():
    return jsonify(prediction=current_prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
