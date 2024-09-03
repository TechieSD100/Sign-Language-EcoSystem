from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np

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
labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Modify this according to your classes

# Capture video
cap = cv2.VideoCapture(0)

# Global variable to store the current prediction
current_prediction = None

def generate_frames():
    global current_prediction
    while True:
        data_aux = [0] * 84  # 21 landmarks * 2 coordinates * 2 hands
        x_ = []
        y_ = []

        success, frame = cap.read()
        if not success:
            break
        else:
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

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

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                current_prediction = predicted_character  # Update the current prediction

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify(prediction=current_prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
