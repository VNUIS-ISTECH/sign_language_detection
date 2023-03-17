from flask import Flask, render_template, Response, request, url_for
import cv2
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from sklearn.datasets import make_classification
from tensorflow.keras.models import load_model


app = Flask(__name__)
cap = cv2.VideoCapture(0)

global switch
switch = 1

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def style_of_landmarks(image, results):
    # draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                         mp_drawing.DrawingSpec(color=(80,50,10), thickness=1,circle_radius=1),
    #                       mp_drawing.DrawingSpec(color=(80,50,10),thickness=1, circle_radius=1))
    # draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1))
    # draw left connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1))
    # draw right connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1))

def extract_key_points(results):
    # take all marks possitions
    # and put into an list of array
    # these pose will help for the action detection
    poses_list = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    left_hand = np.array([[i.x, i.y, i.z] for i in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[i.x, i.y, i.z] for i in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([poses_list, left_hand, right_hand])

# Setup Folders for Collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello','iloveyou', 'ok'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


def generate_frames():
    while True:

        ## read the camera frame
        success, frame = cap.read()
        if not success:
            break
        else:
            model = load_model('action.h5')
            sequence = []
            sentence = []
            threshold = 0.8
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = detection(frame, holistic)
                    print(results)

                    # Draw landmarks
                    print("draw land")
                    style_of_landmarks(image, results)

                    # 2. Prediction logic
                    keypoints = extract_key_points(results)
                    #         sequence.insert(0,keypoints)
                    #         sequence = sequence[:30]
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(actions[np.argmax(res)])

                        # 3. Viz logic
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        # Viz probabilities
                        image = prob_viz(res, actions, image, colors)

                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    ret, buffer = cv2.imencode('.jpg', image)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()




@app.route('/')
def index():
    return render_template('base.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,cap
    if request.method == 'POST':
        
        if  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                cap.release()
                cv2.destroyAllWindows()
                
            else:
                cap = cv2.VideoCapture(0)
                switch=1
    elif request.method=='GET':
        return render_template('base.html')
    return render_template('base.html')


if __name__ == "__main__":
    app.run(debug=True)


cap.release()
cv2.destroyAllWindows()