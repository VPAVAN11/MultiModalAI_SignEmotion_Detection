import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time

# Load the pre-trained emotion detection model
emotion_model = load_model('emotion_detection_model.h5')

# Load the pre-trained sign language detection model
sign_model = load_model('smnist.h5')

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe for hand tracking
mphands = mp.solutions.hands
hands = mphands.Hands()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(rgb_frame)

    # Perform hand tracking using MediaPipe
    hand_results = hands.process(rgb_frame)

    if face_results.detections:
        for detection in face_results.detections:
            ih, iw, _ = frame.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract the face ROI for emotion detection
            face_roi = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi_gray_resized = cv2.resize(face_roi_gray, (48, 48))
            face_roi_resized = np.expand_dims(np.expand_dims(face_roi_gray_resized, -1), 0)

            # Predict emotion using the loaded emotion detection model after a 1-second delay
            predicted_emotions = emotion_model.predict(face_roi_resized)
            emotion_label = np.argmax(predicted_emotions)
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            emotion = emotions[emotion_label]

            # Display emotion text and bounding box on the frame for emotion detection
            cv2.putText(frame, emotion, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

    if hand_results.multi_hand_landmarks:
        for handLMsanalysis in hand_results.multi_hand_landmarks:
            # Extract hand region for sign language detection
            x_max = 0
            y_max = 0
            x_min = frame.shape[1]
            y_min = frame.shape[0]
            for lmanalysis in handLMsanalysis.landmark:
                x, y = int(lmanalysis.x * frame.shape[1]), int(lmanalysis.y * frame.shape[0])
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20

            analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]

            # Check if analysisframe is valid before resizing
            if analysisframe.shape[0] > 0 and analysisframe.shape[1] > 0:
                analysisframe = cv2.resize(analysisframe, (28, 28))
                analysisframe = analysisframe.reshape(-1, 28, 28, 1)  # Reshape for model input

                analysisframe = analysisframe / 255.0  # Normalize data
                prediction = sign_model.predict(analysisframe)
                predclass = np.argmax(prediction)
                confidence = np.max(prediction)

                predicted_sign = chr(predclass + 65)  # Convert predicted class to character
                print("Predicted Character: ", predicted_sign)
                print('Confidence: ', confidence * 100)

                # Display sign language prediction and bounding box
                cv2.putText(frame, f"Sign: {predicted_sign} Confidence: {confidence * 100:.2f}%", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Combined Detection", frame)

    # Check for the 'Esc' key press to quit
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
