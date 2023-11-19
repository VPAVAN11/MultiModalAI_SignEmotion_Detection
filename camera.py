import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('emotion_detection_model.h5')

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection using MediaPipe
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            ih, iw, _ = frame.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract the face ROI for emotion detection
            face_roi = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi_gray_resized = cv2.resize(face_roi_gray, (48, 48))
            face_roi_resized = np.expand_dims(np.expand_dims(face_roi_gray_resized, -1), 0)

            # Predict emotion using the loaded model
            predicted_emotions = model.predict(face_roi_resized)
            emotion_label = np.argmax(predicted_emotions)
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            emotion = emotions[emotion_label]

            # Display emotion text and bounding box on the frame
            cv2.putText(frame, emotion, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

    # Show the frame with emotion detection in real-time
    cv2.imshow('Emotion Detection', frame)

    # Check for the 'Esc' key press to quit
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
