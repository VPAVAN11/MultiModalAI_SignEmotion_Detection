import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd

model = load_model('smnist.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    framergbanalysis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultanalysis = hands.process(framergbanalysis)
    hand_landmarksanalysis = resultanalysis.multi_hand_landmarks

    if hand_landmarksanalysis:
        for handLMsanalysis in hand_landmarksanalysis:
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
                prediction = model.predict(analysisframe)
                predclass = np.argmax(prediction)
                confidence = np.max(prediction)

                predicted_sign = chr(predclass + 65)  # Convert predicted class to character
                print("Predicted Character: ", predicted_sign)
                print('Confidence: ', confidence * 100)

                cv2.putText(frame, f"Sign: {predicted_sign} Confidence: {confidence * 100:.2f}%", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
