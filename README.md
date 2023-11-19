**MultiModalAI_SignEmotion_Detection**

**Description:**
This project combines computer vision techniques and deep learning models to enable simultaneous detection of sign language gestures and recognition of emotions from facial expressions in real-time using a webcam. The integration leverages MediaPipe's Face Detection and Hand Tracking modules for emotion detection and sign language recognition, respectively. The project integrates two pre-trained models: one for emotion detection and another for sign language recognition, allowing users to observe and analyze both modalities concurrently.

**Features:**
Emotion Detection: Utilizes a pre-trained deep learning model to detect emotions (e.g., happy, sad, angry) from facial expressions in real time.
Sign Language Recognition: Detects and interprets hand gestures, recognizing sign language alphabets using a separate pre-trained model.
Simultaneous Analysis: Integrates the capabilities of both emotion detection and sign language recognition in a single application using MediaPipe's face and hand tracking functionalities.
Live Visualization: Provides a real-time display of detected emotions and recognized sign language gestures overlaid on the webcam feed.

**Requirements:**
Python 3.8
OpenCV
Mediapipe
Keras with TensorFlow backend

**Usage:**
Clone the repository.
Install the required dependencies using pip install -r requirements.txt.
Run the MultiModalAI_SignEmotion_Detection.py script to launch the application.
Interact with the webcam and observe simultaneous emotion and sign language recognition.


