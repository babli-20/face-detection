# Face Emotion Detection

This project utilizes OpenCV and DeepFace to perform real-time face detection and emotion recognition using a webcam.

## Features
- Detects faces in real-time using OpenCV's Haar Cascade classifier.
- Analyzes emotions using the DeepFace library.
- Displays the detected emotion on the video feed.

## Prerequisites
Ensure you have Python installed along with the required dependencies:

```bash
pip install opencv-python deepface
```

## Usage
Run the script to start real-time face detection and emotion analysis:

```bash
python face_emotion_detection.py
```

## How It Works
1. Captures video from the webcam.
2. Converts frames to grayscale for face detection.
3. Detects faces using OpenCV’s Haar Cascade classifier.
4. Extracts the detected face and analyzes emotions using DeepFace.
5. Displays the detected emotion in real-time.
6. Press 'q' to exit the application.

## Code Breakdown
```python
import cv2
from deepface import DeepFace

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        except Exception as e:
            print(f"Error analyzing face: {e}")

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Notes
- Make sure your webcam is properly connected.
- Install missing dependencies if the script throws an import error.
- Press 'q' to quit the application.

## License
This project is licensed under the MIT License.

