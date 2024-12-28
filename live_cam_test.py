import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter, deque

# Load the pre-trained gesture recognition model
model = load_model('models/m1-2.keras')

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Set up video capture
cap = cv2.VideoCapture(0) # use default cam (webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000) # frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000) # frame height
cap.set(cv2.CAP_PROP_FPS, 30) # frame rate

# Skip initial frames to stabilize the camera
for i in range(100):
    cap.read()

# Initialize deque for temporal smoothing of predictions
predictions = deque()
prediction_text = None # store prediction text for displaying

# Main loop for processing video frames
while cap.isOpened():
    ret, frame = cap.read() # capture frame from camara
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.flip(frame, 1) # Flip the frame horizontally for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB (required by MediaPipe)
    result = hands.process(rgb_frame) # Process the frame to detect hand landmarks

    h, w, _ = frame.shape
    black_frame = np.zeros((h, w), dtype=np.uint8)

    # If hand landmarks are detected, process them
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract and scale landmark points
            landmark_points = [
                (int(landmark.x * w), int(landmark.y * h))
                for landmark in hand_landmarks.landmark
            ]
            for x, y in landmark_points:
                cv2.circle(black_frame, (x, y), 10, (255), -1) # Draw landmarks

            # Draw connections between landmarks
            for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                cv2.line(black_frame, start_point, end_point, (255), 11)

            # Preprocess the black frame for model input
            input_image = cv2.resize(black_frame, (224, 224)) # resize to 224 x 224
            input_image = input_image.reshape((224, 224, 1)) # add channel dimention
            input_image = input_image / 255.0  # Normalize pixel values
            input_images = np.array([input_image])

            # Predict gesture and update deque
            mpredictions = model.predict(input_images)
            mpredicted_classes = np.argmax(mpredictions, axis=1) # Get predicted class
            predictions.append(mpredicted_classes[0])
            
            # Limit the size of the deque to the last 16 predictions
            if len(predictions) >= 16:
                predictions.popleft()

            # Get the most common prediction
            most_common_prediction = Counter(predictions).most_common(1)[0][0]
            prediction_text = f"Prediction: {most_common_prediction}" # formatting prediction text

    # Display prediction text on the frame
    if prediction_text:
        cv2.putText(
            black_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 255, 255), 2, cv2.LINE_AA
        )

    # display black frame with landmarks and predictions
    cv2.imshow("Camera Output - Adjust Your Hand Position", black_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
