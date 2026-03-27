import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Load the trained model
model = load_model('gesture_recognition_model.h5')

# Class names corresponding to your gestures
classNames = ["mom", "dada", "hello", "thank_you", "me", "tanu"]

# Initialize the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Read each frame from the webcam
        _, frame = cap.read()

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # Post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * frame.shape[1])
                    lmy = int(lm.y * frame.shape[0])
                    landmarks.append([lmx, lmy])

                # Predict gesture
                prediction = model.predict(np.expand_dims(landmarks, axis=0))
                classID = np.argmax(prediction)
                className = classNames[classID]

                # Drawing landmarks on frames with customized colors
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS,
                                    mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                    mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # Show the prediction on the frame with green text
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Keyboard Interrupt detected. Exiting gracefully...")

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
