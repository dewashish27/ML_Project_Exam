import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import serial


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
ser = serial.Serial('COM8', 9600)
# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    gesture = ''
    num_fingers = 0

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Count the number of fingers detected
            if len(landmarks) > 0:
                if landmarks[4][0] < landmarks[3][0]:
                    num_fingers += 1
                if landmarks[8][1] < landmarks[6][1]:
                    num_fingers += 1
                if landmarks[12][1] < landmarks[10][1]:
                    num_fingers += 1
                if landmarks[16][1] < landmarks[14][1]:
                    num_fingers += 1
                if landmarks[20][1] < landmarks[18][1]:
                    num_fingers += 1
            ser.write(b'0')
            # Determine the gesture based on the number of fingers detected
            if num_fingers == 1:
                gesture = 'Forward'
                ser.write(b'1')
            elif num_fingers == 2:
                gesture = 'Backward'
                ser.write(b'2')
            elif num_fingers == 3:
                gesture = 'Right'
                ser.write(b'3')
            elif num_fingers == 4:
                gesture = 'Left'
                ser.write(b'4')
            elif num_fingers == 5:
                gesture = 'Fast'
                ser.write(b'5')
            else:
                gesture = 'Stop'
                ser.write(b'0')

    # Show the gesture on the frame
    cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
    # Show the number of fingers on the frame
    cv2.putText(frame, f"Number of Fingers: {num_fingers}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
    print(num_fingers)
    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break
ser.close()
# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
