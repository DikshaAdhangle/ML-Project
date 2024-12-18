import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Directory to store data
data_dir = 'collected_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define the labels for the gestures
labels = ['I','want','cup of tea','start','stop','my','name is' ,'Anjali','Have a good day!']
num_samples = 200  # Number of samples per gesture

# Create CSV file to store the data
with open(os.path.join(data_dir, 'hand_landmarks.csv'), mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in
                                                         range(21)]  # 21 landmarks with x, y coordinates
    csv_writer.writerow(header)

    for label in labels:
        print(f"Get ready to collect data for the '{label}' gesture.")
        input(f"Press Enter when you are ready to start collecting data for '{label}' gesture...")

        samples_collected = 0

        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    if len(landmarks) == 42:  # 21 landmarks * 2 (x, y)
                        # Write landmarks to CSV
                        csv_writer.writerow([label] + landmarks)
                        samples_collected += 1
                        print(f"Samples collected for '{label}': {samples_collected}/{num_samples}")

                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the frame
            cv2.imshow(f'Collecting data for {label}', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"Finished collecting data for '{label}' gesture.\n")

cap.release()
cv2.destroyAllWindows()

