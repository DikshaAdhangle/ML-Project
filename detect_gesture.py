import cv2
import mediapipe as mp
import numpy as np
import joblib
import requests
import os
import time
import google.generativeai as genai

class GestureDetectionApp:
    def __init__(self):
        # Initialize MediaPipe hands and OpenCV
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = None
        self.running = False

        # Load your gesture recognition model
        self.model = joblib.load('gesture_model.pkl')

        # Gesture-related variables
        self.detected_words = []  # Store words to form a sentence
        self.last_detected_gesture = None
        self.sentence_started = False

        # Configure the Gemini API
        self.gemini_api_key = "AIzaSyCg29-Ppsdh-Te0EWAIRnRv2V20E5Dr2mY"  # Your API key
        genai.configure(api_key=self.gemini_api_key)  # Configure with the API key

        # Sentence display properties
        self.labels = ['I', 'want', 'cup', 'of', 'tea', 'hi', 'my', 'name', 'is', 'Anjali', 'start', 'stop']  # Gesture labels
        self.max_sentence_length = 10  # Limit sentence length

        # Add a variable for time tracking
        self.last_gesture_time = time.time()  # Initialize the last gesture time to now

    def predict_gesture(self, landmarks):
        """
        Takes hand landmarks as input and predicts the corresponding gesture.
        """
        landmarks = np.array(landmarks).reshape(1, -1)  # Reshape input for model
        prediction = self.model.predict(landmarks)
        return prediction[0]

    def correct_sentence(self, words):
        """
        Use the Gemini API to construct a meaningful and grammatically correct sentence using
        the detected words, allowing minimal additions for structure.
        """
        # Construct a specific prompt to guide the API in forming a meaningful sentence
        prompt_text = f"Using only these words: {' '.join(words)}, create a coherent and grammatically correct sentence. " \
                      "Feel free to add necessary articles or prepositions, but keep the main words as they are."

        try:
            # Generate the content using the Gemini API with the specified prompt
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt_text)

            # Extract and return the corrected sentence
            corrected_sentence = response.text.strip()
            return corrected_sentence

        except Exception as e:
            print(f"API Error: {e}")
            # Fallback to join words if API call fails
            return ' '.join(words)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)

            while self.cap.isOpened() and self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(frame_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Collect hand landmarks
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append(lm.x)
                            landmarks.append(lm.y)

                        if len(landmarks) == 42:  # 21 landmarks * 2 (x, y)
                            current_time = time.time()
                            # Check if 2 seconds have passed since the last detected gesture
                            if current_time - self.last_gesture_time >= 2:  # 2 second pause
                                # Predict gesture
                                gesture = self.predict_gesture(landmarks)
                                print(f"Detected Gesture: {gesture}")  # Debug print to check detection

                                if gesture == 'start':
                                    self.detected_words = []  # Reset sentence
                                    self.sentence_started = True
                                    print("Sentence capture started.")
                                elif gesture == 'stop' and self.sentence_started:
                                    # Display the complete detected sentence
                                    print(f"Words Detected: {self.detected_words}")

                                    # Correct the sentence using Gemini API
                                    corrected_sentence = self.correct_sentence(self.detected_words)
                                    print(f"Corrected Sentence: {corrected_sentence}")

                                    # Display the corrected sentence on a new window
                                    corrected_frame = np.zeros((200, 600, 3), np.uint8)
                                    cv2.putText(corrected_frame, corrected_sentence, (10, 100),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                    cv2.imshow('Corrected Sentence', corrected_frame)

                                    cv2.waitKey(2000)  # Display the corrected sentence for 2 seconds

                                    self.sentence_started = False  # Stop sentence capturing
                                    self.detected_words.clear()  # Clear words after processing
                                elif self.sentence_started and gesture != 'start' and gesture != 'stop':
                                    # Add the detected word to the sentence
                                    if gesture not in self.detected_words and len(self.detected_words) < self.max_sentence_length:
                                        self.detected_words.append(gesture)
                                    elif gesture in self.detected_words:
                                        print(f"Gesture '{gesture}' already detected, skipping.")

                                # Update the time of the last gesture detection
                                self.last_gesture_time = current_time

                        # Draw hand landmarks on the frame
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Display the detected words in real-time on a new window
                sentence_display = ' '.join(self.detected_words)
                if sentence_display:
                    live_frame = np.zeros((400, 1000, 3), np.uint8)
                    cv2.putText(live_frame, sentence_display, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Live Words', live_frame)

                # Display the camera feed
                cv2.imshow('Real-Time Gesture Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the camera and close windows when detection stops
            self.cap.release()
            cv2.destroyAllWindows()

    def stop_detection(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# Create and start the application
if __name__ == "__main__":
    app = GestureDetectionApp()
    app.start_detection()
