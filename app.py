import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
# from utils import mediapipe_detection, landmarks_data, prob_viz

def mediapipe_detection(image, model):
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results
    else:
        return None, None

def landmarks_data(results):
    pose = np.array([[res.x, res.y] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)

    # face = np.array([[res.x, res.y, res.z] for res in
    #                  results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array([[res.x, res.y] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 2)
    rh = np.array([[res.x, res.y] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 2)

    return np.concatenate([pose, lh, rh])

def prob_viz(res, actions, input_frame):
    if res is not None:
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.putText(output_frame, f"{actions[num]} : {int(prob * 100)}% ", (10, 85 + num * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
        return output_frame

    else:
        output_frame = input_frame.copy()
        for num in range(len(actions)):
            prob = 0
            cv2.putText(output_frame, f"{actions[num]} : {int(prob * 100)}% ", (10, 85 + num * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
        return output_frame

# ---------------------------
# Load Model and Configurations
# ---------------------------
model = load_model('final_isl_model.keras')
actions = os.listdir('greetings_data')

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Streamlit UI
st.title("🤟 Sign Language Gesture Recognition")
st.write("Upload a video or use the preloaded sample video for gesture recognition.")

# File Uploader for Video
uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "mov", "avi"])
use_sample_video = st.checkbox("Use Sample Video (MVI_0042.MOV)")

# Define Threshold and Frame Settings
thresh = st.slider("Prediction Confidence Threshold", 0.5, 1.0, 0.85, 0.05)
frame_skip = st.slider("Frame Skip (Process every Nth frame)", 1, 10, 2)

# Button to Start Processing
if st.button("Start Gesture Detection"):
    stframe = st.empty()
    sentence = []
    sequence = []
    frame_count = 0
    
    # Choose video source
    if uploaded_file is not None:
        temp_video_path = "uploaded_video.mov"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_video_path)
    elif use_sample_video:
        cap = cv2.VideoCapture('MVI_0042.MOV')
    else:
        st.warning("Please upload a video or select the sample video.")
        st.stop()
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1

            if not ret:
                st.write("Video processing complete!")
                break

            if frame_count % frame_skip == 0:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                keypoints = landmarks_data(results)
                sequence.append(keypoints)

                # Display Predictions Safely
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    sequence = []

                    if res[np.argmax(res)] > thresh:
                        predicted_action = actions[np.argmax(res)]
                        if len(sentence) == 0 or predicted_action != sentence[-1]:
                            sentence.append(predicted_action)

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                # Ensure 'res' is defined before calling prob_viz
                if 'res' in locals() and res is not None:
                    image = prob_viz(res, actions, image)

                # Display Predictions on the Frame
                cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
                cv2.putText(image, ' '.join(sentence), (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display frame in Streamlit
                stframe.image(image, channels="BGR", use_column_width=True)
    
    cap.release()
    if uploaded_file is not None:
        os.remove(temp_video_path)
    st.success("Video processing completed!")

# Footer
st.write("Made with ❤️ by TEAM-ALPHA")
