import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from .predict_emotion import predict_emotion
from constants import emotion_labels
from .emotion_stabalizer import stabilize_emotion
import numpy as np
cascade = cv2.CascadeClassifier('face_cascade.xml')


def detect_video(net):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_cam = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = cascade.detectMultiScale(
            gray_cam, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop the face region
            face_region = frame[y:y + h + 50, x:x + w]
            face_region = np.expand_dims(np.expand_dims(
                cv2.resize(face_region, (48, 48)), -1), 0)
            # cv2.imshow('Live Emotion Detection2', face_region)
            # Convert the face region to PIL Image for preprocessing
            pil_face = Image.fromarray(
                cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))

            # Predict the emotion
            emotion_id = predict_emotion(net, pil_face)
            # emotion_id = stabilize_emotion(emotion_id)

            if emotion_id is not None:
                emotion = emotion_labels[emotion_id]

                # Draw rectangle around the face and display the emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('Live Emotion Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_image(net):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        # Load the image
        img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = cascade.detectMultiScale(
            gray_img, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop the face region
            face_region = img[y:y + h, x:x + w]
            face_region = np.expand_dims(np.expand_dims(
                cv2.resize(face_region, (48, 48)), -1), 0)
            # Convert the face region to PIL Image for preprocessing
            pil_face = Image.fromarray(
                cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))

            # Predict the emotion
            emotion_id = predict_emotion(net, pil_face)
            emotion = emotion_labels[emotion_id]

            # Draw rectangle around the face and display the emotion
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f"Emotion: {emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image with detected emotions
        cv2.imshow("Emotion Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
