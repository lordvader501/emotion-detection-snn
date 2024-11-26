import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from .predict_emotion import predict_emotion
from constants import emotion_labels
from .emotion_stabalizer import stabilize_emotion


def detect_video(net):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image for preprocessing
        pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(pil_frame)

        # Predict the emotion
        emotion_id = predict_emotion(net, pil_frame)
        emotion_id = stabilize_emotion(emotion_id)
        # emotion = emotion_labels[emotion_id]
        if emotion_id is not None:
            emotion = emotion_labels[emotion_id]
            cv2.putText(frame, emotion, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the emotion on the frame
        # cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
        img = Image.open(file_path)
        emotion_id = predict_emotion(net, img, 0)
        print(emotion_id)
        emotion = emotion_labels[emotion_id]

        # Display the image and detected emotion
        img = cv2.imread(file_path)
        cv2.putText(img, f"Emotion: {emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
