import cv2
import torch

from PIL import Image

from model.network import CSNN, FeedforwardSNN, SJCSNN
from pkg.predict_emotion import predict_emotion
from pkg.emotion_stabalizer import stabilize_emotion
from pkg.detection import detect_image, detect_video
# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = CSNN(beta=0.9).to(device)
net = SJCSNN(T=4).to(device)
net.load_state_dict(torch.load('sjcnn_final.pth', map_location=device))
# net = FeedforwardSNN().to(device)
# net.load_state_dict(torch.load('model/ffsnn.pth', map_location=device))
net.eval()  # Set the model to evaluation mode


# Define the emotion labels
# detect_image(net)
detect_video(net)
