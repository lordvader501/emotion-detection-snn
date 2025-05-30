import torch
from torchvision import transforms
import torch.nn.functional as F
from constants import emotion_labels


def preprocess_image(image):
    """
    Preprocess the image: Grayscale, Resize, Normalize
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(1)  # Add batch dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return image.to(device)


def predict_emotion(net, frame, threshold=0.1):
    """
    Predict the emotion of the given frame.
    """
    with torch.no_grad():
        # processed_frame = preprocess_image(frame)
        # outputs, _, _ = net(processed_frame)
        # _, predicted = torch.max(outputs, 1)
        # return predicted.item()
        processed_frame = preprocess_image(frame)
        outputs = net(processed_frame)
        probs = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probs, 1)
        print(emotion_labels[predicted.item()], max_prob.item())
        # Only return the emotion if the confidence is above the threshold
        if max_prob.item() > threshold:
            return predicted.item()
        else:
            return None  # No prediction if confidence is too low
