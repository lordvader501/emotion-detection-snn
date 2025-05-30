from collections import deque


prediction_queue = deque(maxlen=10)  # Keep predictions from the last 10 frames


def stabilize_emotion(predicted_emotion):
    prediction_queue.append(predicted_emotion)
    # Find the most common prediction in the queue
    stabilized_emotion = max(set(prediction_queue), key=prediction_queue.count)
    return stabilized_emotion
