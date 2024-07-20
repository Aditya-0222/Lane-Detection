import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained lane detection model
model = tf.keras.models.load_model('path/to/your/lane_detection_model.h5')

def preprocess_image(img):
    """Preprocess the input image for the model."""
    img = cv2.resize(img, (256, 256))  # Resize to the input size expected by the model
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess_output(output, original_size):
    """Convert model output to lane lines on the original image."""
    output = np.squeeze(output)  # Remove batch dimension
    output = (output > 0.5).astype(np.uint8)  # Threshold to binary
    output = cv2.resize(output, original_size)  # Resize back to original size
    return output

def traditional_lane_detection(frame):
    """Traditional lane detection using OpenCV."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    roi = edges[int(height/2):, :]  # Region of interest

    vertices = np.array([[(0, height), (width, height), (width, int(height/2)), (0, int(height/2))]], dtype=np.int32)
    mask = np.zeros_like(roi)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(roi, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1 + int(height/2)), (x2, y2 + int(height/2)), (0, 255, 0), 2)

    return frame

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_height, original_width, _ = frame.shape

    # Apply traditional lane detection
    traditional_lanes = traditional_lane_detection(frame.copy())

    # Preprocess for deep learning model
    preprocessed_frame = preprocess_image(frame)
    
    # Predict lane markings using the deep learning model
    predictions = model.predict(preprocessed_frame)
    
    # Post-process the output
    lane_mask = postprocess_output(predictions, (original_width, original_height))
    
    # Combine results
    frame[lane_mask > 0] = [0, 255, 0]  # Deep learning lane markings in green
    frame = cv2.addWeighted(frame, 0.7, traditional_lanes, 0.3, 0)  # Blend traditional lanes with model output

    cv2.imshow('Lane Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
