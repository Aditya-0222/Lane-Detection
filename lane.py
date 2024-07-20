import cv2
import numpy as np

def apply_mask(image, vertices):
    """
    Apply an image mask. Only the region inside the polygon formed
    by the vertices will be included in the mask.
    """
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=3):
    """
    Draw lane lines on the image.
    """
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def process_frame(frame):
    """
    Process a single frame of the video to detect lane lines.
    """
    height, width = frame.shape[:2]

    # Define the region of interest vertices
    roi_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detector
    edges = cv2.Canny(gray_frame, 100, 200)

    # Mask the edges image
    masked_edges = apply_mask(edges, np.array([roi_vertices], np.int32))

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=2, 
        theta=np.pi / 180, 
        threshold=100, 
        minLineLength=40, 
        maxLineGap=5
    )

    # Create an image to draw lines on
    line_image = np.zeros_like(frame)

    # Draw the lines on the line image
    line_image = draw_lane_lines(line_image, lines)

    # Combine the original frame with the line image
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined_image

# Read the input video file
video_path = "C:\\Users\\Aditya Thakur\\Dropbox\\PC\\Downloads\\driving_-_800 (360p).mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Lane Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
