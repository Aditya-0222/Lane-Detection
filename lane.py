import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)

def process_image(image):
    height = image.shape[0]
    width = image.shape[1]

# Define region of interest (polygonal mask for the lanes)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = np.zeros_like(image)
    if lines is not None:
        draw_lines(line_image, lines)
    
    return cv2.addWeighted(image, 0.8, line_image, 1, 1)

# Read the input video file
cap = cv2.VideoCapture("C:\\Users\\Aditya Thakur\\Dropbox\\PC\\Downloads\\driving_-_800 (360p).mp4")

while cap.isOpened():
    ret, frame = cap.read()
    frame = process_image(frame)
    cv2.imshow('Lane Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
