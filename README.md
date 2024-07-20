Libraries Used:

OpenCV (cv2): Tools for image and video processing.
NumPy: Numerical library for handling arrays and mathematical operations.
Functions Defined:

apply_mask(img, vertices): Creates a mask to define a region of interest in the image.

Operations:
Initializes a black mask with dimensions of the input image.
Fills a polygon specified by vertices with white color on the mask.
Applies a bitwise AND operation between the original image and the mask to retain only the defined area.
draw_lines(img, lines): Draws lines on the image using provided coordinates.

Operations:
Extracts start (x1, y1) and end (x2, y2) positions for each line.
Draws blue lines (BGR color space: 255, 0, 0) of thickness 3 between these points.
Process Image Function:

process_image(image): Performs lane detection on a video frame.
Steps:
Converts the frame to grayscale.
Applies Canny edge detection (thresholds: 100, 200).
Defines region of interest vertices: bottom-left (0, height), top-middle (width/2, height/2), bottom-right (width, height).
Masks edges image within the region of interest.
Detects lines using Hough Transform (parameters: rho=2, theta=π/180, threshold=100, minLineLength=40, maxLineGap=5).
Draws detected lines on the frame.
Combines the original frame with the line image (weight: original frame 0.8, line image 1, scalar 1).
Video Processing:

Reads input video from the specified path.
Processes each frame to detect and highlight lane lines.
Displays processed frames in real-time.
Exits on pressing 'q'.
Numerical Values:

Canny Edge Detection Thresholds: 100, 200.
Hough Transform Parameters:
Rho: 2
Theta: π/180
Threshold: 100
Min Line Length: 40
Max Line Gap: 5
Line Color and Thickness: Blue (255, 0, 0), Thickness: 3
Video Frame Weighting: Original frame 0.8, Line image 1, Scalar 1.
