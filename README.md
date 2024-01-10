**# Lane-Detection** 
This Python script uses the OpenCV (cv2) library for computer vision tasks, specifically for lane detection in a video.
 The goal is to process each frame of the given video file to identify and highlight the lanes on a road.
 Here's an explanation of what happens as we step through the code: 1.
 Import Libraries: - `cv2`: This is an alias for the OpenCV (Open Source Computer Vision Library), which provides tools for image and video processing.
 - `numpy`: A powerful numerical library for handling arrays, mathematics, and more.
 - Define helper functions: - `region_of_interest(img, vertices)`: Takes an image and a set of polygon vertices as input and creates a mask that defines a region of interest.
 Only this defined area of the image will be considered when processing it further.
 - Initializes a black mask with the same dimensions as the input `img`.
 - Fills the polygon specified by `vertices` with white color on the mask.
 - Applies bitwise AND operation between the original `img` and this mask so only parts of the image within this white filled region remain.
 - `draw_lines(img, lines)`: Takes an image and an array of line coordinates.
 For each line coordinate: - Extracts start `(x1, y1)` and end `(x2, y2)` positions.
 - Draws a blue (255, 0, 0 in BGR color space) line of thickness 3 between these two points onto the input image.
 - Define `process_image(image)` function: Performs actual lane detection steps on a given frame (`image`) received from a video feed or any other source.
 - Retrieves height and width of images using
 The code attempts to detect and draw lines on the lanes of a video, using computer vision techniques such as Canny edge detection and Hough line transformation.**
