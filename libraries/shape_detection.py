import cv2
import numpy as np

from libraries.rdp import straightenLine

# Function to approximate contours to polygons and find the best star shape
def find_best_star(contours):
    best_star = None
    min_diff = float('inf')
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 10:  # Check if the approximated contour has 10 vertices
            star_area = cv2.contourArea(approx)
            contour_area = cv2.contourArea(contour)
            area_diff = abs(contour_area - star_area)
            if area_diff < min_diff:
                min_diff = area_diff
                best_star = approx
    return best_star

# Function to approximate contours to polygons and find the best rectangle
def find_best_rectangle(contours):
    best_rect = None
    min_diff = float('inf')
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # Check if the approximated contour has 4 vertices
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # Calculate the difference between the contour area and the rectangle area
            contour_area = cv2.contourArea(contour)
            rect_area = cv2.contourArea(box)
            area_diff = abs(contour_area - rect_area)
            if area_diff < min_diff:
                min_diff = area_diff
                best_rect = box
    return best_rect

# Function to identify circles in the image
def detect_circles(img, gray):    
    # Convert to grayscale
    grayC = cv2.medianBlur(gray, 5)

    # Detect circles
    circles = cv2.HoughCircles(grayC, cv2.HOUGH_GRADIENT, 1, 2000,
                              param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        return circles
    else:
        return None

# Function to identify straight lines in the image
def detect_lines(paths_XYs, gray, edges, img):  
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect lines using Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)

    # Filter lines based on distance
    def filter_lines(lines, min_distance=10):
        if lines is None:
            return []
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            keep = True
            for f_line in filtered_lines:
                fx1, fy1, fx2, fy2 = f_line[0]
                if np.sqrt((x1 - fx1)**2 + (y1 - fy1)**2) < min_distance and np.sqrt((x2 - fx2)**2 + (y2 - fy2)**2) < min_distance:
                    keep = False
                    break
            if keep:
                filtered_lines.append(line)
        return filtered_lines
    
    filtered_lines = filter_lines(lines)

    starighten_lines = straightenLine(filtered_lines)
    return filtered_lines, img