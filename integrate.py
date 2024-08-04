import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

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

# Function to regularize the corners to form a perfect rectangle
def regularize_corners(corners):
    rect = cv2.minAreaRect(np.array(corners))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

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

def detect_lines(paths_XYs, gray, edges, img):  
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("Gaussian blur applied.")

    # Detect lines using Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
    print(f"Found {len(lines)} lines." if lines is not None else "No lines found.")

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
    print(f"Filtered to {len(filtered_lines)} lines.")

    return filtered_lines, img

# Main pipeline function
def main_pipeline(csv_path):
    paths_XYs = read_csv(csv_path)
    if paths_XYs is None:
        raise ValueError("Image not found or unable to load.")
    print("Image loaded successfully.")

    # Create a blank image
    img_size = 1000  # Adjust size as needed
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Draw the paths on the image
    for XYs in paths_XYs:
        for XY in XYs:
            for x, y in XY:
                cv2.rectangle(img, (int(x), int(y)), (int(x+2), int(y+2)), (255, 255, 255), -1)


    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale.")

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    print("Edge detection applied.")

    # Find contours of the irregular shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")

    # Create a new blank image to draw only the rectangle
    rect_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    best_rect = find_best_rectangle(contours)
    if best_rect is None:
        print("No rectangle found.")
    else:
        print("Best rectangle found.")
        # Regularize the corners
        regularized_corners = regularize_corners(best_rect)
        # Extract top-left and bottom-right corners
        top_left = np.min(regularized_corners, axis=0)
        bottom_right = np.max(regularized_corners, axis=0)

        # Ensure the rectangle is within the image bounds and adjust for thickness
        thickness = 1
        top_left = np.maximum(top_left, [0, 0])
        bottom_right = np.minimum(bottom_right, [img_size - 1, img_size - 1])
        # Draw the rectangle using cv2.rectangle
        cv2.rectangle(rect_img, tuple(top_left), tuple(bottom_right), (0, 255, 0), thickness)
        print("Regularized rectangle drawn.")

    # Detect circles
    circle_detected = detect_circles(img, gray)
    if circle_detected is not None:
        detected_circles = np.uint16(np.around(circle_detected))
        for (x, y, r) in detected_circles[0, :]:
            # Draw the regularized circle boundary
            cv2.circle(rect_img, (x, y), r, (0, 255, 255), 1)
            # Mask to retain pixels inside the unregularized circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)
            img[mask == 0] = 0
            # Copy the contents of the unregularized circle to the regularized circle
            circle_contents = cv2.bitwise_and(img, img, mask=mask)
            rect_img = cv2.bitwise_or(rect_img, circle_contents)
    else:
        print("No circle detected.")
        
    # Draw stars
    best_star = find_best_star(contours)
    if best_star is None:
        print("No star shape found.")
    else:
        print("Best star shape found.")
        cv2.drawContours(rect_img, [best_star], -1, (0, 255, 0), 1)
        print("Star shape drawn.")
    
    # Combine the images
    combined_img = cv2.addWeighted(img, 1, rect_img, 1, 0)

    # Save and show the result
    cv2.imwrite('detected_shapes.png', combined_img)
    print("Result saved as 'detected_shapes.png'.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
csv_path = 'problems/frag0.csv'
main_pipeline(csv_path)


