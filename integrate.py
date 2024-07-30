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

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2) 
    ax.set_aspect('equal')
    plt.show()

def find_best_rectangle(contours):
    best_rect = None
    min_diff = float('inf')
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # Check if the approximated contour has 4 vertices
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            # Calculate the difference between the contour area and the rectangle area
            contour_area = cv2.contourArea(contour)
            rect_area = cv2.contourArea(box)
            area_diff = abs(contour_area - rect_area)
            if area_diff < min_diff:
                min_diff = area_diff
                best_rect = box
    return best_rect

def regularize_corners(corners):
    rect = cv2.minAreaRect(np.array(corners))
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    return box

def detect_circles(paths_XYs):
    # Create a blank image
    img_size = 1000  # Adjust size as needed
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Draw the paths on the image
    for XYs in paths_XYs:
        for XY in XYs:
            for x, y in XY:
                cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 2000,
                              param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        detected_circles = np.uint16(np.around(circles))
        for (x, y, r) in detected_circles[0, :]:
            cv2.circle(img, (x, y), r, (0, 0, 255), 3)
            cv2.circle(img, (x, y), 2, (0, 255, 255), 3)
        return True, img
    else:
        return False, img

def detect_lines(paths_XYs):
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

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("Gaussian blur applied.")

    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    print("Edge detection applied.")

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

    # Draw the lines on the image
    if filtered_lines:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return filtered_lines, img

# Main pipeline function
def main_pipeline(csv_path):
    paths_XYs = read_csv(csv_path)
    plot(paths_XYs)
    if paths_XYs is None:
        raise ValueError("Paths not found or unable to load.")
    print("Paths loaded successfully.")

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

    # Detect and regularize rectangle
    best_rect = find_best_rectangle(contours)
    if best_rect is not None:
        print("Best rectangle found.")
        regularized_corners = regularize_corners(best_rect)
        cv2.drawContours(img, [regularized_corners], 0, (0, 255, 0), 5)
        print("Regularized rectangle drawn.")
    else:
        print("No rectangle found.")

    # Detect lines
    filtered_lines, lines_img = detect_lines(paths_XYs)
    if filtered_lines:
        print("Lines detected and drawn.")
        img = cv2.addWeighted(img, 1, lines_img, 1, 0)
    else:
        print("No lines detected.")

    # Detect circles
    circle_detected, circles_img = detect_circles(paths_XYs)
    if circle_detected:
        print("Circle detected and drawn.")
        img = cv2.addWeighted(img, 1, circles_img, 1, 0)
    else:
        print("No circle detected.")

    # Save and show the result
    cv2.imwrite('detected_shapes.png', img)
    print("Result saved as 'detected_shapes.png'.")
    cv2.imshow('Detected Shapes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
csv_path = 'problems/isolated.csv'
main_pipeline(csv_path)
