import cv2 as cv2
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

# Read image
path_XYs = read_csv('problems/problems/frag0.csv')
if path_XYs is None:
    raise ValueError("Image not found or unable to load.")
print("Image loaded successfully.")

# Create a blank image
img_size = 1000  # Adjust size as needed
img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

# Draw the paths on the image
for XYs in path_XYs:
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

# Find the best rectangle
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

    # Ensure the rectangle is within the image bounds
    top_left = np.maximum(top_left, [0, 0])
    bottom_right = np.minimum(bottom_right, [img_size, img_size])

    # Create a new blank image to draw only the rectangle
    rect_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Draw the rectangle using cv2.rectangle
    cv2.rectangle(rect_img, tuple(top_left), tuple(bottom_right), (0, 255, 0), 5)
    print("Regularized rectangle drawn.")

    # Save and show the result
    cv2.imwrite('regularized_rectangle_only.png', rect_img)
    print("Result saved as 'regularized_rectangle_only.png'.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()





