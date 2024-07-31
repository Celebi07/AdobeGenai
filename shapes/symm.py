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

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2) 
    ax.set_aspect('equal')
    plt.show()

# Function to approximate contours to polygons and find the best rectangle
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

# Function to regularize the corners to form a perfect rectangle
def regularize_corners(corners):
    rect = cv2.minAreaRect(np.array(corners))
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    return box

# Function to draw symmetry lines within the bounding box
def draw_symmetry_lines(img, corners):
    # Calculate the center of the rectangle
    center_x = int(np.mean(corners[:, 0]))
    center_y = int(np.mean(corners[:, 1]))

    # Find the bounding box limits
    min_x = np.min(corners[:, 0])
    max_x = np.max(corners[:, 0])
    min_y = np.min(corners[:, 1])
    max_y = np.max(corners[:, 1])

    # Draw vertical and horizontal lines within the bounding box
    cv2.line(img, (center_x, min_y), (center_x, max_y), (0, 0, 255), 2)  # Vertical line
    cv2.line(img, (min_x, center_y), (max_x, center_y), (0, 0, 255), 2)  # Horizontal line

    # Draw diagonal lines within the bounding box
    cv2.line(img, tuple(corners[0]), tuple(corners[2]), (0, 0, 255), 2)  # Top-left to bottom-right
    cv2.line(img, tuple(corners[1]), tuple(corners[3]), (0, 0, 255), 2)  # Top-right to bottom-left

# Read image
path_XYs = read_csv('problems/isolated.csv')
plot(path_XYs)
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
    # Draw the regularized rectangle
    cv2.drawContours(img, [regularized_corners], 0, (0, 255, 0), 5)
    print("Regularized rectangle drawn.")
    # Draw symmetry lines
    draw_symmetry_lines(img, regularized_corners)
    print("Symmetry lines drawn.")

# Save and show the result
cv2.imwrite('regularized_rectangle_with_symmetry.png', img)
print("Result saved as 'regularized_rectangle_with_symmetry.png'.")
cv2.imshow('Regularized Rectangle with Symmetry', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
