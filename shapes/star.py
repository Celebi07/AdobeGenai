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

# Read image
csv_path = 'problems/isolated.csv'
path_XYs = read_csv(csv_path)
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

# Find the best star shape
best_star = find_best_star(contours)
if best_star is None:
    print("No star shape found.")
else:
    print("Best star shape found.")
    # Draw the star shape on the image
    star_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cv2.drawContours(star_img, [best_star], -1, (0, 255, 0), 1)
    print("Star shape drawn.")

    # Save and show the result
    cv2.imwrite('star_shape_resized.png', star_img)
    print("Result saved as 'star_shape_resized.png'.")
    cv2.imshow('Star Shape Resized', star_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

