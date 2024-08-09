import cv2
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

def regularize_polygon(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

def main_pipeline(csv_path):
    paths_XYs = read_csv(csv_path)
    if paths_XYs is None:
        raise ValueError("CSV not found or unable to load.")
    print("CSV loaded successfully.")

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

    # Create a new blank image to draw the regularized polygons
    regularized_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    for contour in contours:
        regularized_polygon = regularize_polygon(contour)
        cv2.drawContours(regularized_img, [regularized_polygon], -1, (0, 255, 0), 1)

    # Save and show the result
    cv2.imwrite('regularized_polygons.png', img)
    print("Result saved as 'regularized_polygons.png'.")
    cv2.imshow('Regularized Polygons', regularized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
csv_path = 'poly.csv'
main_pipeline(csv_path)
