import cv2
import numpy as np

import libraries.shape_detection as detect_function
import libraries.symmetry_detection as detect_symmetry

# Function to read CSV and receive pathXYs from them 
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

# Function to regularize the corners to form a perfect rectangle
def regularize_corners(corners):
    rect = cv2.minAreaRect(np.array(corners))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

# Main pipeline function
def main_pipeline(csv_path):
    paths_XYs = read_csv(csv_path)
    if paths_XYs is None:
        raise ValueError("Image not found or unable to load.")

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

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours of the irregular shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new blank image 
    regularised_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Identify rectangle and draw a regularised version of it
    best_rect = detect_function.find_best_rectangle(contours)
    if best_rect is None:
        print("Rectangle found - 0")
    else:
        print("Rectangle found - 1")
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
        cv2.rectangle(regularised_img, tuple(top_left), tuple(bottom_right), (0, 255, 0), thickness)

    # Detect circles
    circle_detected = detect_function.detect_circles(img, gray)
    if circle_detected is not None:
        detected_circles = np.uint16(np.around(circle_detected))
        for (x, y, r) in detected_circles[0, :]:
            # Draw the regularized circle boundary
            cv2.circle(regularised_img, (x, y), r, (0, 255, 255), 1)
            print("Circle found - 1")
    else:
       print("Circle found - 0")
    
    # Draw stars
    best_star = detect_function.find_best_star(contours)
    if best_star is None:
        print("Star found - 0")
    else:
        print("Star found - 1")
        cv2.drawContours(regularised_img, [best_star], -1, (0, 255, 0), 1)
        
    # Create a new blank image 
    symmetry_img = regularised_img.copy() 
        
    # Draw symmetry lines for the rectangle
    if best_rect is not None:
        detect_symmetry.draw_symmetry_lines_rectangle(symmetry_img, regularized_corners)
        
    # Draw symmetry lines for the circle
    if circle_detected is not None:
        detect_symmetry.draw_symmetry_lines_circle(symmetry_img, (x, y, r))
        
    # Draw symmetry lines for the star
    if best_star is not None:
        detect_symmetry.draw_symmetry_lines_star(symmetry_img, best_star)

    # Save and show the result
    cv2.imwrite('regularised_image.png', regularised_img)
    cv2.imwrite('symmetric_image.png', symmetry_img)
    print("Result saved as regularised_image.png and symmetric_image.png.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Input of the CSV 
csv_path = input("Enter the path of csv file : ")
main_pipeline(csv_path)

