import roboflow
import cv2
import matplotlib.pyplot as plt
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
            
def plot(paths_XYs, save_path="plot.png"):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define some colours to use
    
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            if np.array_equal(XY[0], XY[-1]):  # Check if the shape is closed
                ax.fill(XY[:, 0], XY[:, 1], c=c, alpha=0.5)  # Fill the shape with a semi-transparent color
            else:
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)  # Plot without filling if not closed
    
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off the axis
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save the plot without any padding
    plt.show()

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


    plot(paths_XYs, "temp.png") 

    # Create a blank image
    img_size = 1000  # Adjust size as needed
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for XYs in paths_XYs:
        for XY in XYs:
            for x, y in XY:
                cv2.rectangle(img, (int(x), int(y)), (int(x+2), int(y+2)), (255, 255, 255), -1)



    rf = roboflow.Roboflow(api_key="Rs2aIbfkwaKn9iBVHydo")

    project = rf.workspace().project("tpshape")
    model = project.version("4").model

    # optionally, change the confidence and overlap thresholds
    # values are percentages
    model.confidence = 50
    model.overlap = 25

    # predict on a local image
    prediction = model.predict("temp.png")

    # Plot the prediction in an interactive environment
    prediction.plot()

    # Convert predictions to JSON
    prediction_json = prediction.json()

    # print(prediction)



    for pred in prediction_json['predictions']:
        # Extract details from the prediction
        detected_class = pred['class']
        x = int(pred['x'])
        y = int(pred['y'])
        width = int(pred['width'])
        height = int(pred['height'])

        # Print the detected object
        print(f"Detected object: {detected_class}")

        # Step 7: Crop the image according to the bounding box
        x1 = x - width // 2
        y1 = y - height // 2
        x2 = x + width // 2
        y2 = y + height // 2

        cropped_img = img[y1:y2, x1:x2]

        # Optional: Save the cropped image
        cropped_image_path = f"cropped_{detected_class}.png"
        cv2.imwrite(cropped_image_path, cropped_img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours of the irregular shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new blank image with a white background
    regularised_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
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
        thickness = 2
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
            cv2.circle(regularised_img, (x, y), r, (0, 255, 0), 2)
            print("Circle found - 1")
    else:
       print("Circle found - 0")
    
    # Draw stars
    best_star = detect_function.find_best_star(contours)
    if best_star is None:
        print("Star found - 0")
    else:
        print("Star found - 1")
        cv2.drawContours(regularised_img, [best_star], -1, (0, 255, 0), 2)
        
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

# Input of the CSV 
csv_path = input("Enter the path of csv file : ")
main_pipeline(csv_path)

