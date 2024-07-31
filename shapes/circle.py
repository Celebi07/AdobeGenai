import cv2 as cv
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

def detect_circles(paths_XYs):
    # Create a blank image
    img_size = 1000  # Adjust size as needed
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Draw the paths on the image
    for XYs in paths_XYs:
        for XY in XYs:
            for x, y in XY:
                cv.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1)

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # Detect circles
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 2000,
                              param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        detected_circles = np.uint16(np.around(circles))
        for (x, y, r) in detected_circles[0, :]:
            cv.circle(img, (x, y), r, (0, 0, 255), 3)
            cv.circle(img, (x, y), 2, (0, 255, 255), 3)
        return True, img
    else:
        return False, img

# Example usage
csv_path = 'problems/isolated.csv'
paths_XYs = read_csv(csv_path)
plot(paths_XYs)
circle_detected, output_img = detect_circles(paths_XYs)

if circle_detected:
    print("Circle detected!")
else:
    print("No circle detected!")

# Display the output image with detected circles
cv.imshow('Detected Circles', output_img)
cv.waitKey(0)
cv.destroyAllWindows()
