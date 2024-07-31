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

# Read image
path_XYs = read_csv('problems/frag1.csv')
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

# Save and show the result
cv2.imwrite('detected_lines.png', img)
print("Result saved as 'detected_lines.png'.")
cv2.imshow('Detected Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
