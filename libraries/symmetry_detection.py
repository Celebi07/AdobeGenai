import cv2
import numpy as np

# Function to draw symmetry lines within the bounding box
def draw_symmetry_lines_rectangle(img, corners):
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

# Function to draw symmetry lines within the circle
def draw_symmetry_lines_circle(img, circle):
    x, y, r = circle
    center = (x, y)

    # Draw vertical and horizontal lines through the center of the circle
    cv2.line(img, (x, y - r), (x, y + r), (0, 0, 255), 2)  # Vertical line
    cv2.line(img, (x - r, y), (x + r, y), (0, 0, 255), 2)  # Horizontal line

    # Draw diagonal lines through the center of the circle
    cv2.line(img, (x - r, y - r), (x + r, y + r), (0, 0, 255), 2)  # Top-left to bottom-right
    cv2.line(img, (x - r, y + r), (x + r, y - r), (0, 0, 255), 2)  # Bottom-left to top-right

# Function to draw symmetry lines within the star
def draw_symmetry_lines_star(img, star):
    # Calculate the center of the star
    center_x = int(np.mean(star[:, 0, 0]))
    center_y = int(np.mean(star[:, 0, 1]))

    # Draw lines connecting opposite vertices through the center
    for i in range(5):
        start_point = tuple(star[i][0])
        end_point = tuple(star[i + 5][0])
        cv2.line(img, start_point, end_point, (0, 0, 255), 2)