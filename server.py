# import cv2
# import numpy as np

# # Read the input image
# image = cv2.imread('image.png')

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Use Canny edge detection
# edges = cv2.Canny(gray, 500, 1050, apertureSize=5)

# # Detect lines using the Hough Line Transform
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# # Draw the lines on the original image
# if lines is not None:
#     for r_theta in lines:
#         arr = np.array(r_theta[0], dtype=np.float64)
#         r, theta = arr
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * r
#         y0 = b * r
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# # Save the result image with detected lines
# cv2.imwrite('linesDetected.png', image)


# # import cv2
# # import numpy as np

# # # Read image
# # image = cv2.imread('image.png')

# # # Convert image to grayscale
# # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# # # Use canny edge detection
# # edges = cv2.Canny(gray,50,150,apertureSize=3)

# # # Apply HoughLinesP method to 
# # # to directly obtain line end points
# # lines_list =[]
# # lines = cv2.HoughLinesP(
# #             edges, # Input edge image
# #             1, # Distance resolution in pixels
# #             np.pi/180, # Angle resolution in radians
# #             threshold=100, # Min number of votes for valid line
# #             minLineLength=5, # Min allowed length of line
# #             maxLineGap=10 # Max allowed gap between line for joining them
# #             )

# # # Iterate over points
# # for points in lines:
# #       # Extracted points nested in the list
# #     x1,y1,x2,y2=points[0]
# #     # Draw the lines joing the points
# #     # On the original image
# #     cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
# #     # Maintain a simples lookup list for points
# #     lines_list.append([(x1,y1),(x2,y2)])
    
# # # Save the result image
# # cv2.imwrite('detectedLines.png',image)

# import cv2
# import numpy as np

# def merge_lines(lines, min_angle_diff=5, min_dist=10):
#     """
#     Merges lines that are collinear and close to each other.

#     Parameters:
#     lines: List of lines detected by HoughLinesP
#     min_angle_diff: Minimum difference in angle (degrees) to consider lines collinear
#     min_dist: Minimum distance to consider lines close enough to merge

#     Returns:
#     merged_lines: List of merged lines
#     """
#     merged_lines = []
    
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         added = False
#         for idx, line2 in enumerate(merged_lines):
#             x3, y3, x4, y4 = line2[0]
#             if np.abs(np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y4 - y3, x4 - x3)) * 180 / np.pi < min_angle_diff:
#                 if np.sqrt((x1 - x3)**2 + (y1 - y3)**2) < min_dist or np.sqrt((x2 - x4)**2 + (y2 - y4)**2) < min_dist:
#                     # Merge lines
#                     merged_lines[idx] = [[min(x1, x3, x2, x4), min(y1, y3, y2, y4), 
#                                           max(x1, x3, x2, x4), max(y1, y3, y2, y4)]]
#                     added = True
#                     break
#         if not added:
#             merged_lines.append([[x1, y1, x2, y2]])
    
#     return merged_lines

# # Read image
# image = cv2.imread('image.png')

# # Convert image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Use Canny edge detection
# edges = cv2.Canny(gray, 500, 700, apertureSize=7)

# # Apply HoughLinesP method to detect line segments
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=5, maxLineGap=10)

# # Merge lines that are collinear and close to each other
# if lines is not None:
#     merged_lines = merge_lines(lines)

#     # Draw the merged lines on the original image
#     for line in merged_lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # Save the result image
# cv2.imwrite('detectedFullLines.png', image)


import numpy as np
import cv2 as cv
img = cv.imread('image.png')
output = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 2000,
                          param1=50, param2=30, minRadius=0, maxRadius=0)
detected_circles = np.uint16(np.around(circles))
for (x, y ,r) in detected_circles[0, :]:
    cv.circle(output, (x, y), r, (0, 0, 0), 3)
    cv.circle(output, (x, y), 2, (0, 255, 255), 3)

# Save the output image
cv.imwrite('detected_circles.png', output)
print("Image saved as detected_circles.png")


# import cv2
# import numpy as np

# # Function to check if two lines are perpendicular
# def is_perpendicular(line1, line2, angle_threshold=10):
#     x1, y1, x2, y2 = line1[0]
#     x3, y3, x4, y4 = line2[0]
#     angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#     angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
#     angle_diff = np.abs(angle1 - angle2)
#     return np.abs(angle_diff - 90) <= angle_threshold

# # Function to check if four lines form a rectangle
# def form_rectangle(lines):
#     if len(lines) != 4:
#         return False
#     # Check all pairwise combinations for perpendicularity
#     perpendicular_pairs = 0
#     for i in range(4):
#         for j in range(i + 1, 4):
#             if is_perpendicular(lines[i], lines[j]):
#                 perpendicular_pairs += 1
#     return perpendicular_pairs == 4

# # Read image
# img = cv2.imread('image.png')

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply edge detection
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# # Apply Hough Line Transform
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# if lines is None:
#     print("No lines found.")
# else:
#     # Filter out non-straight lines based on a threshold
#     straight_lines = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         if abs(x2 - x1) > 10 or abs(y2 - y1) > 10:  # Example threshold, adjust as needed
#             straight_lines.append(line)

#     # Attempt to form rectangles
#     found_rectangle = False
#     for i in range(len(straight_lines)):
#         for j in range(i + 1, len(straight_lines)):
#             for k in range(j + 1, len(straight_lines)):
#                 for l in range(k + 1, len(straight_lines)):
#                     lines_subset = [straight_lines[i], straight_lines[j], straight_lines[k], straight_lines[l]]
#                     if form_rectangle(lines_subset):
#                         for line in lines_subset:
#                             x1, y1, x2, y2 = line[0]
#                             cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         found_rectangle = True

#     if not found_rectangle:
#         print("No rectangles found.")
#     else:
#         print("Rectangle(s) found and outlined in green.")

#     # Save and show the result
#     cv2.imwrite('detected_rectangles.png', img)

# import cv2
# import numpy as np
# import svgwrite
# import cairosvg

# # Convert SVG to PNG
# def svg_to_png(svg_path, png_path):
#     cairosvg.svg2png(url=svg_path, write_to=png_path)

# # Function to check if two lines are perpendicular
# def is_perpendicular(line1, line2, angle_threshold=10):
#     x1, y1, x2, y2 = line1[0]
#     x3, y3, x4, y4 = line2[0]
#     angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#     angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
#     angle_diff = np.abs(angle1 - angle2)
#     return np.abs(angle_diff - 90) <= angle_threshold

# # Function to check if four lines form a rectangle
# def form_rectangle(lines):
#     if len(lines) != 4:
#         return False
#     perpendicular_pairs = 0
#     for i in range(4):
#         for j in range(i + 1, 4):
#             if is_perpendicular(lines[i], lines[j]):
#                 perpendicular_pairs += 1
#     return perpendicular_pairs == 4

# # Function to detect lines
# def detect_lines(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=5, maxLineGap=10)
#     lines_list = []
#     if lines is not None:
#         for points in lines:
#             x1, y1, x2, y2 = points[0]
#             lines_list.append([(x1, y1), (x2, y2)])
#     return lines_list

# # Function to detect circles
# def detect_circles(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 2000, param1=50, param2=30, minRadius=0, maxRadius=0)
#     circles_list = []
#     if circles is not None:
#         detected_circles = np.uint16(np.around(circles))
#         for (x, y, r) in detected_circles[0, :]:
#             circles_list.append((x, y, r))
#     return circles_list

# # Function to detect rectangles
# def detect_rectangles(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
#     rectangles = []
#     if lines is not None:
#         straight_lines = []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             if abs(x2 - x1) > 10 or abs(y2 - y1) > 10:
#                 straight_lines.append(line)
#         for i in range(len(straight_lines)):
#             for j in range(i + 1, len(straight_lines)):
#                 for k in range(j + 1, len(straight_lines)):
#                     for l in range(k + 1, len(straight_lines)):
#                         lines_subset = [straight_lines[i], straight_lines[j], straight_lines[k], straight_lines[l]]
#                         if form_rectangle(lines_subset):
#                             rectangles.append(lines_subset)
#     return rectangles

# # Function to create SVG from detected shapes
# def shapes2svg(lines, circles, rectangles, svg_path):
#     dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
#     group = dwg.g()

#     # Add lines
#     for line in lines:
#         x1, y1 = line[0]
#         x2, y2 = line[1]
#         group.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke='black'))

#     # Add circles
#     for circle in circles:
#         x, y, r = circle
#         group.add(dwg.circle(center=(x, y), r=r, stroke='black', fill='none'))

#     # Add rectangles
#     for rect in rectangles:
#         points = [rect[0][0], rect[1][0], rect[2][0], rect[3][0]]
#         group.add(dwg.polygon(points, stroke='black', fill='none'))

#     dwg.add(group)
#     dwg.save()

# # Main pipeline function
# def pipeline(svg_path, png_path, output_svg_path):
#     # Convert SVG to PNG
#     svg_to_png(svg_path, png_path)

#     # Detect shapes
#     lines = detect_lines(png_path)
#     circles = detect_circles(png_path)
#     rectangles = detect_rectangles(png_path)

#     # Generate SVG from detected shapes
#     shapes2svg(lines, circles, rectangles, output_svg_path)

#     print(f"Processed SVG saved as {output_svg_path}")

# # Example usage
# input_svg_path = 'problems/frag0.svg'
# intermediate_png_path = 'intermediate_image.png'
# output_svg_path = 'output_image.svg'
# pipeline(input_svg_path, intermediate_png_path, output_svg_path)



# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import csv

# # Function to read CSV file
# def read_csv(csv_path):
#     np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
#     path_XYs = []
#     for i in np.unique(np_path_XYs[:, 0]):
#         npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
#         XYs = []
#         for j in np.unique(npXYs[:, 0]):
#             XY = npXYs[npXYs[:, 0] == j][:, 1:]
#             XYs.append(XY)
#         path_XYs.append(XYs)
#     return path_XYs

# # Function to plot paths and save as an image
# def plot(paths_XYs, filename='output_image.png'):
#     fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
#     colours = ['r', 'g', 'b', 'c', 'm', 'y']
#     for i, XYs in enumerate(paths_XYs):
#         c = colours[i % len(colours)]
#         for XY in XYs:
#             ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
#     ax.set_aspect('equal')
#     plt.savefig(filename)
#     plt.close()

# # Function to detect lines
# def detect_lines(image_path, output_path='detected_lines.png'):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines_list = []
#     lines = cv2.HoughLinesP(
#         edges,
#         1,
#         np.pi / 180,
#         threshold=100,
#         minLineLength=5,
#         maxLineGap=10
#     )
#     if lines is not None:
#         for points in lines:
#             x1, y1, x2, y2 = points[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             lines_list.append([(x1, y1), (x2, y2)])
#     cv2.imwrite(output_path, image)
#     return lines_list

# # Function to detect circles
# def detect_circles(image_path, output_path='detected_circles.png'):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         1,
#         2000,
#         param1=50,
#         param2=30,
#         minRadius=0,
#         maxRadius=0
#     )
#     detected_circles = np.uint16(np.around(circles))
#     if detected_circles is not None:
#         for (x, y, r) in detected_circles[0, :]:
#             cv2.circle(image, (x, y), r, (0, 0, 0), 3)
#             cv2.circle(image, (x, y), 2, (0, 255, 255), 3)
#     cv2.imwrite(output_path, image)

# # Function to detect rectangles
# def detect_rectangles(image_path, output_path='detected_rectangles.png'):
#     def is_perpendicular(line1, line2, angle_threshold=10):
#         x1, y1, x2, y2 = line1[0]
#         x3, y3, x4, y4 = line2[0]
#         angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#         angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
#         angle_diff = np.abs(angle1 - angle2)
#         return np.abs(angle_diff - 90) <= angle_threshold

#     def form_rectangle(lines):
#         if len(lines) != 4:
#             return False
#         perpendicular_pairs = 0
#         for i in range(4):
#             for j in range(i + 1, 4):
#                 if is_perpendicular(lines[i], lines[j]):
#                     perpendicular_pairs += 1
#         return perpendicular_pairs == 4

#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

#     if lines is not None:
#         straight_lines = []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             if abs(x2 - x1) > 10 or abs(y2 - y1) > 10:
#                 straight_lines.append(line)
        
#         found_rectangle = False
#         for i in range(len(straight_lines)):
#             for j in range(i + 1, len(straight_lines)):
#                 for k in range(j + 1, len(straight_lines)):
#                     for l in range(k + 1, len(straight_lines)):
#                         lines_subset = [straight_lines[i], straight_lines[j], straight_lines[k], straight_lines[l]]
#                         if form_rectangle(lines_subset):
#                             for line in lines_subset:
#                                 x1, y1, x2, y2 = line[0]
#                                 cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                             found_rectangle = True
#         if not found_rectangle:
#             print("No rectangles found.")
#         else:
#             print("Rectangle(s) found and outlined in green.")
#     else:
#         print("No lines found.")
    
#     cv2.imwrite(output_path, img)

# # Function to write CSV file
# def write_csv(csv_path, data):
#     with open(csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         for item in data:
#             writer.writerow(item)

# # Main pipeline
# def pipeline(csv_input, image_output, csv_output):
#     # Step 1: Read CSV and generate image
#     paths_XYs = read_csv(csv_input)
#     plot(paths_XYs, image_output)
    
#     # Step 2: Detect shapes
#     detect_lines(image_output, 'detected_lines.png')
#     detect_circles(image_output, 'detected_circles.png')
#     detect_rectangles(image_output, 'detected_rectangles.png')

#     # Step 3: Write CSV with detected shapes (as an example, let's assume we save lines only)
#     detected_lines = detect_lines(image_output)
#     write_csv(csv_output, detected_lines)

# # Example usage
# pipeline('problems/frag0.csv', 'output_image.png', 'output_shapes.csv')


# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import csv

# # Function to read CSV file
# def read_csv(csv_path):
#     np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
#     path_XYs = []
#     for i in np.unique(np_path_XYs[:, 0]):
#         npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
#         XYs = []
#         for j in np.unique(npXYs[:, 0]):
#             XY = npXYs[npXYs[:, 0] == j][:, 1:]
#             XYs.append(XY)
#         path_XYs.append(XYs)
#     return path_XYs

# # Function to plot paths and save as an image
# def plot(paths_XYs, filename='output_image.png'):
#     fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
#     colours = ['r', 'g', 'b', 'c', 'm', 'y']
#     for i, XYs in enumerate(paths_XYs):
#         c = colours[i % len(colours)]
#         for XY in XYs:
#             ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
#     ax.set_aspect('equal')
#     plt.savefig(filename)
#     plt.close()

# # Function to detect lines
# def detect_lines(image_path, output_path='detected_lines.png'):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines_list = []
#     lines = cv2.HoughLinesP(
#         edges,
#         1,
#         np.pi / 180,
#         threshold=100,
#         minLineLength=5,
#         maxLineGap=10
#     )
#     if lines is not None:
#         for points in lines:
#             x1, y1, x2, y2 = points[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             lines_list.append([x1, y1, x2, y2])
#     cv2.imwrite(output_path, image)
#     return lines_list

# # Function to detect circles
# def detect_circles(image_path, output_path='detected_circles.png'):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         1,
#         2000,
#         param1=50,
#         param2=30,
#         minRadius=0,
#         maxRadius=0
#     )
#     detected_circles = np.uint16(np.around(circles))
#     if detected_circles is not None:
#         for (x, y, r) in detected_circles[0, :]:
#             cv2.circle(image, (x, y), r, (0, 0, 0), 3)
#             cv2.circle(image, (x, y), 2, (0, 255, 255), 3)
#     cv2.imwrite(output_path, image)

# # Function to detect rectangles
# def detect_rectangles(image_path, output_path='detected_rectangles.png'):
#     def is_perpendicular(line1, line2, angle_threshold=10):
#         x1, y1, x2, y2 = line1[0]
#         x3, y3, x4, y4 = line2[0]
#         angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#         angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
#         angle_diff = np.abs(angle1 - angle2)
#         return np.abs(angle_diff - 90) <= angle_threshold

#     def form_rectangle(lines):
#         if len(lines) != 4:
#             return False
#         perpendicular_pairs = 0
#         for i in range(4):
#             for j in range(i + 1, 4):
#                 if is_perpendicular(lines[i], lines[j]):
#                     perpendicular_pairs += 1
#         return perpendicular_pairs == 4

#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

#     if lines is not None:
#         straight_lines = []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             if abs(x2 - x1) > 10 or abs(y2 - y1) > 10:
#                 straight_lines.append(line)
        
#         found_rectangle = False
#         for i in range(len(straight_lines)):
#             for j in range(i + 1, len(straight_lines)):
#                 for k in range(j + 1, len(straight_lines)):
#                     for l in range(k + 1, len(straight_lines)):
#                         lines_subset = [straight_lines[i], straight_lines[j], straight_lines[k], straight_lines[l]]
#                         if form_rectangle(lines_subset):
#                             for line in lines_subset:
#                                 x1, y1, x2, y2 = line[0]
#                                 cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                             found_rectangle = True
#         if not found_rectangle:
#             print("No rectangles found.")
#         else:
#             print("Rectangle(s) found and outlined in green.")
#     else:
#         print("No lines found.")
    
#     cv2.imwrite(output_path, img)

# # Function to write CSV file
# def write_csv(csv_path, data):
#     with open(csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         for item in data:
#             # Format the output to match the desired format
#             formatted_line = [format(coord, '.16e') for coord in item]
#             writer.writerow(formatted_line)

# # Main pipeline
# def pipeline(csv_input, image_output, csv_output):
#     # Step 1: Read CSV and generate image
#     paths_XYs = read_csv(csv_input)
#     plot(paths_XYs, image_output)
    
#     # Step 2: Detect shapes
#     detected_lines = detect_lines(image_output, 'detected_lines.png')
#     detect_circles(image_output, 'detected_circles.png')
#     detect_rectangles(image_output, 'detected_rectangles.png')

#     # Step 3: Write CSV with detected shapes (as an example, let's assume we save lines only)
#     write_csv(csv_output, detected_lines)

# # Example usage
# pipeline('problems/frag0.csv', 'output_image.png', 'output_shapes.csv')





# import numpy as np
# import matplotlib.pyplot as plt
# import csv

# def read_csv(csv_path):
#     """Reads a CSV file and returns the data as a list of arrays."""
#     np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
#     path_XYs = []
#     for i in np.unique(np_path_XYs[:, 0]):
#         npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
#         XYs = []
#         for j in np.unique(npXYs[:, 0]):
#             XY = npXYs[npXYs[:, 0] == j][:, 1:]
#             XYs.append(XY)
#         path_XYs.append(XYs)
#     return path_XYs

# def transform_coordinates(paths_XYs):
#     """Applies a transformation to the coordinates."""
#     transformed_paths_XYs = []
#     for XYs in paths_XYs:
#         transformed_XYs = []
#         for XY in XYs:
#             # Example transformation: rotate the points around the origin
#             transformed_XY = XY + np.array([0.5, 0.5])
#             transformed_XYs.append(transformed_XY)
#         transformed_paths_XYs.append(transformed_XYs)
#     return transformed_paths_XYs

# def save_csv(transformed_paths_XYs, output_csv_path):
#     """Saves the transformed coordinates to a CSV file."""
#     with open(output_csv_path, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         for XYs in transformed_paths_XYs:
#             for XY in XYs:
#                 for point in XY:
#                     csv_writer.writerow([0.0, 0.0, *point])

# def plot(paths_XYs, output_svg_path):
#     """Plots the transformed coordinates and saves the plot as an SVG file."""
#     fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
#     colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#     for i, XYs in enumerate(paths_XYs):
#         c = colours[i % len(colours)]
#         for XY in XYs:
#             ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
#     ax.set_aspect('equal')
#     plt.savefig(output_svg_path, format='svg')
#     plt.show()

# # Example usage
# input_csv_path = 'problems/frag0.csv'
# output_csv_path = 'output.csv'
# output_svg_path = 'output.svg'

# # Read the input CSV file
# paths_XYs = read_csv(input_csv_path)

# # Transform the coordinates
# transformed_paths_XYs = transform_coordinates(paths_XYs)

# # Save the transformed coordinates to a new CSV file
# save_csv(transformed_paths_XYs, output_csv_path)

# # Plot the transformed coordinates and save as an SVG image
# plot(transformed_paths_XYs, output_svg_path)
