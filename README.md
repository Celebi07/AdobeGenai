# CurveCraft: A Journey into the World of Curves

## Project Overview
The goal is to develop an end-to-end process that transforms a CSV file containing line art data into a set of curves. These curves are defined as a connected sequence of cubic Bezier curves.

### Input
- The input is a CSV file that represents the line art as a sequence of points (polylines) in 2D space.

### Process
- We read and process the CSV file to extract the paths and points. These points are then used to reconstruct the image.
- Using OpenCV, we perform shape detection on the reconstructed image to identify and regularize geometric shapes.
- The detected shapes are then transformed into their ideal forms, and their symmetry is analyzed and visualized.

The approach leverages OpenCV's contour detection and various mathematical techniques to achieve regularization and symmetry detection for the shapes derived from the CSV input.

### Expected Output:
- Another set of paths with regularized shapes that exhibit the properties of symmetry, completeness, and beauty.

### Visualization:
- The resulting curves are visualized in SVG format for rendering in a browser.

## Approach

### 1. Shape Detection
- **YOLO Model**: We start by detecting basic shapes in the image by training a custom YOLO model. This helps us identify key geometric structures like rectangles, circles, stars, etc.

### 2. Contouring and Mathematical Techniques
- **OpenCV and Numpy**: We utilize contouring and mathematical approaches (using OpenCV) to identify and regularize detected shapes.
- 
### 3. Regularization Process
- **Shape Regularization**: Detected shapes are regularized by adjusting their geometric properties to achieve a more uniform and ideal form. This may involve refining edges, smoothing boundaries, or adjusting vertices to align with specific geometric constraints. The goal is to transform irregular shapes into their most symmetric and well-defined versions.

### 4. Symmetry Detection
- **Symmetry Analysis**: Symmetry lines are determined based on the regularized geometry of the shapes. The symmetry analysis considers the shape's inherent properties to identify axes of symmetry or central points around which the shape exhibits balanced features. This process applies to a variety of shapes, ensuring that their symmetrical characteristics are highlighted and visualized effectively.

