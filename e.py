import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to read CSV data
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

# Function to fit ellipses
def detect_and_draw_ellipses(path_XYs):
    ellipses = []
    for XYs in path_XYs:
        for XY in XYs:
            if len(XY) >= 5:  # Need at least 5 points to fit an ellipse
                # Reshape XY to be compatible with cv2.fitEllipse
                XY_reshaped = XY.reshape(-1, 1, 2).astype(np.float32)
                try:
                    ellipse = cv2.fitEllipse(XY_reshaped)
                    ellipses.append(ellipse)
                except cv2.error as e:
                    print(f"OpenCV error: {e}")
            else:
                print(f"Not enough points for ellipse fitting: {len(XY)}")
    return ellipses

# Function to plot raw points and ellipses
def plot_data_and_ellipses(path_XYs, ellipses):
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 10))
    # Plot raw points
    for XYs in path_XYs:
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], 'o', label='Raw Points')
    
    if not ellipses:
        print("No ellipses detected.")
    
    # Plot ellipses
    for ellipse in ellipses:
        center, axes, angle = ellipse
        ellipse_patch = patches.Ellipse(
            xy=center, 
            width=axes[0], 
            height=axes[1], 
            angle=angle, 
            edgecolor='r', 
            facecolor='none', 
            linewidth=2
        )
        ax.add_patch(ellipse_patch)
    
    ax.set_aspect('equal')
    plt.legend()
    plt.show()

# Main function to run the detection and plotting
def main(csv_path):
    path_XYs = read_csv(csv_path)
    print(f"Data read from CSV: {path_XYs}")
    ellipses = detect_and_draw_ellipses(path_XYs)
    print(f"Ellipses detected: {ellipses}")
    plot_data_and_ellipses(path_XYs, ellipses)

# Replace 'path_to_your_csv_file.csv' with your actual CSV file path
if __name__ == "__main__":
    main('problems/occlusion1.csv')
