import numpy as np
import matplotlib.pyplot as plt

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

paths = read_csv("problems/frag0.csv")

plot(paths, "output_plot.png")  # Plot the paths and save as PNG
