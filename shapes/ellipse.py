import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def read_csv(csv_path):
    np_data = np.genfromtxt(csv_path, delimiter=',')
    paths = {}
    for row in np_data:
        path_id, _, x, y = row
        if path_id not in paths:
            paths[path_id] = {'X': [], 'Y': []}
        paths[path_id]['X'].append(x)
        paths[path_id]['Y'].append(y)
    for path_id in paths:
        paths[path_id]['X'] = np.array(paths[path_id]['X'])
        paths[path_id]['Y'] = np.array(paths[path_id]['Y'])
    return paths

def fit_ellipse(X, Y):
    x_m = np.mean(X)
    y_m = np.mean(Y)
    
    def calc_R(a, b, xc, yc, theta):
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        Xr = (X - xc) * cos_theta + (Y - yc) * sin_theta
        Yr = -(X - xc) * sin_theta + (Y - yc) * cos_theta
        return np.sqrt((Xr/a)*2 + (Yr/b)*2)
    
    def f(params):
        a, b, xc, yc, theta = params
        Ri = calc_R(a, b, xc, yc, theta)
        return Ri - 1
    
    a_init = (np.max(X) - np.min(X)) / 2
    b_init = (np.max(Y) - np.min(Y)) / 2
    center_estimate = x_m, y_m
    theta_init = 0
    params_init = [a_init, b_init, x_m, y_m, theta_init]
    
    params, _ = leastsq(f, params_init)
    a, b, xc, yc, theta = params
    return a, b, xc, yc, theta

def generate_ellipse(a, b, xc, yc, theta, num_points=100):
    t = np.linspace(0, 2*np.pi, num_points)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    X = xc + a * np.cos(t) * cos_theta - b * np.sin(t) * sin_theta
    Y = yc + a * np.cos(t) * sin_theta + b * np.sin(t) * cos_theta
    return X, Y

def save_csv(paths, output_path):
    data = []
    for path_id, coords in paths.items():
        for x, y in zip(coords['X'], coords['Y']):
            data.append([path_id, 0, x, y])
    np.savetxt(output_path, data, delimiter=',')

def plot_ellipse(X, Y, X_fit, Y_fit, title="Ellipse Fitting"):
    plt.figure(figsize=(6,6))
    plt.plot(X, Y, 'bo', label='Original Data')
    plt.plot(X_fit, Y_fit, 'r-', label='Fitted Ellipse')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(title)
    plt.show()

def process_paths(csv_path, output_path):
    paths = read_csv(csv_path)
    regularized_paths = {}
    
    for path_id, coords in paths.items():
        X, Y = coords['X'], coords['Y']
        
        # Fit the points to an ellipse
        try:
            a, b, xc, yc, theta = fit_ellipse(X, Y)
            
            # Generate points for the perfect ellipse
            X_fit, Y_fit = generate_ellipse(a, b, xc, yc, theta)
            
            # Save the regularized ellipse
            regularized_paths[path_id] = {'X': X_fit, 'Y': Y_fit}
            
            # Plot the original and fitted ellipse for visualization
            plot_ellipse(X, Y, X_fit, Y_fit, title=f"Path ID: {path_id}")
                
        except Exception as e:
            print(f"Could not fit an ellipse for path {path_id}: {e}")
    
    save_csv(regularized_paths, output_path)

# Process the paths from the CSV file
process_paths("problems/occlusion2.csv", "regularized_ellipses.csv")