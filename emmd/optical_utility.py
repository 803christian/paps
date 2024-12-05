import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def generate_random_points(n_points, min_amplitude=0.3, max_amplitude=1.0):
    """
    Generate random points and amplitudes.
    
    :param n_points: Number of points to generate
    :param min_amplitude: Minimum amplitude value (default: 0.3)
    :param max_amplitude: Maximum amplitude value (default: 1.0)
    :return: List of tuples (x, y, amplitude)
    """
    data = []
    for _ in range(n_points):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        data.append((x, y, amplitude))
    return data

def generate_utility_distribution(data, resolution=100, spread_factor=0.05):
    """
    Generate a utility distribution based on input data points.
    
    :param data: List of tuples, each containing (x, y, amplitude)
    :param resolution: Grid resolution (default: 100x100)
    :param spread_factor: Controls the spread of the Gaussian (default: 0.05)
    :return: 2D numpy array representing the utility distribution
    """
    # Create a grid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Stack coordinates for vectorized computation
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    # Initialize the output grid
    grid = np.zeros((resolution, resolution))
    
    # Generate the distribution for each data point
    for x_coord, y_coord, amplitude in data:
        # Create covariance matrix
        cov = np.array([[spread_factor, 0], [0, spread_factor]])
        
        # Create a 2D Gaussian distribution
        rv = multivariate_normal([x_coord, y_coord], cov)
        
        # Calculate the Gaussian values
        gaussian = rv.pdf(pos)
        
        # Normalize the Gaussian to peak at the amplitude value
        max_val = gaussian.max()
        if max_val > 0:  # Avoid division by zero
            gaussian = (gaussian / max_val) * amplitude
            
        # Take maximum value at each point
        grid = np.maximum(grid, gaussian)
    
    return grid

if __name__ == '__main__':
    # Set random seed for reproducibility (optional)
    #np.random.seed(43)
    
    # Parameters
    n_points = 10  # Number of random points to generate
    resolution = 200  # Grid resolution
    spread_factor = 0.005  # Gaussian spread
    
    # Generate random points
    data = generate_random_points(n_points)
    
    # Generate the grid
    grid = generate_utility_distribution(data, resolution=resolution, spread_factor=spread_factor)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot with viridis colormap
    im = plt.imshow(grid, extent=[0, 1, 0, 1], origin='lower', 
                    cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    plt.colorbar(im, label='Utility Value')
    
    # Add the points used to generate the distribution
    for x, y, a in data:
        plt.plot(x, y, 'r.', markersize=10, 
                label=f'Point ({x:.2f}, {y:.2f}, amp={a:.2f})')
    
    plt.title(f'Utility Distribution with {n_points} Random Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()