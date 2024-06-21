import numpy as np

def chan_algorithm(sensor_positions, tdoa_measurements, c=3e8):
    """
    Chan's algorithm for hyperbolic location estimation using TDOA.
    
    Parameters:
    sensor_positions (ndarray): Coordinates of the sensors (N x 2 array).
    tdoa_measurements (ndarray): TDOA measurements (N-1 x 1 array).
    c (float): Speed of signal propagation (default is speed of light in m/s).
    
    Returns:
    ndarray: Estimated position of the source (1 x 2 array).
    """
    
    # Number of sensors
    N = sensor_positions.shape[0]
    
    # Reference sensor (assumed to be the first sensor)
    ref_pos = sensor_positions[0]
    
    # Construct the matrix A and vector b
    A = np.zeros((N-1, 2))
    b = np.zeros(N-1)
    
    for i in range(1, N):
        A[i-1, 0] = 2 * (sensor_positions[i, 0] - ref_pos[0])
        A[i-1, 1] = 2 * (sensor_positions[i, 1] - ref_pos[1])
        b[i-1] = tdoa_measurements[i-1]**2 - np.sum(sensor_positions[i]**2) + np.sum(ref_pos**2)
    
    # Initial estimate using least squares
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Construct weight matrix W (assuming equal variance for simplicity)
    W = np.eye(N-1)
    
    # Refine the initial estimate using weighted least squares
    ATW = A.T @ W
    x_refined = np.linalg.inv(ATW @ A) @ ATW @ b
    
    return x_refined

# Example usage
sensor_positions = np.array([
    [0, 0],     # Sensor 1
    [1, 0],     # Sensor 2
    [0, 1],     # Sensor 3
    [1, 1]      # Sensor 4
])

# Example TDOA measurements (in seconds, converted to distances using speed of light)
tdoa_measurements = np.array([
    1.5e-9 * 3e8,   # TDOA between Sensor 1 and Sensor 2
    2.0e-9 * 3e8,   # TDOA between Sensor 1 and Sensor 3
    1.0e-9 * 3e8    # TDOA between Sensor 1 and Sensor 4
])

# Estimated position
estimated_position = chan_algorithm(sensor_positions, tdoa_measurements)
print("Estimated Position: ", estimated_position)
