import numpy as np

def chan_algorithm(gateway_positions, tdoa_measurements, c=3e8):
    """
    Chan's algorithm for hyperbolic location estimation using TDOA.
    
    Parameters:
    gateway_positions (ndarray): Coordinates of the gateways (N x 2 array).
    tdoa_measurements (ndarray): TDOA measurements (N-1 x 1 array).
    c (float): Speed of signal propagation (default is speed of light in m/s).
    
    Returns:
    ndarray: Estimated position of the source (1 x 2 array).
    """
    
    # Number of gateways
    N = gateway_positions.shape[0]
    
    # Reference gateway (assumed to be the first gateway)
    ref_pos = gateway_positions[0]
    
    # Construct the matrix A and vector b
    A = np.zeros((N-1, 2))
    b = np.zeros(N-1)
    
    for i in range(1, N):
        A[i-1, 0] = 2 * (gateway_positions[i, 0] - ref_pos[0])
        A[i-1, 1] = 2 * (gateway_positions[i, 1] - ref_pos[1])
        b[i-1] = tdoa_measurements[i-1]**2 - np.sum(gateway_positions[i]**2) + np.sum(ref_pos**2)
    
    # Initial estimate using least squares
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Construct weight matrix W (assuming equal variance for simplicity)
    W = np.eye(N-1)
    
    # Refine the initial estimate using weighted least squares
    ATW = A.T @ W
    x_refined = np.linalg.inv(ATW @ A) @ ATW @ b
    
    return x_refined

# Example usage
gateway_positions = np.array([
    [0, 0],     # gateway 1
    [1, 0],     # gateway 2
    [0, 1],     # gateway 3
    [1, 1]      # gateway 4
])

# Example TDOA measurements (in seconds, converted to distances using speed of light)
tdoa_measurements = np.array([
    1.5e-9 * 3e8,   # TDOA between gateway 1 and gateway 2
    2.0e-9 * 3e8,   # TDOA between gateway 1 and gateway 3
    1.0e-9 * 3e8    # TDOA between gateway 1 and gateway 4
])

# Estimated position
estimated_position = chan_algorithm(gateway_positions, tdoa_measurements)
print("Estimated Position: ", estimated_position)
