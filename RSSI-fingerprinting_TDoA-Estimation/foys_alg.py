import numpy as np

def tdoa_position_estimation(initial_coords, gateway_coords, tdoa_values, max_iterations=100, tolerance=1e-6):
    """
    Estimates the position using TDOA and Taylor-series estimation.
    
    Parameters:
    - initial_coords: Initial guess for the target node's coordinates [x, y].
    - gateway_coords: Coordinates of the gateways as a list of [x, y] pairs.
    - tdoa_values: Measured TDOA values between each pair of gateways.
    - max_iterations: Maximum number of iterations for convergence.
    - tolerance: Convergence tolerance for the position updates.
    
    Returns:
    - Estimated coordinates [x, y].
    """
    
    # Convert input lists to numpy arrays for easier manipulation
    initial_coords = np.array(initial_coords, dtype=float)
    gateway_coords = np.array(gateway_coords, dtype=float)
    tdoa_values = np.array(tdoa_values, dtype=float)
    
    num_gateways = len(gateway_coords)
    num_measurements = len(tdoa_values)
    
    # Initialize the estimated position with the initial guess
    estimated_coords = initial_coords
    
    for iteration in range(max_iterations):
        A = np.zeros((num_measurements, 2))
        z = np.zeros(num_measurements)
        
        for k in range(num_measurements):
            i, j = int(tdoa_values[k, 0]), int(tdoa_values[k, 1])
            measured_tdoa = tdoa_values[k, 2]
            
            d_i = np.linalg.norm(estimated_coords - gateway_coords[i])
            d_j = np.linalg.norm(estimated_coords - gateway_coords[j])
            
            A[k, 0] = (estimated_coords[0] - gateway_coords[i, 0]) / d_i - (estimated_coords[0] - gateway_coords[j, 0]) / d_j
            A[k, 1] = (estimated_coords[1] - gateway_coords[i, 1]) / d_i - (estimated_coords[1] - gateway_coords[j, 1]) / d_j
            
            z[k] = measured_tdoa - (d_i - d_j)
        
        # Solve the least squares problem
        delta, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        
        # Update the estimated coordinates
        estimated_coords += delta
        
        # Check for convergence
        if np.linalg.norm(delta) < tolerance:
            break
    
    return estimated_coords.tolist()

# # Example usage:
# initial_coords = [0.0, 0.0]
# gateway_coords = [
#     [0.0, 0.0],
#     [1.0, 0.0],
#     [0.0, 1.0],
#     [1.0, 1.0]
# ]
# tdoa_values = [
#     [0, 1, 0.5],
#     [0, 2, 0.3],
#     [0, 3, 0.4],
#     [1, 2, -0.2],
#     [1, 3, -0.1],
#     [2, 3, 0.1]
# ]

initial_coords = [1313.571538, -6321.141998]

# Receiving gateways: ['FF0178DF', '080E0116', '08060716', 'FF01753E']
# Time of arrivals: ['2018-12-31T08:49:13.832705509+01:00', '2018-12-31T08:49:13.944+01:00', '2018-12-31T08:49:13.949+01:00', '2018-12-31T08:49:13.832710481+01:00']

gateway_coords = [
    [ 1.73193814e+03, -6.53513544e+03],
    [ 6.97356078e+02, -7.32894077e+03],
    [ 1.94347883e+03, -6.10173267e+03],
    [ 2.51986817e+03, -7.02576446e+03]]

tdoa_values = [[0, 1, -0.11129449100000001*3e8],
 [0, 2, -0.11629449100000001*3e8],
 [0, 3, -4.972e-06*3e8],
 [1, 2, -0.005*3e8],
 [1, 3, 0.111289519*3e8],
 [2, 3, 0.11628951900000001*3e8]]

estimated_coords = tdoa_position_estimation(initial_coords, gateway_coords, tdoa_values)
print("Estimated Coordinates:", estimated_coords)
