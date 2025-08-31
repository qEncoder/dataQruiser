# This is a mock script to make a measurement

import os
import time
import random
import numpy as np

def simulate_measurement(x_setpoints, polynomial_coeffs, noise_level=0.1):
    # Create a timestamp for the folder name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_path = os.path.join("data", timestamp)
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Create the file name
    file_name = f"measurement_data_{timestamp}.txt"
    file_path = os.path.join(folder_path, file_name)
    
    # Open the file and write the header
    with open(file_path, 'w') as f:
        f.write("x,y\n")
        
        # Simulate the measurement for each x setpoint
        for x in x_setpoints:
            # Calculate y using the polynomial function with added noise
            y = np.polyval(polynomial_coeffs, x) + random.gauss(0, noise_level)
            
            # Write the data point to the file
            f.write(f"{x},{y}\n")
            
            # Simulate some processing time
            time.sleep(0.1)
    
    print(f"Measurement complete. Data saved in {file_path}")

# Example usage
if __name__ == "__main__":
    # Define x setpoints
    x_setpoints = np.linspace(0, 10, 100)
    
    # Define polynomial coefficients (e.g., for y = 2x^2 + 3x + 1)
    polynomial_coeffs = [2, 3, 1]
    
    # Run the simulation
    simulate_measurement(x_setpoints, polynomial_coeffs)