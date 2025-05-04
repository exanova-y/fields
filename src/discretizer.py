import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from visualize import visualize_stacked_tensor

# Define sensor coordinates
sensor_coordinates = {"A":(0, 0), "B":(4, 0), "C":(0, 4), "D":(4, 4)}

# Define file paths for each sensor with the correct paths
sensor_files = {
    "A": "lavender-a.txt",   # Sensor at (0,0)
    "B": "lemongrass-b.txt", # Sensor at (4,0)
    "C": "orange-c.txt",     # Sensor at (0,4)
    "D": "street-air-d.txt", # Sensor at (4,4)
}

# Set up timing parameters
start_time = datetime.now()
delta_t = timedelta(seconds=5)  # 5 seconds between readings

def load_offline_data(folder_path="data"):
    """
    Load data from all sensors and return a dictionary of sensor readings.
    """
    all_sensor_data = {}
    
    for sensor_id, coord in sensor_coordinates.items():
        # Use absolute path to ensure file is found
        filepath = os.path.join(os.path.dirname(os.getcwd()), folder_path, sensor_files[sensor_id])
        # Alternative: If you're running from the project root
        # filepath = os.path.join(folder_path, sensor_files[sensor_id])
        
        readings = []
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip()]
                
                for i in range(0, len(lines), 3):
                    if i+2 < len(lines):  # Ensure we have all 3 lines
                        try:
                            # Extract values using split for better performance
                            gas = float(lines[i].split(': ')[1].split(' ')[0])
                            temp = float(lines[i+1].split(': ')[1].split(' ')[0])
                            humidity = float(lines[i+2].split(': ')[1].split(' ')[0])
                            
                            timestamp = i//3  # Use simple sequential numbering for timestamps
                            
                            readings.append({
                                "sensor_id": sensor_id,
                                "timestamp": timestamp,
                                "gas": gas,
                                "temp": temp,
                                "humidity": humidity,
                                "x": coord[0],
                                "y": coord[1]
                            })
                        except (IndexError, ValueError) as e:
                            print(f"Error parsing data at index {i}: {e}")
                
            all_sensor_data[sensor_id] = readings
            print(f"Loaded {len(readings)} readings for sensor {sensor_id}")
            
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found for sensor {sensor_id}")
            print("Current directory:", os.getcwd())
            print("Available files:", os.listdir(folder_path))
        except Exception as e:
            print(f"Error processing data for sensor {sensor_id}: {e}")
    
    return all_sensor_data

def interpolate_grid_with_layers(sensor_data, grid_size=5):
    """
    Interpolate sensor data onto a regular grid with 3 layers.
    
    Args:
        sensor_data (dict): Dictionary with sensor data
        grid_size (int): Size of the grid (grid_size x grid_size)
    
    Returns:
        tuple: (grid_tensor, timestamps, x_grid, y_grid) where:
               - grid_tensor is a 4D numpy array with shape [time, x, y, 3]
               - timestamps is a list of timestamps corresponding to the time dimension
               - x_grid and y_grid are the coordinate arrays
    """
    # Find all unique timestamps
    timestamps = set()
    for sensor_readings in sensor_data.values():
        for reading in sensor_readings:
            timestamps.add(reading['timestamp'])
    
    # Sort timestamps
    timestamps = sorted(list(timestamps))
    
    # Create coordinate grid once - use 4x4 grid to match the sensor layout
    grid_size = 4  # Override to ensure we use a 4x4 grid
    x_grid = np.linspace(0, 4, grid_size+1)  # 0, 1, 2, 3, 4
    y_grid = np.linspace(0, 4, grid_size+1)  # 0, 1, 2, 3, 4
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
    # Initialize the 4D tensor: [time, x, y, parameter]
    # Parameters: [gas, temperature, humidity]
    valid_timestamps = []
    valid_grids = []
    
    for timestamp in timestamps:
        # Extract coordinates and values for this timestamp
        points = []
        gas_values = []
        temp_values = []
        humidity_values = []
        
        for sensor_id, sensor_readings in sensor_data.items():
            for reading in sensor_readings:
                if reading['timestamp'] == timestamp:
                    points.append([reading['x'], reading['y']])
                    gas_values.append(reading['gas'])
                    temp_values.append(reading['temp'])
                    humidity_values.append(reading['humidity'])
        
        # Only proceed if we have data for this timestamp
        if len(points) >= 3:  # Need at least 3 points for meaningful interpolation
            # Create a 3-layer grid (grid_size × grid_size × 3)
            grid_3d = np.zeros((grid_size+1, grid_size+1, 3))
            
            # Choose interpolation method based on number of points
            # Cubic requires at least 4 points, otherwise use linear
            method = 'cubic' if len(points) >= 4 else 'linear'
            
            # Interpolate each parameter
            try:
                gas_grid = griddata(points, gas_values, (x_mesh, y_mesh), method=method, fill_value=0)
                temp_grid = griddata(points, temp_values, (x_mesh, y_mesh), method=method, fill_value=0)
                humidity_grid = griddata(points, humidity_values, (x_mesh, y_mesh), method=method, fill_value=0)
                
                # Assign to layers
                grid_3d[:, :, 0] = gas_grid      # Layer 0: Gas resistance
                grid_3d[:, :, 1] = temp_grid     # Layer 1: Temperature
                grid_3d[:, :, 2] = humidity_grid # Layer 2: Humidity
                
                # Store the valid timestamp and grid
                valid_timestamps.append(timestamp)
                valid_grids.append(grid_3d)
                
            except Exception as e:
                print(f"Error interpolating data for timestamp {timestamp}: {e}")
                # If interpolation fails, try nearest neighbor instead
                try:
                    print(f"Falling back to nearest neighbor interpolation for timestamp {timestamp}")
                    gas_grid = griddata(points, gas_values, (x_mesh, y_mesh), method='nearest', fill_value=0)
                    temp_grid = griddata(points, temp_values, (x_mesh, y_mesh), method='nearest', fill_value=0)
                    humidity_grid = griddata(points, humidity_values, (x_mesh, y_mesh), method='nearest', fill_value=0)
                    
                    grid_3d[:, :, 0] = gas_grid
                    grid_3d[:, :, 1] = temp_grid
                    grid_3d[:, :, 2] = humidity_grid
                    
                    # Store the valid timestamp and grid
                    valid_timestamps.append(timestamp)
                    valid_grids.append(grid_3d)
                    
                except Exception as e2:
                    print(f"Nearest neighbor interpolation also failed for timestamp {timestamp}: {e2}")
        else:
            print(f"Skipping timestamp {timestamp} - not enough points for interpolation (found {len(points)}, need at least 3)")
    
    # If we have any valid grids, stack them into a 4D tensor
    if valid_grids:
        # Stack along the first axis to create a 4D tensor [time, x, y, parameter]
        grid_tensor = np.stack(valid_grids, axis=0)
        return grid_tensor, valid_timestamps, x_grid, y_grid
    else:
        print("No valid grids were created!")
        return None, [], [], []

# Example usage
print("Loading sensor data...")
sensor_data = load_offline_data(folder_path="data")  # Explicitly set folder path

if sensor_data:
    print("\nInterpolating data to 4D tensor...")
    print(f"Number of sensors with data: {len(sensor_data)}")
    
    # Print how many readings we have for each sensor
    for sensor_id, readings in sensor_data.items():
        print(f"Sensor {sensor_id}: {len(readings)} readings")
    
    grid_tensor, timestamps, x_grid, y_grid = interpolate_grid_with_layers(sensor_data)
    
    if grid_tensor is not None:
        print(f"\nCreated 4D tensor with shape: {grid_tensor.shape}")
        print(f"Number of timestamps: {len(timestamps)}")
        print(f"First timestamp: {timestamps[0]}")
        
        # Display info about the first timestamp's grid
        print(f"\nGrid at first timestamp:")
        print(f"Gas resistance range: {grid_tensor[0, :, :, 0].min():.2f} to {grid_tensor[0, :, :, 0].max():.2f}")
        print(f"Temperature range: {grid_tensor[0, :, :, 1].min():.2f} to {grid_tensor[0, :, :, 1].max():.2f}")
        print(f"Humidity range: {grid_tensor[0, :, :, 2].min():.2f} to {grid_tensor[0, :, :, 2].max():.2f}")
        
        # Example of accessing data:
        print("\nExample: Accessing data at timestamp 0, position (2,2):")
        print(f"Gas resistance: {grid_tensor[0, 2, 2, 0]:.2f}")
        print(f"Temperature: {grid_tensor[0, 2, 2, 1]:.2f}")
        print(f"Humidity: {grid_tensor[0, 2, 2, 2]:.2f}")
        
        # Create and display a single visualization of the stacked tensor
        print("\nCreating visualization of the stacked tensor...")
        fig = visualize_stacked_tensor(grid_tensor, timestamps, x_grid, y_grid, sensor_coordinates)
        plt.show()