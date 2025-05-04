#!/usr/bin/env python3
"""
Test workflow that integrates real-time simulated sensor data
with motion simulation and visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import os
import requests
import json
from datetime import datetime
import sys

# Import our modules
import sender
import receiver
from ingestor import sensor_coordinates

# We'll import simulate_field_motion directly to avoid running motion.py's example code
# This requires a temporary path hack to get the function without triggering the file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import simulate_field_motion
from visualize import create_flow_visualization

# Configuration
DATA_COLLECTION_TIME = 30  # seconds to collect data
GRID_SIZE = 5  # Grid dimensions
PARAMS = 3  # Gas, Temperature, Humidity
UPDATE_INTERVAL = 1  # How often to update the visualization (seconds)
SENSOR_URL = f"http://{sender.HOST}:{sender.PORT}/"

class RealTimeWorkflow:
    """Workflow that processes and visualizes real-time sensor data"""
    
    def __init__(self):
        self.running = True
        self.tensor = np.zeros((1, GRID_SIZE, GRID_SIZE, PARAMS))
        self.time_steps = [datetime.now().isoformat()]
        self.x_grid = np.linspace(0, 1, GRID_SIZE)
        self.y_grid = np.linspace(0, 1, GRID_SIZE)
        self.lock = threading.Lock()
        self.server = None
        self.fig = None
    
    def start_data_source(self):
        """Start the simulated data source"""
        print("Starting data source...")
        self.server = sender.start_server()
        if not self.server:
            print("Failed to start server")
            return False
        
        # Set up receiver to point to our data source
        receiver.SENSOR_URL = SENSOR_URL
        receiver.start()
        return True
    
    def fetch_latest_data(self):
        """Fetch the latest data point directly from the sensor"""
        try:
            response = requests.get(SENSOR_URL, timeout=1)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None
    
    def update_tensor(self, sensor_data):
        """Update tensor with new sensor data"""
        # This is a simplification - in a real application, you'd implement
        # proper interpolation based on sensor locations
        with self.lock:
            # Create a new time slice with the same data at all grid points
            new_slice = np.zeros((1, GRID_SIZE, GRID_SIZE, PARAMS))
            
            # For simplicity: fill all grid cells with the same values
            # A real implementation would use proper spatial interpolation
            new_slice[0, :, :, 0] = sensor_data['gas']  # Gas
            new_slice[0, :, :, 1] = sensor_data['temp']  # Temperature
            new_slice[0, :, :, 2] = sensor_data['humidity']  # Humidity
            
            # Add random variations for visualization purposes
            variations = np.random.normal(0, 0.05, new_slice.shape)
            new_slice = new_slice * (1 + variations)
            
            # Append to tensor
            self.tensor = np.concatenate([self.tensor, new_slice], axis=0)
            self.time_steps.append(datetime.now().isoformat())
    
    def data_collection_loop(self):
        """Loop that collects data for the specified time"""
        start_time = time.time()
        count = 0
        
        print(f"Collecting data for {DATA_COLLECTION_TIME} seconds...")
        while time.time() - start_time < DATA_COLLECTION_TIME and self.running:
            data = self.fetch_latest_data()
            if data:
                self.update_tensor(data)
                count += 1
                print(f"Data point {count}: gas={data['gas']:.2f}, temp={data['temp']:.2f}°C, humidity={data['humidity']:.2f}%")
            time.sleep(0.5)  # Collect at 2Hz
        
        print(f"Collected {count} data points")
        return count > 0
    
    def run_simulation(self):
        """Run the motion simulation on the collected data"""
        print("Running motion simulation...")
        with self.lock:
            # Only process if we have enough data
            if len(self.tensor) < 2:
                print("Not enough data for simulation")
                return None, None, None
            
            # Copy the tensor to avoid modification during simulation
            tensor_copy = self.tensor.copy()
        
        # Print tensor shape for debugging
        print(f"Tensor shape: {tensor_copy.shape}")
        
        # Run the simulation
        sim_steps = min(100, max(30, len(tensor_copy) * 3))
        simulated_fields, velocity_x, velocity_y = simulate_field_motion(
            tensor_copy,
            num_steps=sim_steps,
            dt=0.05,
            diffusion_coef=0.03,
            velocity_scale=0.025
        )
        
        print(f"Simulation complete: {sim_steps} steps")
        return simulated_fields, velocity_x, velocity_y
    
    def visualize_results(self, simulated_fields, velocity_x, velocity_y):
        """Visualize the simulation results"""
        if simulated_fields is None:
            return
        
        # Create timestamps for visualization
        sim_timestamps = list(range(len(simulated_fields)))
        
        # Select an interesting frame in the middle
        time_idx = len(simulated_fields) // 2
        
        # Create flow visualization
        self.fig = create_flow_visualization(
            simulated_fields,
            sim_timestamps,
            self.x_grid, self.y_grid,
            sensor_coordinates,
            time_idx=time_idx,
            param_idx=0,  # Gas
            style='advanced'
        )
        
        # Save the visualization
        output_dir = "../output"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/realtime_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                   dpi=150, bbox_inches='tight')
        
        # Display the figure
        plt.show()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        receiver.stop()
        if self.server:
            sender.stop_server(self.server)
        if self.fig:
            plt.close(self.fig)
        print("Workflow complete")
    
    def run(self):
        """Run the complete workflow"""
        try:
            if not self.start_data_source():
                return False
            
            if self.data_collection_loop():
                simulated_fields, velocity_x, velocity_y = self.run_simulation()
                self.visualize_results(simulated_fields, velocity_x, velocity_y)
            
            return True
        finally:
            self.cleanup()

# Run the workflow when script is executed
if __name__ == "__main__":
    workflow = RealTimeWorkflow()
    workflow.run()