#!/usr/bin/env python3
"""
Real-time field visualization that continuously updates as new sensor data arrives,
using a stable fluids simulation for realistic fluid dynamics.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import os
from datetime import datetime, timedelta
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

# Import our modules
import sender

# Constants
GRID_SIZE = 64          # Higher resolution for fluid simulation
PARAMS = 3              # Gas, Temperature, Humidity
DATA_DIR = "../data"    # Where receiver.py saves data
SENSOR_ID = "sensor1"   # Default sensor ID
FPS = 30                # Target frames per second for visualization
INTERP_WINDOW = 2       # Number of seconds to interpolate between

# Fluid simulation parameters
ITERATIONS = 4          # Pressure solver iterations
DIFFUSION = 0.0001      # Diffusion rate
VISCOSITY = 0.00001     # Viscosity
DT = 0.1                # Time step
DECAY = 0.999           # Velocity decay factor

class FluidSim:
    """2D Stable Fluids simulator based on Jos Stam's paper"""
    
    def __init__(self, size):
        self.size = size
        self.dt = DT
        
        # Velocity fields
        self.u = np.zeros((size, size))  # x velocity
        self.v = np.zeros((size, size))  # y velocity
        self.u_prev = np.zeros((size, size))
        self.v_prev = np.zeros((size, size))
        
        # Density field (scalar quantities)
        self.density = np.zeros((size, size))
        self.density_prev = np.zeros((size, size))
        
        # Temperature field
        self.temp = np.zeros((size, size))
        self.temp_prev = np.zeros((size, size))
        
        # Boundary conditions - 1 for fluid cells, 0 for solid boundaries
        self.boundary = np.ones((size, size))
        
        # Make boundaries solid (0)
        self.boundary[0, :] = 0
        self.boundary[-1, :] = 0
        self.boundary[:, 0] = 0
        self.boundary[:, -1] = 0
    
    def add_density(self, x, y, amount):
        """Add density at position (x,y)"""
        if 1 <= x < self.size-1 and 1 <= y < self.size-1:
            self.density[y, x] += amount
    
    def add_velocity(self, x, y, amount_x, amount_y):
        """Add velocity at position (x,y)"""
        if 1 <= x < self.size-1 and 1 <= y < self.size-1:
            self.u[y, x] += amount_x
            self.v[y, x] += amount_y
    
    def add_temperature(self, x, y, amount):
        """Add temperature at position (x,y)"""
        if 1 <= x < self.size-1 and 1 <= y < self.size-1:
            self.temp[y, x] += amount
    
    def diffuse(self, field, prev_field, diffusion, dt):
        """Diffuse field using Gauss-Seidel relaxation"""
        a = dt * diffusion * (self.size - 2) * (self.size - 2)
        
        for k in range(ITERATIONS):
            field[1:-1, 1:-1] = (
                prev_field[1:-1, 1:-1] + 
                a * (
                    field[0:-2, 1:-1] + 
                    field[2:, 1:-1] + 
                    field[1:-1, 0:-2] + 
                    field[1:-1, 2:]
                )
            ) / (1 + 4 * a)
            
            self.set_boundary(field)
    
    def advect(self, field, prev_field, u, v, dt):
        """Advect field through velocity field using semi-Lagrangian method"""
        dt0 = dt * (self.size - 2)
        
        # For each cell, trace particle back in time
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                # Calculate previous position
                x = j - dt0 * u[i, j]
                y = i - dt0 * v[i, j]
                
                # Clamp to grid boundaries
                x = max(0.5, min(self.size - 1.5, x))
                y = max(0.5, min(self.size - 1.5, y))
                
                # Integer indices for bilinear interpolation
                i0 = int(y)
                j0 = int(x)
                i1 = i0 + 1
                j1 = j0 + 1
                
                # Interpolation weights
                s1 = x - j0
                s0 = 1 - s1
                t1 = y - i0
                t0 = 1 - t1
                
                # Bilinear interpolation
                field[i, j] = (
                    t0 * (s0 * prev_field[i0, j0] + s1 * prev_field[i0, j1]) +
                    t1 * (s0 * prev_field[i1, j0] + s1 * prev_field[i1, j1])
                )
        
        self.set_boundary(field)
    
    def project(self, u, v):
        """Project velocity field to be mass-conserving (divergence free)"""
        # Calculate divergence
        div = np.zeros((self.size, self.size))
        p = np.zeros((self.size, self.size))
        
        # Compute divergence
        div[1:-1, 1:-1] = -0.5 * (
            u[1:-1, 2:] - u[1:-1, :-2] +
            v[2:, 1:-1] - v[:-2, 1:-1]
        ) / self.size
        
        self.set_boundary(div)
        
        # Solve Poisson equation for pressure
        for k in range(ITERATIONS):
            p[1:-1, 1:-1] = (
                div[1:-1, 1:-1] + 
                p[0:-2, 1:-1] + 
                p[2:, 1:-1] + 
                p[1:-1, 0:-2] + 
                p[1:-1, 2:]
            ) / 4
            
            self.set_boundary(p)
        
        # Subtract pressure gradient from velocity
        u[1:-1, 1:-1] -= 0.5 * self.size * (p[1:-1, 2:] - p[1:-1, :-2])
        v[1:-1, 1:-1] -= 0.5 * self.size * (p[2:, 1:-1] - p[:-2, 1:-1])
        
        self.set_boundary(u)
        self.set_boundary(v)
    
    def set_boundary(self, field):
        """Apply boundary conditions"""
        # Edges
        field[0, :] = field[1, :]    # Top edge
        field[-1, :] = field[-2, :]  # Bottom edge
        field[:, 0] = field[:, 1]    # Left edge
        field[:, -1] = field[:, -2]  # Right edge
        
        # Corners
        field[0, 0] = 0.5 * (field[1, 0] + field[0, 1])      # Top-left
        field[0, -1] = 0.5 * (field[1, -1] + field[0, -2])   # Top-right
        field[-1, 0] = 0.5 * (field[-2, 0] + field[-1, 1])   # Bottom-left
        field[-1, -1] = 0.5 * (field[-2, -1] + field[-1, -2]) # Bottom-right
    
    def step(self):
        """Perform one simulation step"""
        # Swap buffers
        self.u_prev, self.u = self.u.copy(), self.u_prev
        self.v_prev, self.v = self.v.copy(), self.v_prev
        self.density_prev, self.density = self.density.copy(), self.density_prev
        self.temp_prev, self.temp = self.temp.copy(), self.temp_prev
        
        # Apply buoyancy forces from temperature to velocity
        self.apply_buoyancy()
        
        # Diffuse velocity
        self.diffuse(self.u, self.u_prev, VISCOSITY, self.dt)
        self.diffuse(self.v, self.v_prev, VISCOSITY, self.dt)
        
        # Project to enforce incompressibility
        self.project(self.u, self.v)
        
        # Advect velocity
        self.u_prev, self.u = self.u.copy(), self.u_prev
        self.v_prev, self.v = self.v.copy(), self.v_prev
        self.advect(self.u, self.u_prev, self.u_prev, self.v_prev, self.dt)
        self.advect(self.v, self.v_prev, self.u_prev, self.v_prev, self.dt)
        
        # Project again
        self.project(self.u, self.v)
        
        # Diffuse density and temperature
        self.diffuse(self.density, self.density_prev, DIFFUSION, self.dt)
        self.diffuse(self.temp, self.temp_prev, DIFFUSION * 0.1, self.dt)  # Temperature diffuses slower
        
        # Advect density and temperature
        self.advect(self.density, self.density_prev, self.u, self.v, self.dt)
        self.advect(self.temp, self.temp_prev, self.u, self.v, self.dt)
        
        # Apply decay
        self.u *= DECAY
        self.v *= DECAY
        self.density *= DECAY
    
    def apply_buoyancy(self):
        """Apply buoyancy forces from temperature gradient"""
        # Simple buoyancy model: hot regions rise (negative y-velocity)
        # Scale factor for buoyancy
        buoyancy = 0.5
        
        # Temperature gradient creates upward force
        self.v -= buoyancy * self.temp * self.dt

class RealtimeFieldVis:
    """Real-time field visualization that updates as new data arrives"""
    
    def __init__(self):
        self.running = True
        self.sensor_readings = []  # List of sensor readings in time order
        self.timestamps = [datetime.now().isoformat()]
        self.lock = threading.Lock()
        self.fig = None
        self.ax = None
        self.field_img = None  # For updating the visualization
        
        # Initialize fluid simulator
        self.fluid = FluidSim(GRID_SIZE)
        
        # Sensor coordinates (just one for testing)
        self.sensor_coordinates = {
            SENSOR_ID: (GRID_SIZE//2, GRID_SIZE//2)  # Center of grid
        }
        
        # Store the latest data
        self.latest_data = None
    
    def start_visualization(self):
        """Start the live visualization"""
        # Set up the matplotlib figure
        self.fig, axs = plt.subplots(1, 2, figsize=(14, 7), 
                                     gridspec_kw={'width_ratios': [1, 1]})
        
        # Left plot for gas concentration
        self.ax_density = axs[0]
        self.density_img = self.ax_density.imshow(
            self.fluid.density,
            extent=[0, GRID_SIZE-1, 0, GRID_SIZE-1],
            cmap='viridis',
            origin='lower'
        )
        self.ax_density.set_title('Gas Concentration')
        plt.colorbar(self.density_img, ax=self.ax_density)
        
        # Right plot for velocity field
        self.ax_vel = axs[1]
        self.temp_img = self.ax_vel.imshow(
            self.fluid.temp,
            extent=[0, GRID_SIZE-1, 0, GRID_SIZE-1],
            cmap='hot',
            origin='lower'
        )
        self.ax_vel.set_title('Temperature Field')
        plt.colorbar(self.temp_img, ax=self.ax_vel)
        
        # Add sensor location marker
        for sensor_id, (x, y) in self.sensor_coordinates.items():
            for ax in [self.ax_density, self.ax_vel]:
                ax.plot(x, y, 'ro', markersize=8)
                ax.text(x, y, sensor_id, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.7))
        
        # Add timestamp info
        self.title = self.fig.suptitle(f"Fluid Simulation - {datetime.now().strftime('%H:%M:%S')}")
        
        # Create animation with high frame rate
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=1000/FPS, blit=False
        )
        
        # Start the data processing thread
        self.processor_thread = threading.Thread(target=self.data_processor_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        plt.show()
    
    def read_latest_data(self):
        """Read the latest data saved by receiver.py"""
        try:
            # Find the most recent data file
            date_str = datetime.now().strftime("%Y%m%d")
            filepath = os.path.join(DATA_DIR, f"sensor-{date_str}.txt")
            
            if not os.path.exists(filepath):
                return None
            
            # Read all lines to get recent data
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                return None
            
            # Get current time for filtering
            now = datetime.now()
            window_start = now - timedelta(seconds=INTERP_WINDOW)
            
            # Parse recent lines within our interpolation window
            recent_readings = []
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        # Parse timestamp
                        timestamp = datetime.fromisoformat(parts[0])
                        
                        # Only use data from last INTERP_WINDOW seconds
                        if timestamp >= window_start:
                            reading = {
                                'timestamp': parts[0],
                                'timestamp_obj': timestamp,
                                'sensor_id': parts[1],
                                'gas': float(parts[2]),
                                'temp': float(parts[3]),
                                'humidity': float(parts[4])
                            }
                            recent_readings.append(reading)
                    except (ValueError, TypeError):
                        continue
            
            # Return the most recent readings
            if recent_readings:
                recent_readings.sort(key=lambda x: x['timestamp_obj'])
                return recent_readings[-1]  # Return most recent reading
                
        except Exception as e:
            print(f"Error reading latest data: {e}")
        
        return None
    
    def update_simulation(self, data):
        """Update fluid simulation with new sensor data"""
        if not data:
            return
        
        # Get normalized values (0-1 range for better simulation)
        gas = min(1.0, data['gas'] / 500.0)  # Assume 500 is max gas reading
        temp = (data['temp'] - 15) / 15.0    # Normalize temp (15-30C to 0-1)
        temp = max(0, min(1, temp))          # Clamp to 0-1
        
        # Add to simulation at sensor location
        for sensor_id, (x, y) in self.sensor_coordinates.items():
            # Add density around sensor location with Gaussian falloff
            radius = GRID_SIZE // 8
            for i in range(max(1, x-radius), min(GRID_SIZE-1, x+radius)):
                for j in range(max(1, y-radius), min(GRID_SIZE-1, y+radius)):
                    # Distance from sensor
                    dist = np.sqrt((i-x)**2 + (j-y)**2)
                    if dist < radius:
                        # Gaussian falloff
                        factor = np.exp(-(dist/radius)**2 * 4)
                        
                        # Add quantities based on sensor reading
                        self.fluid.add_density(i, j, gas * factor * 0.2)
                        self.fluid.add_temperature(i, j, temp * factor * 0.2)
                        
                        # Add velocity based on temperature (hot rises)
                        vx = np.sin(time.time() + i/10) * temp * factor * 0.1
                        vy = -temp * factor * 0.2  # Negative y means upward
                        self.fluid.add_velocity(i, j, vx, vy)
    
    def data_processor_loop(self):
        """Background loop that processes new data"""
        last_processed_time = None
        
        while self.running:
            # Read the latest data
            data = self.read_latest_data()
            
            if data and (last_processed_time is None or data['timestamp'] != last_processed_time):
                # New data available
                self.latest_data = data
                last_processed_time = data['timestamp']
                
                # Update fluid simulation with new data
                self.update_simulation(data)
            
            # Run fluid simulation step (always advance simulation)
            self.fluid.step()
            
            # Sleep to prevent high CPU usage
            time.sleep(0.01)
    
    def update_plot(self, frame):
        """Update the visualization with the latest simulation state"""
        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Update density visualization
        self.density_img.set_array(self.fluid.density)
        vmax_density = max(0.1, np.max(self.fluid.density))
        self.density_img.set_clim(vmin=0, vmax=vmax_density)
        
        # Update temperature visualization
        self.temp_img.set_array(self.fluid.temp)
        vmax_temp = max(0.1, np.max(self.fluid.temp))
        self.temp_img.set_clim(vmin=0, vmax=vmax_temp)
        
        # Update velocity field visualization
        if hasattr(self, 'quiver'):
            self.quiver.remove()
        
        # Downsample velocity field for visualization
        step = max(1, GRID_SIZE // 20)
        X, Y = np.meshgrid(np.arange(0, GRID_SIZE, step), np.arange(0, GRID_SIZE, step))
        U = self.fluid.u[::step, ::step]
        V = self.fluid.v[::step, ::step]
        
        # Only show velocity vectors in one plot
        self.quiver = self.ax_vel.quiver(
            X, Y, U, V, 
            color='white', scale=5, alpha=0.6,
            headwidth=3, headlength=3
        )
        
        # Update title with sensor data
        if self.latest_data:
            gas = self.latest_data['gas']
            temp = self.latest_data['temp']
            humidity = self.latest_data['humidity']
            self.title.set_text(
                f"Fluid Simulation [{current_time}] - Gas: {gas:.1f} Temp: {temp:.1f}°C Humidity: {humidity:.1f}%"
            )
        else:
            self.title.set_text(f"Fluid Simulation - {current_time} - Waiting for data...")
        
        return [self.density_img, self.temp_img, self.quiver, self.title]
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        plt.close(self.fig)
        print("Visualization stopped")

# When run directly
if __name__ == "__main__":
    print("Starting real-time fluid simulation at 30 FPS...")
    print("Make sure receiver.py is already running in another terminal!")
    print("Data will be read from: ", DATA_DIR)
    
    try:
        vis = RealtimeFieldVis()
        vis.start_visualization()
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    finally:
        if 'vis' in locals():
            vis.cleanup()
