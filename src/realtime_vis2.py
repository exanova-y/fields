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
GRID_SIZE = 128         # Higher resolution for fluid simulation
PARAMS = 3              # Gas, Temperature, Humidity
DATA_DIR = "../data"    # Where receiver.py saves data
SENSOR_ID = "sensor1"   # Default sensor ID
FPS = 60                # Higher frame rate for smoother visualization
INTERP_WINDOW = 2       # Number of seconds to interpolate between

# Fluid simulation parameters
ITERATIONS = 16         # Increased pressure solver iterations for accuracy
DIFFUSION = 0.00005     # Reduced diffusion rate for sharper visuals
VISCOSITY = 0.000005    # Reduced viscosity for more dynamic flow
DT = 0.05               # Smaller time step for stability at high resolution
DECAY = 0.997           # Slower decay for persistent fluid

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
        
        # Additional field for visual interest (vorticity)
        self.vorticity = np.zeros((size, size))
        
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
        
        # Apply vorticity confinement for more swirling motion
        self.apply_vorticity_confinement()
        
        # Diffuse density and temperature
        self.diffuse(self.density, self.density_prev, DIFFUSION, self.dt)
        self.diffuse(self.temp, self.temp_prev, DIFFUSION * 0.1, self.dt)  # Temperature diffuses slower
        
        # Advect density and temperature
        self.advect(self.density, self.density_prev, self.u, self.v, self.dt)
        self.advect(self.temp, self.temp_prev, self.u, self.v, self.dt)
        
        # Calculate vorticity (curl of velocity field)
        self.compute_vorticity()
        
        # Apply decay
        self.u *= DECAY
        self.v *= DECAY
        self.density *= DECAY
        
    def apply_vorticity_confinement(self):
        """Add energy at vortices to create more interesting swirling"""
        # Parameters for vorticity confinement
        vorticity_strength = 10.0
        
        if hasattr(self, 'vorticity') and np.any(self.vorticity):
            # Calculate gradient of vorticity magnitude
            vort_mag = np.abs(self.vorticity)
            
            # Normalized gradient vectors of the vorticity magnitude
            grad_x = np.zeros_like(self.vorticity)
            grad_y = np.zeros_like(self.vorticity)
            
            # Central differences for gradient
            grad_x[1:-1, 1:-1] = (vort_mag[1:-1, 2:] - vort_mag[1:-1, :-2]) * 0.5
            grad_y[1:-1, 1:-1] = (vort_mag[2:, 1:-1] - vort_mag[:-2, 1:-1]) * 0.5
            
            # Normalize
            length = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
            grad_x /= length
            grad_y /= length
            
            # Apply force
            self.u[1:-1, 1:-1] += vorticity_strength * self.dt * (
                grad_y[1:-1, 1:-1] * self.vorticity[1:-1, 1:-1]
            )
            self.v[1:-1, 1:-1] -= vorticity_strength * self.dt * (
                grad_x[1:-1, 1:-1] * self.vorticity[1:-1, 1:-1]
            )
    
    def compute_vorticity(self):
        """Compute vorticity (curl) of the velocity field"""
        # Curl in 2D is just dv/dx - du/dy
        self.vorticity = np.zeros_like(self.u)
        self.vorticity[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) * 0.5 -
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) * 0.5
        )
    
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
        
        # For FPS calculation
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_display = 0
    
    def start_visualization(self):
        """Start the live visualization"""
        # Set up the matplotlib figure with black background
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(10, 10), facecolor='black')
        self.ax = self.fig.add_subplot(111)
        
        # Combined visualization that blends density and temperature
        blended_field = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # RGBA
        self.img = self.ax.imshow(
            blended_field,
            extent=[0, GRID_SIZE-1, 0, GRID_SIZE-1],
            origin='lower',
            interpolation='bilinear'
        )
        
        # Hide axes for cleaner look
        self.ax.axis('off')
        
        # Optional: Add very subtle sensor marker
        for sensor_id, (x, y) in self.sensor_coordinates.items():
            self.ax.plot(x, y, 'o', markersize=3, color='white', alpha=0.3)
        
        # Add minimal timestamp and FPS counter in corner
        self.info_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            color='white', alpha=0.7, fontsize=8, verticalalignment='top'
        )
        
        # Create animation with high frame rate
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=1000/FPS, blit=True,
            cache_frame_data=False  # Prevent memory issues with caching
        )
        
        # Start the data processing thread
        self.processor_thread = threading.Thread(target=self.data_processor_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()

        # Set tight layout and make figure larger (macOS compatible approach)
        self.fig.tight_layout(pad=0)
        try:
            # Try to maximize window if backend supports it
            mng = plt.get_current_fig_manager()
            # Different backends have different ways to maximize
            if hasattr(mng, 'window'):
                # For Qt backend
                mng.window.showMaximized()
            elif hasattr(mng, 'frame'):
                # For wxPython
                mng.frame.Maximize(True)
            elif hasattr(mng, 'resize'):
                # For TkAgg
                mng.resize(*mng.window.maxsize())
            elif hasattr(mng, 'full_screen_toggle'):
                # For some other backends
                mng.full_screen_toggle()
            else:
                # macOS/matplotlib combo likely doesn't support window maximizing
                # Just make the figure larger
                plt.gcf().set_size_inches(12, 12)
        except Exception as e:
            # Fallback to setting a larger size
            plt.gcf().set_size_inches(12, 12)
            print(f"Note: Could not maximize window, using large figure size instead: {e}")
        
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
            
            # Create multiple sources in a pattern
            num_sources = 6
            for i in range(num_sources):
                angle = i * 2 * np.pi / num_sources
                offset = int(radius * 0.7)
                src_x = x + int(offset * np.cos(angle + time.time() * 0.1))
                src_y = y + int(offset * np.sin(angle + time.time() * 0.1))
                
                # Alternate between gas and temperature
                if i % 2 == 0:
                    self.add_source(src_x, src_y, radius // 2, gas, temp, True)
                else:
                    self.add_source(src_x, src_y, radius // 2, temp, gas, False)
    
    def add_source(self, x, y, radius, value1, value2, is_density_primary):
        """Add a fluid source at location with specific properties"""
        # Ensure coordinates are within bounds
        if not (radius <= x < GRID_SIZE-radius and radius <= y < GRID_SIZE-radius):
            return
            
        # Add with Gaussian falloff
        for i in range(max(1, x-radius), min(GRID_SIZE-1, x+radius)):
            for j in range(max(1, y-radius), min(GRID_SIZE-1, y+radius)):
                # Distance from source
                dist = np.sqrt((i-x)**2 + (j-y)**2)
                if dist < radius:
                    # Gaussian falloff
                    factor = np.exp(-(dist/radius)**2 * 4)
                    
                    # Add quantities based on pattern
                    if is_density_primary:
                        self.fluid.add_density(i, j, value1 * factor * 0.3)
                        self.fluid.add_temperature(i, j, value2 * factor * 0.2)
                    else:
                        self.fluid.add_density(i, j, value2 * factor * 0.2)
                        self.fluid.add_temperature(i, j, value1 * factor * 0.3)
                    
                    # Add swirling velocity
                    angle = np.arctan2(j-y, i-x) + time.time() * (0.5 if is_density_primary else -0.5)
                    power = max(value1, value2) * factor * 0.2
                    vx = np.cos(angle) * power
                    vy = np.sin(angle) * power
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
        # Update frame counter and calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:  # Update FPS once per second
            self.fps_display = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            
        # Create a blended visualization
        blended = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # RGBA
        
        # Normalize field values for visualization
        max_density = max(0.01, np.max(self.fluid.density))
        max_temp = max(0.01, np.max(self.fluid.temp))
        max_vorticity = max(0.01, np.max(np.abs(self.fluid.vorticity)))
        
        # Create colorful blended visualization inspired by the first image
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Normalized values
                d = self.fluid.density[i, j] / max_density
                t = self.fluid.temp[i, j] / max_temp
                v = abs(self.fluid.vorticity[i, j]) / max_vorticity
                
                # Vibrant color mapping:
                # - Density maps to green/yellow
                # - Temperature maps to red/orange
                # - Blend based on relative strengths
                
                # Base color (deep blue for background)
                r, g, b = 0.0, 0.0, 0.1
                
                # Add density contribution (green-yellow)
                r += d * 0.7 * t  # Yellow when both density and temp are high
                g += d * 0.5
                
                # Add temperature contribution (red-orange)
                r += t * 0.6
                g += t * 0.3 * d
                
                # Add vorticity for visual interest (purplish glow)
                r += v * 0.3
                b += v * 0.5
                
                # Ensure values are in range [0,1]
                r = min(1.0, r)
                g = min(1.0, g)
                b = min(1.0, b)
                
                # Calculate alpha (transparency)
                alpha = min(1.0, (d + t) * 2.0)
                alpha = max(0.0, alpha)  # Ensure non-negative
                
                # Add slight ambient glow
                alpha = max(alpha, 0.05)
                
                # Store the RGBA values
                blended[i, j, 0] = r
                blended[i, j, 1] = g
                blended[i, j, 2] = b
                blended[i, j, 3] = alpha
        
        # Apply a slight blur for glow effect
        for c in range(4):
            blended[:,:,c] = gaussian_filter(blended[:,:,c], sigma=0.5)
        
        # Update the image
        self.img.set_array(blended)
        
        # Update information text
        if self.latest_data:
            gas = self.latest_data['gas']
            temp = self.latest_data['temp']
            humidity = self.latest_data['humidity']
            now = datetime.now().strftime('%H:%M:%S')
            info = f"FPS: {self.fps_display:.1f} | {now} | Gas: {gas:.1f} | Temp: {temp:.1f}°C | Humidity: {humidity:.1f}%"
        else:
            now = datetime.now().strftime('%H:%M:%S')
            info = f"FPS: {self.fps_display:.1f} | {now} | Waiting for data..."
            
        self.info_text.set_text(info)
        
        # Return the updated artists
        return [self.img, self.info_text]
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        plt.close(self.fig)
        print("Visualization stopped")

# When run directly
if __name__ == "__main__":
    print("Starting high-resolution fluid simulation at 60 FPS...")
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
