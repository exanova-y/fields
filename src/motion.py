import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from visualize import (visualize_stacked_tensor, visualize_simulation_frames, create_animation,
                      visualize_vector_field, create_transport_visualization, create_flow_visualization, create_vector_field_animation)

def simulate_field_motion(grid_tensor, num_steps=20, dt=0.1, diffusion_coef=0.05, velocity_scale=0.02):
    """
    Simulate convection and diffusion of scent field over time.
    
    Args:
        grid_tensor (numpy.ndarray): 4D tensor with shape [time, x, y, param]
                                    where param[0] = gas resistance, param[1] = temperature, param[2] = humidity
        num_steps (int): Number of simulation steps to run
        dt (float): Time step for simulation
        diffusion_coef (float): Coefficient for diffusion term
        velocity_scale (float): Scaling factor for temperature-based velocity
    
    Returns:
        tuple: (simulated_fields, velocity_x, velocity_y) where:
              - simulated_fields is a 4D tensor [num_steps, x, y, param]
              - velocity_x and velocity_y are 3D tensors [num_steps, x, y] with the velocity field
                at each time step
    """
    # Get initial state from the first timestamp
    initial_state = grid_tensor[0].copy()  # Shape: [x, y, param]
    
    # Get grid dimensions
    grid_size_x, grid_size_y, num_params = initial_state.shape
    
    # Create empty array for simulation results
    result = np.zeros((num_steps, grid_size_x, grid_size_y, num_params))
    result[0] = initial_state  # Set initial state
    
    # Create empty arrays to store velocity fields at each time step
    velocity_x_history = np.zeros((num_steps, grid_size_x, grid_size_y))
    velocity_y_history = np.zeros((num_steps, grid_size_x, grid_size_y))
    
    # Precompute grid coordinates for gradient calculation
    x, y = np.meshgrid(np.arange(grid_size_x), np.arange(grid_size_y))
    
    # Run simulation for num_steps
    for step in range(1, num_steps):
        prev_state = result[step-1].copy()
        
        # Create velocity fields from temperature gradient
        # We'll use temperature (param[1]) as a proxy for velocity
        # Higher temperature = higher velocity
        
        # Smooth the temperature field to avoid numerical instability
        smooth_temp = gaussian_filter(prev_state[:, :, 1], sigma=1.0)
        
        # Calculate temperature gradients in x and y directions
        # Central difference approximation
        grad_temp_x = np.zeros_like(smooth_temp)
        grad_temp_y = np.zeros_like(smooth_temp)
        
        # Interior points
        grad_temp_x[1:-1, 1:-1] = (smooth_temp[1:-1, 2:] - smooth_temp[1:-1, :-2]) / 2.0
        grad_temp_y[1:-1, 1:-1] = (smooth_temp[2:, 1:-1] - smooth_temp[:-2, 1:-1]) / 2.0
        
        # Scale gradients to get velocity fields
        velocity_x = velocity_scale * grad_temp_x
        velocity_y = velocity_scale * grad_temp_y
        
        # Store velocity fields for visualization
        velocity_x_history[step] = velocity_x
        velocity_y_history[step] = velocity_y
        
        # Update each parameter
        for param in range(num_params):
            curr_param = prev_state[:, :, param].copy()
            
            # Apply diffusion (Laplacian)
            laplacian = np.zeros_like(curr_param)
            # Interior points only
            laplacian[1:-1, 1:-1] = (
                curr_param[1:-1, 2:] + curr_param[1:-1, :-2] + 
                curr_param[2:, 1:-1] + curr_param[:-2, 1:-1] - 
                4 * curr_param[1:-1, 1:-1]
            )
            
            # Apply diffusion term
            curr_param += diffusion_coef * dt * laplacian
            
            # Apply advection (semi-Lagrangian scheme)
            # Calculate where each particle would have been in the previous step
            x_prev = np.clip(x - dt * velocity_x, 0, grid_size_x - 1.001)
            y_prev = np.clip(y - dt * velocity_y, 0, grid_size_y - 1.001)
            
            # Integer indices
            x0 = np.floor(x_prev).astype(int)
            y0 = np.floor(y_prev).astype(int)
            x1 = x0 + 1
            y1 = y0 + 1
            
            # Ensure indices are within bounds
            x1 = np.minimum(x1, grid_size_x - 1)
            y1 = np.minimum(y1, grid_size_y - 1)
            
            # Weights for bilinear interpolation
            wx = x_prev - x0
            wy = y_prev - y0
            
            # Bilinear interpolation
            top = curr_param[y0, x0] * (1 - wx) + curr_param[y0, x1] * wx
            bottom = curr_param[y1, x0] * (1 - wx) + curr_param[y1, x1] * wx
            advected = top * (1 - wy) + bottom * wy
            
            # Apply convective term
            result[step, :, :, param] = advected
            
            # Apply boundary conditions (zero flux)
            result[step, 0, :, param] = result[step, 1, :, param]
            result[step, -1, :, param] = result[step, -2, :, param]
            result[step, :, 0, param] = result[step, :, 1, param]
            result[step, :, -1, param] = result[step, :, -2, param]
    
    return result, velocity_x_history, velocity_y_history

if __name__ == "__main__":
    from ingestor import load_offline_data, interpolate_grid_with_layers, sensor_coordinates

    # Load sensor data
    sensor_data = load_offline_data(folder_path="data")

    # Create initial tensor
    grid_tensor, timestamps, x_grid, y_grid = interpolate_grid_with_layers(sensor_data)

    # Configure simulation parameters for a longer, more detailed run
    num_simulation_steps = 300
    dt = 0.05  # Smaller time step for stability
    diffusion_coef = 0.03  # Diffusion coefficient
    velocity_scale = 0.025  # Scale temperature gradient to velocity

    print(f"\nRunning simulation with {num_simulation_steps} steps...")
    print(f"Grid shape: {grid_tensor.shape}")

    # Simulate field motion with longer timespan
    simulated_fields, velocity_x, velocity_y = simulate_field_motion(
        grid_tensor, 
        num_steps=num_simulation_steps, 
        dt=dt, 
        diffusion_coef=diffusion_coef,
        velocity_scale=velocity_scale
    )

    print(f"Simulation complete. Result shape: {simulated_fields.shape}")

    # Generate fake timestamps for the simulation timespan
    # Since we have real timestamps only for the first frame
    sim_timestamps = list(range(num_simulation_steps))

    # Create vector field animation showing gas resistance and temperature
    print("\nCreating vector field animation...")
    # Create output directory if it doesn't exist
    import os
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define animation output path
    animation_path = os.path.join(output_dir, "vector_field_animation.gif")

    # Use only a subset of frames to make the animation more manageable
    # Take every 10th frame from the simulation
    frame_step = 6  # Adjust this value to include more or fewer frames
    selected_frames = list(range(0, num_simulation_steps, frame_step))
    print(f"Creating animation with {len(selected_frames)} frames instead of {num_simulation_steps}")

    # Extract the data for only the selected frames to reduce memory usage
    selected_fields = simulated_fields[selected_frames]
    selected_velocity_x = velocity_x[selected_frames]
    selected_velocity_y = velocity_y[selected_frames]
    selected_timestamps = [sim_timestamps[i] for i in selected_frames]

    # Create the animation
    vector_animation = create_vector_field_animation(
        selected_fields,
        selected_velocity_x,
        selected_velocity_y,
        selected_timestamps,
        x_grid,
        y_grid,
        sensor_coordinates,
        fps=10,  # Lower fps for smoother animation and smaller file
        save_path=animation_path
    )

    # Show the animation in an interactive window
    print(f"\nAnimation saved to {animation_path}")
    print("Displaying animation (close the window to continue)...")
    plt.show()

    # Optional: Create still frames of key time points
    # Uncomment if you want to generate these additional visualizations
    """
    # Create vector field visualizations similar to the reference image
    print("\nCreating vector field visualizations...")

    # Select the time steps to visualize
    time_steps = [0, 50, 150, 299]  # Initial, two intermediate, and final states

    for time_idx in time_steps:
        # Create advanced flow visualization (Line Integral Convolution-like)
        fig = create_flow_visualization(
            simulated_fields, 
            sim_timestamps, 
            x_grid, y_grid, 
            sensor_coordinates,
            time_idx=time_idx,
            param_idx=0,  # Gas resistance
            style='advanced'
        )
        plt.savefig(f"flow_visual_gas_{time_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create artistic flow visualization
        fig = create_flow_visualization(
            simulated_fields, 
            sim_timestamps, 
            x_grid, y_grid, 
            sensor_coordinates,
            time_idx=time_idx,
            param_idx=1,  # Temperature
            style='artistic'
        )
        plt.savefig(f"flow_visual_temp_{time_idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Show a few key frames from the simulation
    print("\nCreating visualization of selected frames...")
    key_frames = [0, 50, 100, 150, 200, 250, 299]  # First, last, and some intermediate frames
    visualize_simulation_frames(
        simulated_fields, 
        sim_timestamps, 
        x_grid, y_grid, 
        sensor_coordinates, 
        frames=key_frames, 
        param_idx=0  # 0=gas, 1=temp, 2=humidity
    )
    """