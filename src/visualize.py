import numpy as np
import matplotlib.pyplot as plt

def visualize_stacked_tensor(grid_tensor, timestamps, x_grid, y_grid, sensor_coordinates, time_idx=0):
    """
    Create a single visualization showing all three parameters of the stacked tensor.
    
    Args:
        grid_tensor (numpy.ndarray): 4D tensor with shape [time, x, y, param]
        timestamps (list): List of timestamps corresponding to the time dimension
        x_grid (numpy.ndarray): Array of x coordinates
        y_grid (numpy.ndarray): Array of y coordinates
        sensor_coordinates (dict): Dictionary mapping sensor IDs to (x,y) coordinates
        time_idx (int): Index of the timestamp to visualize
    """
    if grid_tensor is None or len(timestamps) == 0:
        print("No valid grid tensor to visualize")
        return
    
    if time_idx >= len(timestamps):
        print(f"Time index {time_idx} out of range. Using time index 0.")
        time_idx = 0
    
    # Parameters to visualize
    param_names = ['Gas Resistance', 'Temperature', 'Humidity']
    param_units = ['Ω', '°C', '%']
    cmaps = ['viridis', 'hot', 'Blues']
    
    # Create a figure with three subplots in one row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each parameter
    for i, (param_name, param_unit, cmap) in enumerate(zip(param_names, param_units, cmaps)):
        data = grid_tensor[time_idx, :, :, i]
        
        # Create a meshgrid for contourf
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Plot as a filled contour map
        contour = axes[i].contourf(X, Y, data, 15, cmap=cmap)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=axes[i])
        cbar.set_label(f'{param_name} ({param_unit})')
        
        # Set titles and labels
        axes[i].set_title(f'{param_name} at Time {timestamps[time_idx]}')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        
        # Plot sensor locations
        for sensor_id, coord in sensor_coordinates.items():
            axes[i].plot(coord[0], coord[1], 'ro', markersize=8)
            axes[i].text(coord[0], coord[1], sensor_id, fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    return fig

def visualize_simulation_frames(simulation_results, timestamps, x_grid, y_grid, sensor_coordinates, 
                               frames=None, param_idx=0):
    """
    Visualize multiple frames of a simulation.
    
    Args:
        simulation_results (numpy.ndarray): 4D tensor with shape [time, x, y, param]
        timestamps (list): List of timestamps corresponding to the time dimension
        x_grid (numpy.ndarray): Array of x coordinates
        y_grid (numpy.ndarray): Array of y coordinates
        sensor_coordinates (dict): Dictionary mapping sensor IDs to (x,y) coordinates
        frames (list): List of time indices to visualize (default: first, middle, last)
        param_idx (int): Index of parameter to visualize (0=gas, 1=temp, 2=humidity)
    """
    if simulation_results is None:
        print("No valid simulation results to visualize")
        return
    
    # Parameters to visualize
    param_names = ['Gas Resistance', 'Temperature', 'Humidity']
    param_units = ['Ω', '°C', '%']
    
    # If frames not specified, show first, middle, and last
    num_steps = simulation_results.shape[0]
    if frames is None:
        frames = [0, num_steps//2, num_steps-1]
    
    # Create a figure with subplots for each frame
    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
    if len(frames) == 1:
        axes = [axes]  # Make sure axes is iterable
    
    # Plot each frame
    for i, time_idx in enumerate(frames):
        if time_idx >= num_steps:
            print(f"Time index {time_idx} out of range. Using time index 0.")
            time_idx = 0
            
        data = simulation_results[time_idx, :, :, param_idx]
        
        # Create a meshgrid for contourf
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Plot as a filled contour map
        contour = axes[i].contourf(X, Y, data, 15, cmap='viridis')
        
        # Set titles and labels
        time_label = timestamps[time_idx] if timestamps and time_idx < len(timestamps) else time_idx
        axes[i].set_title(f'{param_names[param_idx]} at Time {time_label}')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        
        # Plot sensor locations
        for sensor_id, coord in sensor_coordinates.items():
            axes[i].plot(coord[0], coord[1], 'ro', markersize=8)
            axes[i].text(coord[0], coord[1], sensor_id, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=axes)
    cbar.set_label(f'{param_names[param_idx]} ({param_units[param_idx]})')
    
    # Adjust layout
    plt.tight_layout()
    return fig

def create_animation(simulation_results, timestamps, x_grid, y_grid, sensor_coordinates, fps=10, save_path=None):
    """
    Create an animation of the simulation results.
    
    Args:
        simulation_results (numpy.ndarray): 4D tensor with shape [time, x, y, param]
        timestamps (list): List of timestamps corresponding to the time dimension
        x_grid (numpy.ndarray): Array of x coordinates
        y_grid (numpy.ndarray): Array of y coordinates
        sensor_coordinates (dict): Dictionary mapping sensor IDs to (x,y) coordinates
        fps (int): Frames per second in the animation
        save_path (str): Path to save the animation (optional)
    
    Returns:
        matplotlib.animation.Animation: Animation object
    """
    from matplotlib.animation import FuncAnimation
    
    num_steps, grid_size_x, grid_size_y, _ = simulation_results.shape
    
    # Create a figure for animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    param_names = ['Gas Resistance', 'Temperature', 'Humidity']
    param_units = ['Ω', '°C', '%']
    cmaps = ['viridis', 'hot', 'Blues']
    
    # Create meshgrid for contour plots
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Get data ranges for each parameter for consistent color scales
    vmin = [np.min(simulation_results[:, :, :, i]) for i in range(3)]
    vmax = [np.max(simulation_results[:, :, :, i]) for i in range(3)]
    
    # Initialize plots
    contours = []
    for i, (param, unit, cmap) in enumerate(zip(param_names, param_units, cmaps)):
        contour = axes[i].contourf(
            X, Y,
            simulation_results[0, :, :, i], 
            15,
            cmap=cmap,
            vmin=vmin[i],
            vmax=vmax[i]
        )
        axes[i].set_title(f'{param} ({unit})')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        
        # Plot sensor locations (only need to do this once)
        for sensor_id, coord in sensor_coordinates.items():
            axes[i].plot(coord[0], coord[1], 'ro', markersize=6)
            axes[i].text(coord[0], coord[1], sensor_id, fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7))
        
        plt.colorbar(contour, ax=axes[i])
        contours.append(contour)
    
    # Update function for animation
    def update(frame):
        # Update each subplot
        for i, ax in enumerate(axes):
            # Clear previous contours by clearing the axes and redrawing
            ax.clear()
            
            # Create new contour plot
            new_contour = ax.contourf(
                X, Y, 
                simulation_results[frame, :, :, i],
                15,
                cmap=cmaps[i],
                vmin=vmin[i], 
                vmax=vmax[i]
            )
            
            # Add sensor locations back
            for sensor_id, coord in sensor_coordinates.items():
                ax.plot(coord[0], coord[1], 'ro', markersize=6)
                ax.text(coord[0], coord[1], sensor_id, fontsize=8,
                      bbox=dict(facecolor='white', alpha=0.7))
            
            # Update title with timestamp
            time_label = timestamps[frame] if timestamps and frame < len(timestamps) else frame
            ax.set_title(f'{param_names[i]} at Time {time_label}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            # We don't need to return anything since we're modifying the axes directly
        
        # Return a list of artists that changed (required for FuncAnimation)
        return []
    
    # Tight layout before animation starts
    plt.tight_layout()
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=num_steps, interval=1000/fps, blit=False)
    
    # Save if requested
    if save_path:
        ani.save(save_path, writer='pillow', fps=fps)
    
    return ani

def visualize_vector_field(x_grid, y_grid, vector_field_x, vector_field_y, 
                          sensor_coordinates=None, scale=20, density=1, title="Vector Field", 
                          cmap='viridis', background=None):
    """
    Visualize a vector field using arrows or streamlines.
    
    Args:
        x_grid (numpy.ndarray): X coordinates
        y_grid (numpy.ndarray): Y coordinates
        vector_field_x (numpy.ndarray): X component of vector field
        vector_field_y (numpy.ndarray): Y component of vector field
        sensor_coordinates (dict): Dictionary mapping sensor IDs to (x,y) coordinates
        scale (float): Scale factor for arrows
        density (float): Density of streamlines
        title (str): Plot title
        cmap (str): Colormap for color coding
        background (numpy.ndarray): Optional scalar field to use as background
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create coordinate grid
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate vector magnitudes for color coding
    magnitude = np.sqrt(vector_field_x**2 + vector_field_y**2)
    
    # If a background scalar field is provided, show it
    if background is not None:
        contour = ax.contourf(X, Y, background, 50, cmap=cmap, alpha=0.7)
        plt.colorbar(contour, ax=ax, label='Scalar Field')
    
    # Plot streamlines
    strm = ax.streamplot(X, Y, vector_field_x, vector_field_y, 
                        color=magnitude, 
                        linewidth=1.5,
                        cmap=cmap,
                        density=density,
                        arrowstyle='->')
    
    # Add colorbar for vector magnitude
    plt.colorbar(strm.lines, ax=ax, label='Vector Magnitude')
    
    # Plot sensor locations
    if sensor_coordinates:
        for sensor_id, coord in sensor_coordinates.items():
            ax.plot(coord[0], coord[1], 'ro', markersize=8)
            ax.text(coord[0], coord[1], sensor_id, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def create_transport_visualization(scalar_field, vector_field_x, vector_field_y, 
                                  x_grid, y_grid, sensor_coordinates=None, 
                                  num_iterations=20, diffusion=0.1):
    """
    Create a visualization of scalar transport using Line Integral Convolution-like technique.
    
    Args:
        scalar_field (numpy.ndarray): Scalar field to advect (2D array)
        vector_field_x (numpy.ndarray): X component of vector field (2D array)
        vector_field_y (numpy.ndarray): Y component of vector field (2D array)
        x_grid (numpy.ndarray): X coordinates (1D array)
        y_grid (numpy.ndarray): Y coordinates (1D array)
        sensor_coordinates (dict): Dictionary mapping sensor IDs to (x,y) coordinates
        num_iterations (int): Number of advection iterations
        diffusion (float): Diffusion strength to apply during advection
    """
    from scipy.ndimage import gaussian_filter
    
    # Create coordinate meshgrid
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Normalize the vector field for advection
    magnitude = np.sqrt(vector_field_x**2 + vector_field_y**2)
    max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    
    norm_vx = vector_field_x / max_mag
    norm_vy = vector_field_y / max_mag
    
    # Create a copy of the scalar field for advection
    advected_field = scalar_field.copy()
    
    # Create random noise pattern for visualization
    noise = np.random.random(scalar_field.shape)
    # Smooth the noise a bit
    noise = gaussian_filter(noise, sigma=0.5)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # First subplot: Scalar field with streamlines
    ax0 = axes[0]
    contour = ax0.contourf(X, Y, scalar_field, 50, cmap='viridis')
    plt.colorbar(contour, ax=ax0, label='Original Scalar Field')
    
    # Add streamlines
    strm = ax0.streamplot(X, Y, vector_field_x, vector_field_y,
                         color='white', linewidth=1.2, density=1.5)
    
    ax0.set_title('Vector Field and Original Scalar')
    ax0.set_xlabel('X Position')
    ax0.set_ylabel('Y Position')
    ax0.set_aspect('equal')
    
    # Advect the noise field along the vector field
    advected_noise = noise.copy()
    for i in range(num_iterations):
        # Get grid coordinates
        y_indices, x_indices = np.indices(advected_noise.shape)
        
        # Calculate where each point came from
        x_from = x_indices - norm_vx
        y_from = y_indices - norm_vy
        
        # Clip to valid grid locations
        x_from = np.clip(x_from, 0, advected_noise.shape[1] - 1.001)
        y_from = np.clip(y_from, 0, advected_noise.shape[0] - 1.001)
        
        # Get integer indices for bilinear interpolation
        x0 = np.floor(x_from).astype(int)
        y0 = np.floor(y_from).astype(int)
        x1 = np.minimum(x0 + 1, advected_noise.shape[1] - 1)
        y1 = np.minimum(y0 + 1, advected_noise.shape[0] - 1)
        
        # Calculate weights
        wx = x_from - x0
        wy = y_from - y0
        
        # Bilinear interpolation
        top = advected_noise[y0, x0] * (1 - wx) + advected_noise[y0, x1] * wx
        bottom = advected_noise[y1, x0] * (1 - wx) + advected_noise[y1, x1] * wx
        advected_noise = top * (1 - wy) + bottom * wy
        
        # Add diffusion
        advected_noise = gaussian_filter(advected_noise, sigma=diffusion)
    
    # Composite the advected noise with the scalar field
    transport_vis = advected_noise * scalar_field
    
    # Second subplot: Transport visualization
    ax1 = axes[1]
    transport_img = ax1.imshow(transport_vis, cmap='inferno', origin='lower',
                              extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])
    plt.colorbar(transport_img, ax=ax1, label='Transport Visualization')
    
    ax1.set_title('Advected Scalar Transport')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_aspect('equal')
    
    # Plot sensor locations
    if sensor_coordinates:
        for sensor_id, coord in sensor_coordinates.items():
            for ax in axes:
                ax.plot(coord[0], coord[1], 'ro', markersize=8)
                ax.text(coord[0], coord[1], sensor_id, fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_flow_visualization(simulation_results, timestamps, x_grid, y_grid, sensor_coordinates,
                             time_idx=0, param_idx=0, style='advanced'):
    """
    Create sophisticated flow visualization for simulation results.
    
    Args:
        simulation_results (numpy.ndarray): 4D tensor with shape [time, x, y, param]
        timestamps (list): List of timestamps
        x_grid (numpy.ndarray): X coordinates
        y_grid (numpy.ndarray): Y coordinates
        sensor_coordinates (dict): Dictionary mapping sensor IDs to (x,y) coordinates
        time_idx (int): Index of time step to visualize
        param_idx (int): Index of parameter to visualize
        style (str): Visualization style ('basic', 'advanced', 'artistic')
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    param_names = ['Gas Resistance', 'Temperature', 'Humidity']
    
    # Extract data for the requested time step
    data = simulation_results[time_idx, :, :, param_idx]
    
    # Create coordinate grid
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate gradient of the scalar field to use as vector field
    grad_x, grad_y = np.gradient(data)
    
    if style == 'basic':
        # Basic streamline visualization
        fig = visualize_vector_field(x_grid, y_grid, grad_x, grad_y, 
                                   sensor_coordinates, 
                                   title=f'{param_names[param_idx]} Flow at Time {timestamps[time_idx]}',
                                   background=data)
    
    elif style == 'artistic':
        # More artistic visualization with randomized flow patterns
        from scipy.ndimage import gaussian_filter
        
        # Smooth the gradient fields
        grad_x_smooth = gaussian_filter(grad_x, sigma=1.0)
        grad_y_smooth = gaussian_filter(grad_y, sigma=1.0)
        
        # Add some randomness to create more interesting flow patterns
        noise_x = np.random.normal(0, 1, grad_x.shape)
        noise_y = np.random.normal(0, 1, grad_y.shape)
        
        # Smooth the noise
        noise_x = gaussian_filter(noise_x, sigma=2.0)
        noise_y = gaussian_filter(noise_y, sigma=2.0)
        
        # Combine gradient and noise
        flow_x = grad_x_smooth + 0.2 * noise_x
        flow_y = grad_y_smooth + 0.2 * noise_y
        
        # Create artistic flow visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use a more colorful and artistic color scheme
        contour = ax.contourf(X, Y, data, 100, cmap='plasma', alpha=0.7)
        
        # Add many streamlines with varying colors
        seed_points = np.random.rand(100, 2)
        seed_points = seed_points * np.array([data.shape[1], data.shape[0]])
        
        strm = ax.streamplot(X, Y, flow_x, flow_y, 
                           color=data, 
                           linewidth=1.0,
                           cmap='viridis',
                           density=2.5,
                           arrowstyle='->')
        
        plt.colorbar(contour, ax=ax, label=f'{param_names[param_idx]}')
        
        # Plot sensor locations
        if sensor_coordinates:
            for sensor_id, coord in sensor_coordinates.items():
                ax.plot(coord[0], coord[1], 'ro', markersize=8)
                ax.text(coord[0], coord[1], sensor_id, fontsize=12, 
                      bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f'Artistic Flow: {param_names[param_idx]} at Time {timestamps[time_idx]}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_aspect('equal')
        
    else:  # 'advanced' is default
        # Use the transport visualization for advanced style
        fig = create_transport_visualization(data, grad_x, grad_y, 
                                           x_grid, y_grid, 
                                           sensor_coordinates)
        plt.suptitle(f'{param_names[param_idx]} Transport at Time {timestamps[time_idx]}', 
                    fontsize=16)
    
    return fig

def create_vector_field_animation(simulation_results, velocity_x, velocity_y, timestamps, 
                                 x_grid, y_grid, sensor_coordinates, fps=10, save_path=None):
    """
    Create an animation showing the evolution of vector fields for gas resistance and temperature.
    
    Args:
        simulation_results (numpy.ndarray): 4D tensor with shape [time, x, y, param]
        velocity_x (numpy.ndarray): 3D tensor with shape [time, x, y] for x velocity
        velocity_y (numpy.ndarray): 3D tensor with shape [time, x, y] for y velocity
        timestamps (list): List of timestamps
        x_grid (numpy.ndarray): X coordinates
        y_grid (numpy.ndarray): Y coordinates
        sensor_coordinates (dict): Dictionary mapping sensor IDs to (x,y) coordinates
        fps (int): Frames per second in the animation
        save_path (str): Path to save the animation (optional)
    
    Returns:
        matplotlib.animation.Animation: Animation object
    """
    from matplotlib.animation import FuncAnimation
    from scipy.ndimage import gaussian_filter
    import matplotlib.contour
    import matplotlib as mpl
    
    # Set a non-interactive backend for saving if we're saving to file
    if save_path:
        print("Using Agg backend for saving animation...")
        mpl.use('Agg')
    
    num_steps = min(len(timestamps), simulation_results.shape[0])
    
    # Create figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Create coordinate meshgrid
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate gradients for the first frame (will be updated in animation)
    gas_data = simulation_results[0, :, :, 0]  # Gas resistance
    temp_data = simulation_results[0, :, :, 1]  # Temperature
    
    gas_grad_x, gas_grad_y = np.gradient(gas_data)
    temp_grad_x, temp_grad_y = np.gradient(temp_data)
    
    # Ensure consistent color scales across frames
    vmin_gas = np.min(simulation_results[:, :, :, 0])
    vmax_gas = np.max(simulation_results[:, :, :, 0])
    vmin_temp = np.min(simulation_results[:, :, :, 1])
    vmax_temp = np.max(simulation_results[:, :, :, 1])
    
    velocity_mag = np.sqrt(velocity_x**2 + velocity_y**2)
    vmin_vel = np.min(velocity_mag)
    vmax_vel = np.max(velocity_mag)
    
    # Plot titles
    titles = [
        'Gas Resistance with Field Lines',
        'Temperature with Field Lines',
        'Gas Gradient Vector Field',
        'Temperature-Driven Velocity Field'
    ]
    
    # If saving and there are too many frames, reduce dpi and quality
    dpi = 100
    if save_path and num_steps > 50:
        dpi = 80  # Lower quality for larger animations
    
    # Add timestamp display
    timestamp_text = fig.text(0.5, 0.95, f'Time: {timestamps[0]}', 
                            ha='center', va='center', fontsize=12)
    
    # Create a safer update function that doesn't rely on specific artists
    def update(frame):
        if frame % 10 == 0:
            print(f"Processing frame {frame}/{num_steps}...")
            
        timestamp_text.set_text(f'Time: {timestamps[frame]}')
        
        # Get data for current frame
        gas_data = simulation_results[frame, :, :, 0]
        temp_data = simulation_results[frame, :, :, 1]
        
        # Calculate gradients
        gas_grad_x, gas_grad_y = np.gradient(gas_data)
        temp_grad_x, temp_grad_y = np.gradient(temp_data)
        
        # Smooth gradients for better visualization
        gas_grad_x = gaussian_filter(gas_grad_x, sigma=0.5)
        gas_grad_y = gaussian_filter(gas_grad_y, sigma=0.5)
        temp_grad_x = gaussian_filter(temp_grad_x, sigma=0.5)
        temp_grad_y = gaussian_filter(temp_grad_y, sigma=0.5)
        
        # Clear all axes and redraw everything - most reliable approach
        for i, ax in enumerate(axes):
            ax.clear()
            
            if i == 0:  # Gas concentration with field lines
                contour_gas = ax.contourf(X, Y, gas_data, 20, cmap='viridis',
                                         vmin=vmin_gas, vmax=vmax_gas)
                ax.streamplot(X, Y, gas_grad_x, gas_grad_y, color='white', linewidth=0.8, density=1)
                ax.set_title(titles[i])
                
            elif i == 1:  # Temperature with field lines
                contour_temp = ax.contourf(X, Y, temp_data, 20, cmap='hot',
                                          vmin=vmin_temp, vmax=vmax_temp)
                ax.streamplot(X, Y, temp_grad_x, temp_grad_y, color='white', linewidth=0.8, density=1)
                ax.set_title(titles[i])
                
            elif i == 2:  # Gas gradient vector field
                quiv_gas = ax.quiver(X[::1, ::1], Y[::1, ::1], 
                                   gas_grad_x[::1, ::1], gas_grad_y[::1, ::1],
                                   np.sqrt(gas_grad_x[::1, ::1]**2 + gas_grad_y[::1, ::1]**2),
                                   cmap='viridis', scale=30)
                ax.set_title(titles[i])
                
            elif i == 3:  # Temperature-driven velocity vectors
                quiv_vel = ax.quiver(X[::1, ::1], Y[::1, ::1], 
                                   velocity_x[frame, ::1, ::1], velocity_y[frame, ::1, ::1],
                                   np.sqrt(velocity_x[frame, ::1, ::1]**2 + velocity_y[frame, ::1, ::1]**2),
                                   cmap='coolwarm', scale=30)
                ax.set_title(titles[i])
            
            # Set common properties
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_aspect('equal')
            
            # Redraw sensor locations - simplified for animation speed
            if sensor_coordinates:
                for sensor_id, coord in sensor_coordinates.items():
                    ax.plot(coord[0], coord[1], 'ro', markersize=4)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        return [timestamp_text]
    
    # If saving, use a more efficient approach for large animations
    if save_path:
        print(f"Saving animation to {save_path}...")
        import os
        import tempfile
        
        # Create a temporary directory to store frames
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary directory for frames: {temp_dir}")
            
            # Generate and save individual frames
            for i in range(num_steps):
                # Update the figure
                update(i)
                
                # Save the frame
                frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
                plt.savefig(frame_file, dpi=dpi, bbox_inches='tight')
                
                # Clear the figure for the next frame to save memory
                if i % 10 == 0:
                    plt.clf()
                    plt.figure(fig.number)
            
            # Use imagemagick to create the animation from frames if available
            try:
                import subprocess
                output_path = os.path.abspath(save_path)
                frame_path = os.path.join(temp_dir, "frame_*.png")
                
                print("Combining frames into animation...")
                # Try with ImageMagick if available
                try:
                    cmd = ["convert", "-delay", f"{100/fps}", "-loop", "0", 
                           frame_path, output_path]
                    subprocess.run(cmd, check=True)
                    print(f"Animation saved to {output_path} using ImageMagick")
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Try with ffmpeg if ImageMagick failed
                    try:
                        frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
                        cmd = ["ffmpeg", "-framerate", str(fps), "-i", frame_pattern, 
                               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", 
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                               output_path.replace(".gif", ".mp4")]
                        subprocess.run(cmd, check=True)
                        print(f"Animation saved to {output_path.replace('.gif', '.mp4')} using ffmpeg")
                    except (subprocess.SubprocessError, FileNotFoundError):
                        print("Neither ImageMagick nor ffmpeg available. Using matplotlib...")
                        ani = FuncAnimation(fig, update, frames=num_steps, interval=1000/fps, blit=False)
                        ani.save(save_path, writer='pillow', fps=fps, dpi=dpi)
            except Exception as e:
                print(f"Error creating animation: {e}")
                print("Falling back to matplotlib animation...")
                ani = FuncAnimation(fig, update, frames=num_steps, interval=1000/fps, blit=False)
                ani.save(save_path, writer='pillow', fps=fps, dpi=dpi)
                
        print("Animation creation complete!")
        return None  # No animation object to return as we've already saved it
    else:
        # For display, create and return an animation object
        ani = FuncAnimation(fig, update, frames=num_steps, interval=1000/fps, blit=False)
        return ani