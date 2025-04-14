import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class KalmanFilter:
    """
    A simple implementation of the Kalman Filter for tracking a moving object
    in 2D space with position and velocity state.
    """
    def __init__(self, dt, process_noise_std, measurement_noise_std):
        """
        Initialize the Kalman filter.
        
        Parameters:
        dt (float): Time step between measurements
        process_noise_std (float): Standard deviation of the process noise
        measurement_noise_std (float): Standard deviation of the measurement noise
        """
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State transition model (Physics model: constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation model (We only observe positions, not velocities)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = process_noise_std**2
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ])
        
        # Measurement noise covariance
        r = measurement_noise_std**2
        self.R = np.array([
            [r, 0],
            [0, r]
        ])
        
        # Initial covariance estimate
        self.P = np.eye(4)
        
    def predict(self):
        """
        Prediction step: Project the state and covariance forward in time.
        
        Returns:
        tuple: Predicted state vector and prediction of where measurement should be
        """
        # Predicted state estimate
        self.state = self.F @ self.state
        
        # Predicted error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Return predicted position only
        return self.state, self.H @ self.state
    
    def update(self, measurement):
        """
        Update step: Incorporate new measurement to improve state estimate.
        
        Parameters:
        measurement (numpy.ndarray): Observed position [x, y]
        
        Returns:
        tuple: Updated state vector and innovation (measurement residual)
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Optimal Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Updated state estimate
        self.state = self.state + K @ y
        
        # Updated error covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state, y
    
    def get_position(self):
        """Return the current position estimate."""
        return self.state[:2]
    
    def get_position_covariance(self):
        """Return the position covariance (uncertainty in position)."""
        return self.P[:2, :2]

def generate_trajectory(num_steps, dt, true_initial_state, process_noise_std):
    """
    Generate a true trajectory and noisy measurements.
    
    Parameters:
    num_steps (int): Number of time steps to simulate
    dt (float): Time step between measurements
    true_initial_state (numpy.ndarray): Initial state [x, y, vx, vy]
    process_noise_std (float): Standard deviation of process noise
    
    Returns:
    tuple: True states and noisy measurements
    """
    # Create state transition matrix (constant velocity model)
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Initialize state and storage
    true_state = true_initial_state.copy()
    true_states = [true_state.copy()]
    
    # Simulate the trajectory
    for _ in range(num_steps - 1):
        # Add some process noise (random acceleration)
        process_noise = np.random.normal(0, process_noise_std, 4)
        process_noise[0] = 0  # No direct noise on position
        process_noise[1] = 0  # No direct noise on position
        
        # Update state with noise
        true_state = F @ true_state + process_noise
        true_states.append(true_state.copy())
    
    return np.array(true_states)

def create_measurements(true_states, measurement_noise_std):
    """Create noisy measurements from true states."""
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    measurements = []
    for state in true_states:
        # Extract position from state
        true_measurement = H @ state
        
        # Add measurement noise
        noise = np.random.normal(0, measurement_noise_std, 2)
        noisy_measurement = true_measurement + noise
        measurements.append(noisy_measurement)
    
    return np.array(measurements)

def plot_confidence_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """
    Plot a confidence ellipse representing the covariance matrix.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw on
    mean (numpy.ndarray): The position [x, y]
    cov (numpy.ndarray): The 2x2 covariance matrix
    n_std (float): The number of standard deviations to plot
    **kwargs: Additional arguments passed to Ellipse
    
    Returns:
    matplotlib.patches.Ellipse: The drawn ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    # Get the eigenvalues directly
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Largest eigenvalue first
    vals, vecs = zip(*sorted(zip(eigenvals, eigenvecs.T), key=lambda x: x[0], reverse=True))
    theta = np.degrees(np.arctan2(vecs[0][1], vecs[0][0]))
    
    # Width and height are "full" widths, not radius
    width, height = 2 * n_std * np.sqrt(vals)
    
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        **kwargs
    )
    
    return ax.add_patch(ellipse)

def run_kalman_filter_simulation():
    """Set up and run the simulation."""
    # Simulation parameters
    dt = 0.1  # Time step
    num_steps = 100  # Number of steps to simulate
    process_noise_std = 0.1  # Process noise standard deviation
    measurement_noise_std = 0.5  # Measurement noise standard deviation
    
    # True initial state: [x, y, vx, vy]
    true_initial_state = np.array([0.0, 0.0, 1.0, 0.5])
    
    # Generate true trajectory and noisy measurements
    true_states = generate_trajectory(num_steps, dt, true_initial_state, process_noise_std)
    measurements = create_measurements(true_states, measurement_noise_std)
    
    # Initialize Kalman filter
    kf = KalmanFilter(dt, process_noise_std, measurement_noise_std)
    
    # Set initial state
    kf.state = np.array([measurements[0][0], measurements[0][1], 0.0, 0.0])
    
    # Storage for filter estimates
    filtered_states = []
    predicted_states = []
    position_covariances = []
    
    # Run the Kalman filter
    for t in range(num_steps):
        # Predict
        predicted_state, predicted_measurement = kf.predict()
        predicted_states.append(predicted_measurement)
        
        # Update
        updated_state, innovation = kf.update(measurements[t])
        filtered_states.append(kf.get_position())
        position_covariances.append(kf.get_position_covariance())
    
    filtered_states = np.array(filtered_states)
    predicted_states = np.array(predicted_states)
    
    return true_states, measurements, filtered_states, predicted_states, position_covariances

def create_static_plot(true_states, measurements, filtered_states):
    """Create a static plot showing the entire trajectory."""
    plt.figure(figsize=(10, 6))
    
    # Plot the true trajectory and measurements
    plt.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True Trajectory')
    plt.scatter(measurements[:, 0], measurements[:, 1], s=10, c='r', alpha=0.5, label='Measurements')
    plt.plot(filtered_states[:, 0], filtered_states[:, 1], 'g-', label='Kalman Filter Estimate')
    
    plt.grid(True)
    plt.legend()
    plt.title('Kalman Filter Tracking')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def create_animation(true_states, measurements, filtered_states, predicted_states, position_covariances):
    """Create an animation of the Kalman filter in action."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot settings
    ax.set_xlim([np.min(true_states[:, 0]) - 1, np.max(true_states[:, 0]) + 1])
    ax.set_ylim([np.min(true_states[:, 1]) - 1, np.max(true_states[:, 1]) + 1])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Kalman Filter Tracking')
    ax.grid(True)
    
    # Plots for true path, measurements, and filter estimate
    true_line, = ax.plot([], [], 'b-', label='True Path')
    meas_points, = ax.plot([], [], 'rx', label='Measurements')
    filter_line, = ax.plot([], [], 'g-', label='Kalman Filter Estimate')
    
    # Current state indicators
    current_true, = ax.plot([], [], 'bo', markersize=6)
    current_meas, = ax.plot([], [], 'ro', markersize=6)
    current_est, = ax.plot([], [], 'go', markersize=6)
    
    # Ellipse for uncertainty visualization
    uncertainty_ellipse = None
    
    ax.legend()
    
    def init():
        """Initialize animation."""
        true_line.set_data([], [])
        meas_points.set_data([], [])
        filter_line.set_data([], [])
        current_true.set_data([], [])
        current_meas.set_data([], [])
        current_est.set_data([], [])
        return true_line, meas_points, filter_line, current_true, current_meas, current_est
    
    def update(frame):
        """Update animation for each frame."""
        # Update paths
        true_line.set_data(true_states[:frame+1, 0], true_states[:frame+1, 1])
        meas_points.set_data(measurements[:frame+1, 0], measurements[:frame+1, 1])
        filter_line.set_data(filtered_states[:frame+1, 0], filtered_states[:frame+1, 1])
        
        # Update current positions
        current_true.set_data([true_states[frame, 0]], [true_states[frame, 1]])
        current_meas.set_data([measurements[frame, 0]], [measurements[frame, 1]])
        current_est.set_data([filtered_states[frame, 0]], [filtered_states[frame, 1]])
        
        # Update uncertainty ellipse
        if 'uncertainty_ellipse' in locals() and uncertainty_ellipse is not None:
            uncertainty_ellipse.remove()
        
        # Plot covariance ellipse
        uncertainty_ellipse = plot_confidence_ellipse(
            ax, 
            filtered_states[frame], 
            position_covariances[frame],
            n_std=2.0,
            facecolor='green',
            alpha=0.2
        )
        
        return true_line, meas_points, filter_line, current_true, current_meas, current_est, uncertainty_ellipse
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(true_states), 
                        init_func=init, blit=False, interval=100)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def main():
    """Main function to run the simulation and display results."""
    # Run the simulation
    true_states, measurements, filtered_states, predicted_states, position_covariances = run_kalman_filter_simulation()
    
    # Create static plot of the entire trajectory
    create_static_plot(true_states, measurements, filtered_states)
    
    # Create animation of the filtering process
    ani = create_animation(true_states, measurements, filtered_states, predicted_states, position_covariances)
    
    # Plot errors over time
    plt.figure(figsize=(12, 6))
    
    # Position error calculation
    pos_error_true = np.linalg.norm(true_states[:, :2] - filtered_states, axis=1)
    pos_error_meas = np.linalg.norm(measurements - true_states[:, :2], axis=1)
    
    plt.subplot(1, 2, 1)
    plt.plot(pos_error_true, 'g-', label='Kalman Filter Error')
    plt.plot(pos_error_meas, 'r-', label='Measurement Error')
    plt.grid(True)
    plt.legend()
    plt.title('Position Error Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Error (Distance)')
    
    # Plot uncertainty (trace of covariance matrix) over time
    plt.subplot(1, 2, 2)
    uncertainty = [np.trace(cov) for cov in position_covariances]
    plt.plot(uncertainty, 'b-')
    plt.grid(True)
    plt.title('Position Uncertainty Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Uncertainty (Trace of Covariance)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()