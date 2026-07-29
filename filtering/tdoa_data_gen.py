import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Sensor Geometry
# ==========================================
L_REF = np.array([-3.0, 0.0])  # Pair-1 reference sensor (km)
L_RX  = np.array([3.0, 0.0])   # Pair-1 secondary sensor (km)

# Second TDOA sensor pair, on a baseline perpendicular to the first
# (rotated 90 degrees about the origin). A single range-difference
# measurement only constrains position to a hyperbola -- motion tangential
# to that curve is invisible to the sensor (see the eval analysis: with only
# pair 1, position error was almost entirely tangential to its hyperbola,
# growing to >2km by the end of a rollout). Pair 2's hyperbolas are
# transverse to pair 1's, so the two measurements' null directions don't
# coincide and their intersection localizes position in both dimensions,
# the same way real multilateration systems use 3+ receivers.
L_REF2 = np.array([0.0, -3.0])  # Pair-2 reference sensor (km)
L_RX2  = np.array([0.0, 3.0])   # Pair-2 secondary sensor (km)

MEASUREMENT_STD = 0.200        # 200m measurement noise (km), both pairs
DT = 0.5                       # Time step (s)

# Initial-state distribution -- matches the Gaussian prior used in the
# TDOA simulation example of Crouse, "Particle Flow Solutions Avoiding Stiff
# Integration" (NRL/5340/FR--2021/1), Sec. 5: mean [5, 3] km, diagonal
# covariance 10 km^2 (std ~= 3.16 km). The paper uses this as the filter's
# prior over a fixed true target at [4, 4] km; here it's repurposed as the
# generative distribution over each trajectory's true initial position, to
# inject the same magnitude of positional uncertainty into training.
INIT_MEAN = np.array([5.0, 3.0])  # km
INIT_STD  = np.sqrt(10.0)         # km (per axis, diag covariance)

# ==========================================
# 2. Trajectory Generation
# ==========================================
def generate_tdoa_trajectories(num_trajectories=5000, num_steps=100, dt=DT):
    """
    Generates time-series state and measurement data for TDOA tracking.
    Outputs are formatted for the UncontrolledTrajectoryDataset.
    """
    # Constant Velocity (CV) Motion Model Matrices
    F = np.array([
        [1.0, 0.0,  dt, 0.0],
        [0.0, 1.0, 0.0,  dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Process noise covariance (Q) 
    q_pos, q_vel = 1e-4, 1e-5 
    Q = np.diag([q_pos, q_pos, q_vel, q_vel]) 
    
    # Initialize Data Structures -> Intermediate Shape: (Batch, Sequence, Feature)
    states = np.zeros((num_trajectories, num_steps, 4))
    measurements = np.zeros((num_trajectories, num_steps, 2))  # [pair-1, pair-2] range diffs
    
    for i in range(num_trajectories):
        # Initialize target from the paper's prior distribution (see
        # INIT_MEAN/INIT_STD above), with random velocities
        x_init = INIT_MEAN[0] + np.random.normal(0, INIT_STD)
        y_init = INIT_MEAN[1] + np.random.normal(0, INIT_STD)
        vx_init = np.random.uniform(-0.05, 0.05)
        vy_init = np.random.uniform(-0.05, 0.05)
        
        current_state = np.array([x_init, y_init, vx_init, vy_init])
        
        for k in range(num_steps):
            states[i, k, :] = current_state
            
            # --- Measurement Generation (h(x)) ---
            pos = current_state[0:2]

            # Pair 1
            r_ref = np.linalg.norm(pos - L_REF)
            r_rx  = np.linalg.norm(pos - L_RX)
            z1_true = r_rx - r_ref
            measurements[i, k, 0] = z1_true + np.random.normal(0, MEASUREMENT_STD)

            # Pair 2 (independent noise draw -- separate receiver hardware)
            r_ref2 = np.linalg.norm(pos - L_REF2)
            r_rx2  = np.linalg.norm(pos - L_RX2)
            z2_true = r_rx2 - r_ref2
            measurements[i, k, 1] = z2_true + np.random.normal(0, MEASUREMENT_STD)
            
            # --- State Propagation ---
            process_noise = np.random.multivariate_normal(np.zeros(4), Q)
            current_state = F @ current_state + process_noise
            
    # Convert to PyTorch Tensors and transpose to [N, Features, T] 
    # to comply with UncontrolledTrajectoryDataset API
    states_tensor = torch.tensor(states, dtype=torch.float32).transpose(1, 2)
    measurements_tensor = torch.tensor(measurements, dtype=torch.float32).transpose(1, 2)
    
    return states_tensor, measurements_tensor

# ==========================================
# 3. Save to PyTorch Data File
# ==========================================
def save_dataset(states, measurements, filename="tdoa_tracking_data.pt"):
    """
    Saves the dataset into a dictionary format compatible with the 
    FileTrackingDataModule (use_control=False path).
    """
    data_dict = {
        'true_trajectories': states,       # Shape: [N, state_dim, T]
        'measurements': measurements,      # Shape: [N, obs_dim, T]
        
        # Metadata will be stored automatically in self.norm_stats
        'L_REF': L_REF.tolist(),
        'L_RX': L_RX.tolist(),
        'L_REF2': L_REF2.tolist(),
        'L_RX2': L_RX2.tolist(),
        'MEASUREMENT_STD': MEASUREMENT_STD,
        'INIT_MEAN': INIT_MEAN.tolist(),
        'INIT_STD': INIT_STD
    }
    
    torch.save(data_dict, filename)
    print(f"Dataset saved to '{filename}'")
    print(f"  true_trajectories shape: {data_dict['true_trajectories'].shape}")
    print(f"  measurements shape:      {data_dict['measurements'].shape}")

# ==========================================
# 4. Visualization
# ==========================================
def plot_example_trajectory(states_tensor, measurements_tensor, idx=0, dt=DT):
    """
    Visualizes a single 2D trajectory and its corresponding TDOA measurements.
    Expects tensors in the [N, Features, T] format.
    """
    # Extract the specific trajectory sequence and convert back to numpy
    # state_seq shape: [4, T] -> [x, y, vx, vy]
    state_seq = states_tensor[idx].numpy()
    # meas_seq shape: [2, T] -> [pair-1, pair-2]
    meas_seq = measurements_tensor[idx].numpy()

    num_steps = state_seq.shape[1]
    time_steps = np.arange(num_steps) * dt

    x_pos = state_seq[0, :]
    y_pos = state_seq[1, :]
    z1_noisy = meas_seq[0, :]
    z2_noisy = meas_seq[1, :]

    # Reconstruct true noise-free measurements for the baseline comparison
    true_positions = np.vstack((x_pos, y_pos)).T  # Shape: [T, 2]
    r_ref = np.linalg.norm(true_positions - L_REF, axis=1)
    r_rx  = np.linalg.norm(true_positions - L_RX, axis=1)
    z1_true = r_rx - r_ref
    r_ref2 = np.linalg.norm(true_positions - L_REF2, axis=1)
    r_rx2  = np.linalg.norm(true_positions - L_RX2, axis=1)
    z2_true = r_rx2 - r_ref2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})

    # --- Spatial Trajectory Plot (X vs Y) ---
    ax1.plot(x_pos, y_pos, 'b-', alpha=0.7, label='True Trajectory')
    ax1.scatter(x_pos[0], y_pos[0], c='green', marker='o', s=100, label='Start', zorder=5)
    ax1.scatter(x_pos[-1], y_pos[-1], c='red', marker='X', s=100, label='End', zorder=5)

    # Plot Sensors -- pair 1 (x-axis baseline)
    ax1.scatter(L_REF[0], L_REF[1], c='black', marker='^', s=150, label='Pair-1 Ref Sensor')
    ax1.scatter(L_RX[0], L_RX[1], c='purple', marker='^', s=150, label='Pair-1 Rx Sensor')
    # Pair 2 (y-axis baseline)
    ax1.scatter(L_REF2[0], L_REF2[1], c='dimgray', marker='s', s=150, label='Pair-2 Ref Sensor')
    ax1.scatter(L_RX2[0], L_RX2[1], c='darkorchid', marker='s', s=150, label='Pair-2 Rx Sensor')

    ax1.set_title(f'Target Trajectory in 2D Space (Sample {idx})')
    ax1.set_xlabel('X Position (km)')
    ax1.set_ylabel('Y Position (km)')
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Measurement Plot (Time vs Range Difference) ---
    ax2.plot(time_steps, z1_true, 'k--', linewidth=2, label='True Range Diff (Pair 1)')
    ax2.scatter(time_steps, z1_noisy, c='red', s=10, alpha=0.5, label='Noisy TDOA (Pair 1)')
    ax2.plot(time_steps, z2_true, '--', color='dimgray', linewidth=2, label='True Range Diff (Pair 2)')
    ax2.scatter(time_steps, z2_noisy, c='darkorchid', s=10, alpha=0.5, label='Noisy TDOA (Pair 2)')

    ax2.set_title('TDOA Measurements Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range Difference (km)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('tdoa_sample_trajectory.png')

# ==========================================
# 5. Execution Block
# ==========================================
if __name__ == "__main__":
    print("Generating TDOA dataset...")
    # Generate 5000 samples, 100 timesteps each
    states_t, meas_t = generate_tdoa_trajectories(num_trajectories=5000, num_steps=100, dt=DT)
    
    # Save the dataset for the PyTorch Lightning DataModule
    save_dataset(states_t, meas_t, filename="tdoa_tracking_data.pt")
    
    # Plot the first trajectory to visually verify the geometry and noise
    print("Plotting sample index 0...")
    plot_example_trajectory(states_t, meas_t, idx=0, dt=DT)