"""
Range-Only 2D Tracking — persistent bimodal ambiguity dataset.

Two fixed sensors each report only their (noisy) Euclidean distance to the
target — no bearing. A single range reading constrains the target to a
circle; two simultaneous range readings constrain it to the intersection of
two circles, which generically has exactly two solutions: the true position
and its mirror image reflected across the line through the two sensors.

Because reflecting any point across the line through S_REF and S_RX
preserves distance to both sensors, and the NCV motion model (isotropic
process noise, no handedness) is itself symmetric under that same
reflection, the ambiguity is *persistent* — the mirror trajectory is
measurement-indistinguishable from the true one for the entire sequence,
not just transiently. A correctly-behaving filter should carry both
hypotheses the whole way through with roughly equal weight.

IMPORTANT — this only shows up in training if particles/modes are allowed
to actually explore the mirror point. EmpiricalParticleFilter/GMM models in
this repo initialize particles at the *true* state plus small noise
(init_noise_std), so with a small init_noise_std the swarm will simply
never discover the second mode (it's a disconnected local optimum that
local process-noise diffusion won't reach on its own). --init_noise_std at
training time must be set comparable to the true/mirror separation
distance printed by this script — see MIRROR_SEPARATION diagnostic below.

Output format matches FileTrackingDataModule's use_control=False contract
(dataset.py): a dict with 'true_trajectories' [N, state_dim, T] and
'measurements' [N, obs_dim, T]; state_dim/obs_dim are inferred from these
shapes at load time, no other keys or dataset_type string are required.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Sensor Geometry
# ==========================================
S_REF = np.array([-3.0, 0.0])  # Sensor 1 (km)
S_RX  = np.array([3.0, 0.0])   # Sensor 2 (km)
MEASUREMENT_STD = 0.200        # 200m range measurement noise (km), i.i.d. per sensor
DT = 0.5                       # Time step (s)


# ==========================================
# 2. Mirror-hypothesis geometry (ground truth for eval)
# ==========================================
def reflect_across_line(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Reflect `point` across the infinite line through `a` and `b`.
    Works for any two distinct sensor positions, not just colinear-with-origin
    placements — reflection across the line through both sensors always
    preserves distance to both, regardless of where they sit.
    """
    d = b - a
    t = np.dot(point - a, d) / np.dot(d, d)
    proj = a + t * d
    return 2 * proj - point


def get_mirror_hypotheses(pos: np.ndarray, s_ref: np.ndarray = S_REF,
                           s_rx: np.ndarray = S_RX) -> tuple:
    """Returns (true_pos, mirror_pos) — the two positions consistent with
    the same pair of range readings from s_ref/s_rx."""
    return pos, reflect_across_line(pos, s_ref, s_rx)


# ==========================================
# 3. Trajectory Generation
# ==========================================
def generate_range_trajectories(num_trajectories=5000, num_steps=100, dt=DT):
    """
    Generates time-series state and measurement data for range-only tracking.
    Outputs are formatted for the UncontrolledTrajectoryDataset.

    measurements[..., 0] = range to S_REF
    measurements[..., 1] = range to S_RX
    """
    # Constant Velocity (NCV) Motion Model Matrices — must stay NCV (not CT)
    # for the mirror ambiguity to persist for the whole sequence; see module
    # docstring.
    F = np.array([
        [1.0, 0.0,  dt, 0.0],
        [0.0, 1.0, 0.0,  dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    q_pos, q_vel = 1e-4, 1e-5
    Q = np.diag([q_pos, q_pos, q_vel, q_vel])

    states = np.zeros((num_trajectories, num_steps, 4))
    measurements = np.zeros((num_trajectories, num_steps, 2))
    mirror_seps = np.zeros(num_trajectories)  # true/mirror separation at t=0, for diagnostics

    for i in range(num_trajectories):
        x_init = 4.0 + np.random.normal(0, 0.5)
        y_init = 4.0 + np.random.normal(0, 0.5)
        vx_init = np.random.uniform(-0.05, 0.05)
        vy_init = np.random.uniform(-0.05, 0.05)

        current_state = np.array([x_init, y_init, vx_init, vy_init])

        pos0 = current_state[0:2]
        mirror0 = reflect_across_line(pos0, S_REF, S_RX)
        mirror_seps[i] = np.linalg.norm(pos0 - mirror0)

        for k in range(num_steps):
            states[i, k, :] = current_state

            # --- Measurement Generation (h(x)) ---
            pos = current_state[0:2]
            r_ref = np.linalg.norm(pos - S_REF)
            r_rx  = np.linalg.norm(pos - S_RX)

            measurements[i, k, 0] = r_ref + np.random.normal(0, MEASUREMENT_STD)
            measurements[i, k, 1] = r_rx  + np.random.normal(0, MEASUREMENT_STD)

            # --- State Propagation ---
            process_noise = np.random.multivariate_normal(np.zeros(4), Q)
            current_state = F @ current_state + process_noise

    states_tensor = torch.tensor(states, dtype=torch.float32).transpose(1, 2)
    measurements_tensor = torch.tensor(measurements, dtype=torch.float32).transpose(1, 2)

    print(f"  True/mirror separation at t=0: mean={mirror_seps.mean():.3f} km, "
          f"min={mirror_seps.min():.3f} km, max={mirror_seps.max():.3f} km")
    print(f"  -> set --init_noise_std to at least ~{mirror_seps.mean():.2f} "
          f"(mean separation) when training, or the second mode will never "
          f"be explored (see module docstring).")

    return states_tensor, measurements_tensor


# ==========================================
# 4. Save to PyTorch Data File
# ==========================================
def save_dataset(states, measurements, filename="range_only_tracking_data.pt"):
    """
    Saves the dataset into a dictionary format compatible with the
    FileTrackingDataModule (use_control=False path).
    """
    data_dict = {
        'true_trajectories': states,       # Shape: [N, state_dim=4, T]
        'measurements': measurements,      # Shape: [N, obs_dim=2, T]  (r_ref, r_rx)

        # Metadata stored automatically in self.norm_stats
        'S_REF': S_REF.tolist(),
        'S_RX': S_RX.tolist(),
        'MEASUREMENT_STD': MEASUREMENT_STD,
    }

    torch.save(data_dict, filename)
    print(f"Dataset saved to '{filename}'")
    print(f"  true_trajectories shape: {data_dict['true_trajectories'].shape}")
    print(f"  measurements shape:      {data_dict['measurements'].shape}")


# ==========================================
# 5. Visualization
# ==========================================
def plot_example_trajectory(states_tensor, measurements_tensor, idx=0, dt=DT):
    """
    Visualizes a single 2D trajectory, its mirror-image trajectory, and the
    corresponding range measurements. Expects tensors in [N, Features, T].
    """
    state_seq = states_tensor[idx].numpy()   # [4, T]
    meas_seq  = measurements_tensor[idx].numpy()  # [2, T]

    num_steps = state_seq.shape[1]
    time_steps = np.arange(num_steps) * dt

    x_pos = state_seq[0, :]
    y_pos = state_seq[1, :]
    r_ref_noisy = meas_seq[0, :]
    r_rx_noisy  = meas_seq[1, :]

    true_positions = np.vstack((x_pos, y_pos)).T  # [T, 2]
    r_ref_true = np.linalg.norm(true_positions - S_REF, axis=1)
    r_rx_true  = np.linalg.norm(true_positions - S_RX, axis=1)

    mirror_positions = np.array([
        reflect_across_line(p, S_REF, S_RX) for p in true_positions
    ])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

    # --- Spatial Trajectory Plot (X vs Y) ---
    ax1.plot(x_pos, y_pos, 'b-', alpha=0.7, label='True Trajectory')
    ax1.plot(mirror_positions[:, 0], mirror_positions[:, 1], 'r--', alpha=0.7,
              label='Mirror Trajectory (measurement-indistinguishable)')
    ax1.scatter(x_pos[0], y_pos[0], c='green', marker='o', s=100, label='Start', zorder=5)
    ax1.scatter(x_pos[-1], y_pos[-1], c='blue', marker='X', s=100, label='End (true)', zorder=5)
    ax1.scatter(mirror_positions[-1, 0], mirror_positions[-1, 1], c='red', marker='X', s=100,
                label='End (mirror)', zorder=5)

    ax1.scatter(S_REF[0], S_REF[1], c='black', marker='^', s=150, label='Sensor S_REF')
    ax1.scatter(S_RX[0], S_RX[1], c='purple', marker='^', s=150, label='Sensor S_RX')
    ax1.plot([S_REF[0], S_RX[0]], [S_REF[1], S_RX[1]], 'k:', alpha=0.4, label='Reflection axis')

    ax1.set_title(f'Range-Only Tracking — True vs. Mirror Trajectory (Sample {idx})')
    ax1.set_xlabel('X Position (km)')
    ax1.set_ylabel('Y Position (km)')
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=8)

    # --- Measurement Plot (Time vs Range) ---
    ax2.plot(time_steps, r_ref_true, 'k--', linewidth=1.5, label='True range to S_REF')
    ax2.scatter(time_steps, r_ref_noisy, c='black', s=8, alpha=0.5, label='Noisy range to S_REF')
    ax2.plot(time_steps, r_rx_true, 'm--', linewidth=1.5, label='True range to S_RX')
    ax2.scatter(time_steps, r_rx_noisy, c='magenta', s=8, alpha=0.5, label='Noisy range to S_RX')

    ax2.set_title('Range Measurements Over Time (identical for true and mirror position)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range (km)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('range_only_sample_trajectory.png')


# ==========================================
# 6. Execution Block
# ==========================================
if __name__ == "__main__":
    print("Generating range-only tracking dataset...")
    states_t, meas_t = generate_range_trajectories(num_trajectories=5000, num_steps=100, dt=DT)

    save_dataset(states_t, meas_t, filename="range_only_tracking_data.pt")

    print("Plotting sample index 0...")
    plot_example_trajectory(states_t, meas_t, idx=0, dt=DT)
