"""
Fork / Maneuvering-Target Tracking — resolving bimodal ambiguity dataset.

A target moves in a straight line, then starting at a per-trajectory random
timestep (fork_step, sampled from FORK_STEP_RANGE) ramps its heading by
+TURN_ANGLE/2 ("left") or -TURN_ANGLE/2 ("right") relative to its pre-fork
heading over RAMP_STEPS steps, chosen 50/50 at random, before continuing
straight at the new heading. Position is observed with noise; nothing in
the measurement reveals which branch was taken.

Two design choices exist specifically to avoid the model learning shortcuts
that don't reflect genuine measurement-driven inference of the turn:
  - fork_step is randomized per trajectory (not fixed), so the network
    can't get away with keying off a fixed temporal/sequence-index
    position instead of the actual incoming measurements.
  - the heading change is ramped over RAMP_STEPS instead of instantaneous,
    matching the *character* of the acoustic dataset's dynamics (data_gen.py
    uses a discrete NCV model whose process noise covariance has nonzero
    velocity-block entries, so heading drifts continuously every step —
    never a sharp corner). physics_predictor has only ever been exercised
    against that smooth regime in this codebase; asking it to reproduce an
    instantaneous discontinuity is a much harder, out-of-distribution
    function class for the same small residual-MLP architecture.

Unlike the range-only dataset's mirror ambiguity (two hypotheses separated
by a fixed, externally-set distance from t=0), the two branches here start
at *exactly* the same position (the fork point) and only diverge going
forward — separation grows roughly linearly with time-since-fork once the
ramp completes. This means the untaken branch is always local and
reachable: it costs nothing to maintain both hypotheses immediately after
the fork, unlike range-only's mirror point which needed informed
initialization to ever be discovered under this codebase's models (they
always start a rollout at/near the true state, not somewhere they'd have
to travel a large fixed distance to reach). Ambiguity should also *resolve*
naturally, once the two branches separate enough relative to
MEASUREMENT_STD that a filter can tell which one the actual observations
are following.

Output format matches FileTrackingDataModule's use_control=False contract
(dataset.py): 'true_trajectories' [N, state_dim=4, T] with state
[x, y, vx, vy], 'measurements' [N, obs_dim=2, T] with raw noisy [x, y]
position readings. Per-trajectory fork_step is saved as plain metadata
('fork_steps', a python list — not a tensor, so it lands in
FileTrackingDataModule.norm_stats automatically) for inspection at eval
time; it is never fed to the model.
"""
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from range_only_data_gen import reflect_across_line

# ==========================================
# 1. Configuration
# ==========================================
DT = 0.5                    # Time step (s)
SPEED = 1.0                  # Constant speed along each leg (km per unit time)
FORK_STEP_RANGE = (20, 50)   # Per-trajectory fork_step sampled uniformly from this range
RAMP_STEPS = 5                # Heading transitions linearly over this many steps, starting at fork_step
TURN_ANGLE_DEG = 12.0         # Total angular separation between the two branches
                               # (each branch is +/- TURN_ANGLE_DEG/2 off the pre-fork heading)
MEASUREMENT_STD = 0.30        # Position measurement noise std (km), i.i.d. per axis
PROCESS_NOISE_STD = 0.01      # Small per-step position process noise (km)


# ==========================================
# 2. Alternative-branch geometry (ground truth for eval)
# ==========================================
def get_alternative_branch(true_traj_phys: np.ndarray, fork_step: int) -> np.ndarray:
    """
    true_traj_phys: [T, 4] physical-units [x, y, vx, vy] trajectory — the
    branch that was actually taken.
    fork_step: this trajectory's actual fork start index (per-trajectory,
    stored in the dataset's 'fork_steps' metadata — no longer a global
    constant).

    Returns [T, 2] positions for the untaken branch. This is a pointwise
    reflection of each true position (from fork_step onward) across the
    line through the fork point in the pre-fork heading direction. That
    reflection is exact regardless of whether the turn is instantaneous or
    ramped: the untaken branch's heading at any step k is by construction
    the angular reflection of the true heading at k about the pre-fork
    heading (heading0 - (true_heading(k) - heading0)), which is exactly
    what reflecting the velocity vector achieves; since reflection is an
    isometry and both branches accumulate displacement from the same fixed
    fork point, reflecting the realized position trajectory pointwise
    reproduces exactly what independently regenerating the mirror branch
    (with reflected noise draws) would have produced.
    """
    T = true_traj_phys.shape[0]
    fork_point = true_traj_phys[fork_step, :2]
    pre_fork_vel = true_traj_phys[fork_step - 1, 2:4]
    axis_b = fork_point + pre_fork_vel  # second point defining the reflection line

    alt = true_traj_phys[:, :2].copy()
    for t in range(fork_step, T):
        alt[t] = reflect_across_line(true_traj_phys[t, :2], fork_point, axis_b)
    return alt


# ==========================================
# 3. Trajectory Generation
# ==========================================
def generate_fork_trajectories(num_trajectories=5000, num_steps=100, dt=DT):
    """
    measurements[..., 0:2] = noisy [x, y] position
    branch_taken[i] = 0 ('left', +turn) or 1 ('right', -turn) — metadata
    only, never fed to the model; ground truth for eval.
    fork_steps[i] = this trajectory's randomized fork start index —
    metadata only, never fed to the model; for eval inspection.
    """
    states = np.zeros((num_trajectories, num_steps, 4))
    measurements = np.zeros((num_trajectories, num_steps, 2))
    branch_taken = np.zeros(num_trajectories, dtype=int)
    fork_steps = np.random.randint(FORK_STEP_RANGE[0], FORK_STEP_RANGE[1] + 1,
                                    size=num_trajectories)

    turn_rad = np.deg2rad(TURN_ANGLE_DEG)

    for i in range(num_trajectories):
        fork_step = int(fork_steps[i])

        pos = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
        ])
        heading0 = np.random.uniform(0, 2 * np.pi)

        branch = np.random.choice([0, 1])
        branch_taken[i] = branch
        turn_sign = 1.0 if branch == 0 else -1.0

        for k in range(num_steps):
            if k < fork_step:
                heading = heading0
            elif k < fork_step + RAMP_STEPS:
                frac = (k - fork_step + 1) / RAMP_STEPS
                heading = heading0 + turn_sign * frac * turn_rad / 2.0
            else:
                heading = heading0 + turn_sign * turn_rad / 2.0
            vel = SPEED * np.array([np.cos(heading), np.sin(heading)])

            states[i, k, 0:2] = pos
            states[i, k, 2:4] = vel
            measurements[i, k, :] = pos + np.random.normal(0, MEASUREMENT_STD, size=2)

            process_noise = np.random.normal(0, PROCESS_NOISE_STD, size=2)
            pos = pos + vel * dt + process_noise

    states_tensor = torch.tensor(states, dtype=torch.float32).transpose(1, 2)
    measurements_tensor = torch.tensor(measurements, dtype=torch.float32).transpose(1, 2)

    # Diagnostic: branch separation vs. steps-since-fork (aligned per-trajectory
    # to each one's own randomized fork_step), and how it compares to
    # measurement noise (governs how long genuine ambiguity should last).
    sample_offsets = [1, 3, 5, 10, 20]
    print(f"  fork_step range: [{FORK_STEP_RANGE[0]}, {FORK_STEP_RANGE[1]}]  "
          f"(mean {fork_steps.mean():.1f})   RAMP_STEPS={RAMP_STEPS}")
    print("  Branch separation vs. steps-since-fork (mean over dataset, per-trajectory fork_step):")
    for off in sample_offsets:
        seps = []
        for i in range(num_trajectories):
            step = fork_steps[i] + off
            if step >= num_steps:
                continue
            pos_true = states[i, step, 0:2]
            pos_pre_fork_vel = states[i, fork_steps[i] - 1, 2:4]
            fork_pt = states[i, fork_steps[i], 0:2]
            d = pos_pre_fork_vel
            t_param = np.dot(pos_true - fork_pt, d) / np.dot(d, d)
            proj = fork_pt + t_param * d
            pos_alt = 2 * proj - pos_true
            seps.append(np.linalg.norm(pos_true - pos_alt))
        if seps:
            mean_sep = np.mean(seps)
            print(f"    +{off:2d} steps post-fork: mean separation = {mean_sep:.3f} km "
                  f"({mean_sep / (2 * MEASUREMENT_STD):.2f}x the 2*MEASUREMENT_STD noise floor)")
    print(f"  -> ambiguity should be genuinely hard to resolve while separation "
          f"< ~2*MEASUREMENT_STD ({2*MEASUREMENT_STD:.2f} km), and should clearly "
          f"resolve well beyond that.")

    return states_tensor, measurements_tensor, branch_taken, fork_steps


# ==========================================
# 4. Save to PyTorch Data File
# ==========================================
def save_dataset(states, measurements, branch_taken, fork_steps, filename="fork_tracking_data.pt"):
    data_dict = {
        'true_trajectories': states,       # Shape: [N, state_dim=4, T]
        'measurements': measurements,      # Shape: [N, obs_dim=2, T]  ([x, y])

        # Metadata stored automatically in self.norm_stats
        'FORK_STEP_RANGE': list(FORK_STEP_RANGE),
        'RAMP_STEPS': RAMP_STEPS,
        'TURN_ANGLE_DEG': TURN_ANGLE_DEG,
        'MEASUREMENT_STD': MEASUREMENT_STD,
        'SPEED': SPEED,
        'DT': DT,
        'branch_taken': branch_taken.tolist(),
        'fork_steps': fork_steps.tolist(),
    }

    torch.save(data_dict, filename)
    print(f"Dataset saved to '{filename}'")
    print(f"  true_trajectories shape: {data_dict['true_trajectories'].shape}")
    print(f"  measurements shape:      {data_dict['measurements'].shape}")
    print(f"  branch split: left={int((branch_taken == 0).sum())}  "
          f"right={int((branch_taken == 1).sum())}")


# ==========================================
# 5. Visualization
# ==========================================
def plot_example_trajectory(states_tensor, measurements_tensor, branch_taken, fork_steps, idx=0, dt=DT):
    state_seq = states_tensor[idx].numpy()   # [4, T]
    meas_seq = measurements_tensor[idx].numpy()  # [2, T]
    fork_step = int(fork_steps[idx])

    num_steps = state_seq.shape[1]
    time_steps = np.arange(num_steps) * dt

    x_pos = state_seq[0, :]
    y_pos = state_seq[1, :]
    true_traj_phys = state_seq.T  # [T, 4]

    alt_pos = get_alternative_branch(true_traj_phys, fork_step)  # [T, 2]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'height_ratios': [2, 1]})

    branch_label = 'left' if branch_taken[idx] == 0 else 'right'
    ax1.plot(x_pos[:fork_step + 1], y_pos[:fork_step + 1], 'b-', linewidth=2, label='Pre-fork (shared)')
    ax1.plot(x_pos[fork_step:], y_pos[fork_step:], 'g-', linewidth=2, label=f'True branch ({branch_label})')
    ax1.plot(alt_pos[fork_step:, 0], alt_pos[fork_step:, 1], 'r--', linewidth=2, alpha=0.7,
              label='Untaken branch (ground truth alt. hypothesis)')
    ax1.scatter(x_pos[fork_step], y_pos[fork_step], c='black', marker='*', s=200,
                label=f'Fork start (t={fork_step})', zorder=6)
    ax1.scatter(x_pos[0], y_pos[0], c='green', marker='o', s=80, label='Start', zorder=5)
    ax1.scatter(meas_seq[0, :], meas_seq[1, :], c='gray', s=6, alpha=0.3, label='Noisy measurements')

    ax1.set_title(f'Fork Tracking — True vs. Untaken Branch (Sample {idx}, fork_step={fork_step})')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8)

    sep = np.linalg.norm(true_traj_phys[:, :2] - alt_pos, axis=-1)
    ax2.plot(time_steps, sep, 'k-', linewidth=2, label='True/alt-branch separation')
    ax2.axhline(2 * MEASUREMENT_STD, color='red', linestyle=':', label='2x measurement std (noise floor)')
    ax2.axvline(fork_step * dt, color='gray', linestyle='--', alpha=0.6, label='Fork start')
    ax2.axvspan(fork_step * dt, (fork_step + RAMP_STEPS) * dt, color='gray', alpha=0.15, label='Ramp window')
    ax2.set_title('Branch Separation Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Separation (km)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('fork_sample_trajectory.png', dpi=200)
    plt.close(fig)


# ==========================================
# 6. Execution Block
# ==========================================
if __name__ == "__main__":
    print("Generating fork/maneuvering-target tracking dataset...")
    states_t, meas_t, branch_t, fork_steps_t = generate_fork_trajectories(
        num_trajectories=5000, num_steps=100, dt=DT)

    save_dataset(states_t, meas_t, branch_t, fork_steps_t, filename="fork_tracking_data.pt")

    print("Plotting sample index 0...")
    plot_example_trajectory(states_t, meas_t, branch_t, fork_steps_t, idx=0, dt=DT)
