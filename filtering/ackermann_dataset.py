"""
Ackermann Drive LiDAR Dataset Generator  (vectorized)
=======================================================
Simulates an Ackermann-drive vehicle navigating a 2D map via waypoint following.
Sensor: 4 range measurements to fixed landmarks (LiDAR model, max-range cutoff).
Stores: states [N,T,4], controls [N,T,2], obs [N,T,4], waypoints [N,5,2].

Vectorized over trajectories: all N trajectories run in parallel each step.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    W: float = 50.0
    H: float = 50.0

    LANDMARKS = np.array([
        [ 2.0,  2.0],
        [48.0,  2.0],
        [48.0, 48.0],
        [ 2.0, 48.0],
    ], dtype=np.float32)          # [4, 2]

    L: float         = 2.5
    DELTA_MAX: float = 0.5
    V_MIN: float     = 2.0
    V_MAX: float     = 6.0
    A_MAX: float     = 1.5
    DT: float        = 0.1
    T: int           = 500
    N_WAYPOINTS: int = 5

    K_HEADING: float = 2.0
    K_SPEED: float   = 1.5
    WP_THRESH: float = 3.0

    SIGMA_XY: float = 0.02
    SIGMA_TH: float = 0.005
    SIGMA_V: float  = 0.02

    SIGMA_R: float  = 0.2
    R_MAX: float    = 60.0

    N_TRAIN: int = 5000
    N_VAL: int   = 800
    N_TEST: int  = 400

    SEED: int     = 42
    OUT_PATH: str = 'ackermann_lidar_dataset_long.pt'


cfg = Config()


# ---------------------------------------------------------------------------
# Vectorized batch rollout
# ---------------------------------------------------------------------------

def generate_split_vectorized(n: int, seed_offset: int = 0) -> dict:
    """
    Generate n trajectories in one vectorized numpy rollout.
    All N trajectories are stepped simultaneously — no Python loop over N.

    Returns dict of float32 tensors:
        states    [N, T, 4]   (x, y, theta, v)
        controls  [N, T, 2]   (delta, a)
        obs       [N, T, 4]   LiDAR ranges to 4 landmarks
        waypoints [N, NW, 2]
        v_targets [N, NW]
    """
    rng = np.random.default_rng(cfg.SEED + seed_offset)
    N, T, NW = n, cfg.T, cfg.N_WAYPOINTS
    L = cfg.LANDMARKS                       # [4, 2]

    # Initial states [N, 4]
    x0  = rng.uniform(5.0, cfg.W - 5.0, N).astype(np.float32)
    y0  = rng.uniform(5.0, cfg.H - 5.0, N).astype(np.float32)
    th0 = rng.uniform(-np.pi, np.pi,    N).astype(np.float32)
    v0  = rng.uniform(cfg.V_MIN, cfg.V_MAX, N).astype(np.float32)
    state = np.stack([x0, y0, th0, v0], axis=1)    # [N, 4]

    # Waypoints [N, NW, 2] and speed targets [N, NW]
    waypoints = rng.uniform(
        [5.0, 5.0], [cfg.W - 5.0, cfg.H - 5.0],
        size=(N, NW, 2)
    ).astype(np.float32)
    v_targets = rng.uniform(
        cfg.V_MIN, cfg.V_MAX, size=(N, NW)
    ).astype(np.float32)

    # Pre-generate all process & sensor noise  [T, N, ...]
    noise_xy  = rng.normal(0, cfg.SIGMA_XY, size=(T, N, 2)).astype(np.float32)
    noise_th  = rng.normal(0, cfg.SIGMA_TH, size=(T, N)).astype(np.float32)
    noise_v   = rng.normal(0, cfg.SIGMA_V,  size=(T, N)).astype(np.float32)
    noise_r   = rng.normal(0, cfg.SIGMA_R,  size=(T, N, 4)).astype(np.float32)

    # Storage
    all_states   = np.empty((N, T, 4), dtype=np.float32)
    all_controls = np.empty((N, T, 2), dtype=np.float32)
    all_obs      = np.empty((N, T, 4), dtype=np.float32)

    wp_idx = np.zeros(N, dtype=np.int32)   # active waypoint index per traj
    idx_N  = np.arange(N)

    for t in range(T):
        x, y, th, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        # Current waypoint & target speed for each trajectory
        cur_wp = waypoints[idx_N, wp_idx]       # [N, 2]
        cur_vt = v_targets[idx_N, wp_idx]       # [N]

        # ── Waypoint-following policy ────────────────────────────────────
        bearing     = np.arctan2(cur_wp[:, 1] - y, cur_wp[:, 0] - x)
        heading_err = ((bearing - th + np.pi) % (2 * np.pi)) - np.pi
        delta = np.clip(cfg.K_HEADING * heading_err, -cfg.DELTA_MAX, cfg.DELTA_MAX)
        a     = np.clip(cfg.K_SPEED   * (cur_vt - v), -cfg.A_MAX,   cfg.A_MAX)
        ctrl  = np.stack([delta, a], axis=1).astype(np.float32)    # [N, 2]

        # ── LiDAR observation ────────────────────────────────────────────
        # diffs: [N, 4, 2]
        pos   = np.stack([x, y], axis=1)[:, None, :]               # [N, 1, 2]
        diffs = L[None, :, :] - pos                                 # [N, 4, 2]
        true_r = np.linalg.norm(diffs, axis=2)                     # [N, 4]
        obs_t  = np.minimum(true_r + noise_r[t], cfg.R_MAX).astype(np.float32)

        # ── Record ───────────────────────────────────────────────────────
        all_states[:, t, :]   = state
        all_controls[:, t, :] = ctrl
        all_obs[:, t, :]      = obs_t

        # ── Kinematic step ───────────────────────────────────────────────
        x_new  = np.clip(x  + v * np.cos(th) * cfg.DT + noise_xy[t, :, 0],
                         0.5, cfg.W - 0.5)
        y_new  = np.clip(y  + v * np.sin(th) * cfg.DT + noise_xy[t, :, 1],
                         0.5, cfg.H - 0.5)
        th_new = ((th + (v / cfg.L) * np.tan(delta) * cfg.DT
                   + noise_th[t] + np.pi) % (2 * np.pi)) - np.pi
        v_new  = np.clip(v + a * cfg.DT + noise_v[t], 0.0, cfg.V_MAX + 2.0)

        state = np.stack([x_new, y_new, th_new, v_new], axis=1)

        # ── Advance waypoint if captured ─────────────────────────────────
        dist_to_wp = np.linalg.norm(state[:, :2] - cur_wp, axis=1)
        advance    = (dist_to_wp < cfg.WP_THRESH) & (wp_idx < NW - 1)
        wp_idx[advance] += 1

    return {
        'states'   : torch.from_numpy(all_states),
        'controls' : torch.from_numpy(all_controls),
        'obs'      : torch.from_numpy(all_obs),
        'waypoints': torch.from_numpy(waypoints),
        'v_targets': torch.from_numpy(v_targets),
    }


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------

def compute_normalisation(train_split: dict) -> dict:
    states = train_split['states'].reshape(-1, 4)
    obs    = train_split['obs'].reshape(-1, 4)
    ctrls  = train_split['controls'].reshape(-1, 2)
    return {
        'state_mean'  : states.mean(0),
        'state_std'   : states.std(0).clamp(min=1e-6),
        'obs_mean'    : obs.mean(0),
        'obs_std'     : obs.std(0).clamp(min=1e-6),
        'control_mean': ctrls.mean(0),
        'control_std' : ctrls.std(0).clamp(min=1e-6),
    }


# ---------------------------------------------------------------------------
# Visualisation — one sample trajectory
# ---------------------------------------------------------------------------

def plot_trajectory(split: dict, traj_idx: int = 0,
                    save_path: str = 'sample_trajectory.png'):
    states    = split['states'][traj_idx].numpy()     # [T, 4]
    controls  = split['controls'][traj_idx].numpy()   # [T, 2]
    obs       = split['obs'][traj_idx].numpy()        # [T, 4]
    waypoints = split['waypoints'][traj_idx].numpy()  # [NW, 2]

    xs     = states[:, 0];  ys     = states[:, 1]
    thetas = states[:, 2];  vs     = states[:, 3]
    deltas = np.degrees(controls[:, 0])
    tt     = np.arange(cfg.T) * cfg.DT

    # ── Layout ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 10))
    fig.patch.set_facecolor('#0d0d1a')
    gs = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.38,
                          left=0.06, right=0.97, top=0.92, bottom=0.07)
    ax_map    = fig.add_subplot(gs[:, :2])
    ax_speed  = fig.add_subplot(gs[0, 2])
    ax_steer  = fig.add_subplot(gs[1, 2])
    ax_ranges = fig.add_subplot(gs[2, 2])

    BG = '#16162a';  GRID = '#282840';  TXT = '#dde0f5'
    for ax in [ax_map, ax_speed, ax_steer, ax_ranges]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID, lw=0.5, ls='--', alpha=0.7)

    # ── Map panel ────────────────────────────────────────────────────────
    cmap    = plt.cm.plasma
    speed_n = (vs - vs.min()) / (vs.max() - vs.min() + 1e-6)
    for i in range(len(xs) - 1):
        ax_map.plot(xs[i:i+2], ys[i:i+2],
                    color=cmap(speed_n[i]), lw=1.6, alpha=0.9)

    for i in np.arange(0, cfg.T, 70):
        ax_map.annotate('',
            xy=(xs[i] + 2.0*np.cos(thetas[i]),
                ys[i] + 2.0*np.sin(thetas[i])),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle='->', color='#aaaaee', lw=1.1))

    ax_map.scatter(cfg.LANDMARKS[:, 0], cfg.LANDMARKS[:, 1],
                   s=180, marker='^', color='#44ff99', zorder=7,
                   edgecolors='white', linewidths=0.7, label='Landmarks')
    for j, lm in enumerate(cfg.LANDMARKS):
        ax_map.text(lm[0]+0.9, lm[1]+0.9, f'L{j+1}',
                    color='#44ff99', fontsize=8, fontweight='bold')

    wp_path = np.vstack([[xs[0], ys[0]], waypoints])
    ax_map.plot(wp_path[:, 0], wp_path[:, 1],
                '--', color='#ffee55', lw=0.9, alpha=0.4, zorder=3)
    ax_map.scatter(waypoints[:, 0], waypoints[:, 1],
                   s=150, marker='*', color='#ffee55', zorder=6,
                   edgecolors='white', linewidths=0.6, label='Waypoints')
    for j, wp in enumerate(waypoints):
        ax_map.text(wp[0]+0.9, wp[1]+0.9, f'WP{j+1}',
                    color='#ffee55', fontsize=7.5)

    ax_map.scatter(xs[0],  ys[0],  s=130, color='#00ffdd',
                   marker='o', zorder=8, label='Start')
    ax_map.scatter(xs[-1], ys[-1], s=130, color='#ff4466',
                   marker='X', zorder=8, label='End')

    ax_map.add_patch(mpatches.Rectangle(
        (0, 0), cfg.W, cfg.H, lw=1.5,
        edgecolor='#5577aa', facecolor='none'))

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vs.min(), vs.max()))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_map, fraction=0.024, pad=0.02)
    cb.set_label('Speed [m/s]', color=TXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TXT, labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TXT)

    ax_map.set_xlim(-1, cfg.W + 1);  ax_map.set_ylim(-1, cfg.H + 1)
    ax_map.set_xlabel('x [m]', color=TXT, fontsize=9)
    ax_map.set_ylabel('y [m]', color=TXT, fontsize=9)
    ax_map.set_title('Trajectory (coloured by speed)',
                     color=TXT, fontsize=11, pad=8)
    ax_map.set_aspect('equal')
    ax_map.legend(loc='upper center', ncol=4, fontsize=8,
                  facecolor='#0d0d1a', edgecolor=GRID, labelcolor=TXT)

    # ── Speed ────────────────────────────────────────────────────────────
    ax_speed.plot(tt, vs, color='#66aaff', lw=1.2)
    ax_speed.set_ylabel('v [m/s]', color=TXT, fontsize=8)
    ax_speed.set_title('Speed', color=TXT, fontsize=9)
    ax_speed.tick_params(labelbottom=False)

    # ── Steering ─────────────────────────────────────────────────────────
    ax_steer.plot(tt, deltas, color='#ff9966', lw=1.2)
    ax_steer.axhline(0, color=GRID, lw=0.8)
    for lim in [np.degrees(cfg.DELTA_MAX), -np.degrees(cfg.DELTA_MAX)]:
        ax_steer.axhline(lim, color='#ff4444', lw=0.8, ls=':', alpha=0.7)
    ax_steer.set_ylabel('δ [deg]', color=TXT, fontsize=8)
    ax_steer.set_title('Steering Angle', color=TXT, fontsize=9)
    ax_steer.tick_params(labelbottom=False)

    # ── LiDAR ranges ─────────────────────────────────────────────────────
    pal = ['#ff4488', '#44ff88', '#4488ff', '#ffcc44']
    for i, c in enumerate(pal):
        ax_ranges.plot(tt, obs[:, i], color=c, lw=0.9,
                       alpha=0.85, label=f'L{i+1}')
    ax_ranges.axhline(cfg.R_MAX, color='white', lw=0.9, ls=':',
                      alpha=0.5, label=f'R_max={cfg.R_MAX:.0f}m')
    ax_ranges.set_ylabel('Range [m]', color=TXT, fontsize=8)
    ax_ranges.set_xlabel('Time [s]', color=TXT, fontsize=8)
    ax_ranges.set_title('LiDAR Ranges to Landmarks', color=TXT, fontsize=9)
    ax_ranges.legend(loc='upper right', fontsize=7, ncol=2,
                     facecolor='#0d0d1a', edgecolor=GRID, labelcolor=TXT)

    fig.suptitle('Ackermann LiDAR Dataset — Sample Trajectory',
                 color=TXT, fontsize=13, fontweight='bold', y=0.97)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("  Ackermann LiDAR Dataset Generator  (vectorized)")
    print("=" * 60)

    splits_cfg = [
        ('train', cfg.N_TRAIN, 0),
        ('val',   cfg.N_VAL,   1000),
        ('test',  cfg.N_TEST,  2000),
    ]

    data = {}
    for name, n, offset in splits_cfg:
        t0 = time.time()
        print(f"\n[{name}]  {n} trajectories × {cfg.T} steps ...")
        data[name] = generate_split_vectorized(n, seed_offset=offset)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        for k, v in data[name].items():
            if hasattr(v, 'shape'):
                print(f"    {k:12s}: {tuple(v.shape)}  {v.dtype}")

    print("\nComputing normalisation stats ...")
    norm_stats = compute_normalisation(data['train'])
    for k, v in norm_stats.items():
        print(f"  {k:16s}: {v.numpy().round(4)}")

    dataset = {
        'train'     : data['train'],
        'val'       : data['val'],
        'test'      : data['test'],
        'norm_stats': norm_stats,
        'config': {
            'W'          : cfg.W,
            'H'          : cfg.H,
            'landmarks'  : torch.tensor(cfg.LANDMARKS),
            'L'          : cfg.L,
            'delta_max'  : cfg.DELTA_MAX,
            'dt'         : cfg.DT,
            'T'          : cfg.T,
            'n_waypoints': cfg.N_WAYPOINTS,
            'sigma_r'    : cfg.SIGMA_R,
            'r_max'      : cfg.R_MAX,
            'state_dim'  : 4,
            'control_dim': 2,
            'obs_dim'    : 4,
        }
    }

    out_path = Path('.') / cfg.OUT_PATH
    torch.save(dataset, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDataset saved → {out_path}  ({size_mb:.1f} MB)")

    print("\nPlotting sample trajectory (train[0]) ...")
    plot_path = '.'
    plot_trajectory(data['train'], traj_idx=0, save_path=plot_path)

    print("\n✓ All done.")