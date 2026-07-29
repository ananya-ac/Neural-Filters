"""
Generates synthetic NCV (Nearly Constant Velocity) tracking data
for acoustic / radar target tracking, compatible with dataset.py.

Observation model: range, bearing, range-rate (spherical coordinates)
  - same as the HFSW radar paper (Maresca et al., IEEE TGRS 2014)

Motion model: NCV in 2D Cartesian coordinates
  state: [x, x_dot, y, y_dot]  (state_dim = 4)
  obs:   [range, bearing, range_rate]  (obs_dim = 3)

Output .pt file keys (all tensors):
  true_trajectories : [N, 4, T]   float32
  measurements      : [N, 3, T]   float32
  x_mean            : [4, 1]      float32   (over training split)
  x_std             : [4, 1]      float32
  z_mean            : [3, 1]      float32
  z_std             : [3, 1]      float32

Usage
-----
  python generate_tracking_data.py                         # default args
  python generate_tracking_data.py --n_traj 5000 \\
      --T 120 --noise_level high --out my_data.pt
"""

import argparse
import math
import os

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters (can be overridden via CLI)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    n_traj       = 3000,    # number of trajectories
    T            = 100,     # time steps per trajectory
    dt           = 33.28,   # sampling interval [s]  (matches WERA CPI)
    sigma_v      = 0.01,    # process noise std [m/s^2]
    sigma_r      = 150.0,   # range measurement noise std [m]
    sigma_b      = 0.026,   # bearing noise std [rad]  (~1.5 deg)
    sigma_rdot   = 0.1,     # range-rate noise std [m/s]
    rho          = 0.1,     # range / range-rate correlation coefficient
    # Initial state distribution
    x_range      = (-80e3, 80e3),   # x position [m]
    y_range      = (10e3, 150e3),   # y position [m]  (keep targets in front)
    vx_range     = (-15.0, 15.0),   # x velocity [m/s]
    vy_range     = (-15.0, 15.0),   # y velocity [m/s]
    train_frac   = 0.8,
    out          = "tracking_data.pt",
    seed         = 42,
    noise_level  = "medium",        # low / medium / high (scales sigma_*)
)

NOISE_SCALE = dict(low=0.5, medium=1.0, high=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_F_Gamma(dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """State transition and process-noise input matrices for NCV (2-D)."""
    F1 = torch.tensor([[1.0, dt], [0.0, 1.0]])
    G1 = torch.tensor([dt**2 / 2.0, dt])
    # block-diagonal for x and y
    F = torch.zeros(4, 4)
    F[0:2, 0:2] = F1
    F[2:4, 2:4] = F1
    Gamma = torch.zeros(4, 2)
    Gamma[0:2, 0] = G1
    Gamma[2:4, 1] = G1
    return F, Gamma


def measurement_function(state: torch.Tensor) -> torch.Tensor:
    """
    Nonlinear measurement h(x): Cartesian state → [range, bearing, range_rate].
    state: [..., 4]  (x, x_dot, y, y_dot)
    returns: [..., 3]
    """
    x     = state[..., 0]
    xdot  = state[..., 1]
    y     = state[..., 2]
    ydot  = state[..., 3]

    r     = torch.sqrt(x**2 + y**2).clamp(min=1.0)   # avoid div-by-zero
    b     = torch.atan2(y, x)
    rdot  = (x * xdot + y * ydot) / r

    return torch.stack([r, b, rdot], dim=-1)


def correlated_meas_noise(
    N: int, T: int,
    sigma_r: float, sigma_b: float, sigma_rdot: float, rho: float,
) -> torch.Tensor:
    """
    Sample [N, T, 3] measurement noise with range / range-rate correlation ρ.
    Covariance structure matches eq. (8) of Maresca et al.
    """
    # Build 3×3 covariance matrix
    R = torch.zeros(3, 3)
    R[0, 0] = sigma_r ** 2
    R[1, 1] = sigma_b ** 2
    R[2, 2] = sigma_rdot ** 2
    R[0, 2] = rho * sigma_r * sigma_rdot
    R[2, 0] = R[0, 2]

    # Cholesky decomposition → sample correlated noise
    L = torch.linalg.cholesky(R)                      # (3, 3)
    z = torch.randn(N, T, 3)                          # i.i.d. standard normal
    noise = (L @ z.unsqueeze(-1)).squeeze(-1)         # (N, T, 3)
    return noise


def simulate_trajectories(
    n_traj: int,
    T: int,
    dt: float,
    sigma_v: float,
    sigma_r: float,
    sigma_b: float,
    sigma_rdot: float,
    rho: float,
    x_range: tuple,
    y_range: tuple,
    vx_range: tuple,
    vy_range: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    true_states  : [N, 4, T]  float32
    measurements : [N, 3, T]  float32
    """
    F, Gamma = make_F_Gamma(dt)      # (4,4), (4,2)
    Q = sigma_v ** 2 * torch.eye(2)  # process noise covariance (2×2)
    L_Q = torch.linalg.cholesky(Q)   # (2,2)

    # ── Initial states ──────────────────────────────────────────────────────
    x0 = torch.zeros(n_traj, 4)
    x0[:, 0] = torch.empty(n_traj).uniform_(*x_range)
    x0[:, 1] = torch.empty(n_traj).uniform_(*vx_range)
    x0[:, 2] = torch.empty(n_traj).uniform_(*y_range)
    x0[:, 3] = torch.empty(n_traj).uniform_(*vy_range)

    # ── Propagate ────────────────────────────────────────────────────────────
    states_list = []
    x = x0.clone()   # (N, 4)

    for t in range(T):
        states_list.append(x.clone())
        # Process noise: v_k ~ N(0, Q)
        v = (L_Q @ torch.randn(n_traj, 2, 1)).squeeze(-1)   # (N, 2)
        x = (F @ x.unsqueeze(-1)).squeeze(-1) + (Gamma @ v.unsqueeze(-1)).squeeze(-1)

    # Stack → (N, T, 4) then transpose → (N, 4, T)
    true_states = torch.stack(states_list, dim=1).float()    # (N, T, 4)
    true_states = true_states.permute(0, 2, 1)               # (N, 4, T)

    # ── Measurements ────────────────────────────────────────────────────────
    # h applied to (N, T, 4)
    noiseless = measurement_function(true_states.permute(0, 2, 1))   # (N, T, 3)
    noise     = correlated_meas_noise(n_traj, T, sigma_r, sigma_b, sigma_rdot, rho)
    meas      = (noiseless + noise).float()                           # (N, T, 3)
    meas      = meas.permute(0, 2, 1)                                 # (N, 3, T)

    return true_states, meas


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(
    tensor: torch.Tensor, split_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean and std over the training split only.
    tensor : [N, F, T]  → mean/std shape [F, 1]  (broadcastable)
    """
    train = tensor[:split_idx]                    # (N_train, F, T)
    mean  = train.mean(dim=(0, 2), keepdim=False) # (F,)
    std   = train.std(dim=(0, 2), keepdim=False)
    std   = torch.where(std == 0, torch.ones_like(std), std)
    return mean.unsqueeze(-1), std.unsqueeze(-1)  # (F, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NCV tracking dataset.")
    p.add_argument("--n_traj",      type=int,   default=DEFAULTS["n_traj"])
    p.add_argument("--T",           type=int,   default=DEFAULTS["T"])
    p.add_argument("--dt",          type=float, default=DEFAULTS["dt"])
    p.add_argument("--sigma_v",     type=float, default=DEFAULTS["sigma_v"])
    p.add_argument("--sigma_r",     type=float, default=DEFAULTS["sigma_r"])
    p.add_argument("--sigma_b",     type=float, default=DEFAULTS["sigma_b"])
    p.add_argument("--sigma_rdot",  type=float, default=DEFAULTS["sigma_rdot"])
    p.add_argument("--rho",         type=float, default=DEFAULTS["rho"])
    p.add_argument("--train_frac",  type=float, default=DEFAULTS["train_frac"])
    p.add_argument("--out",         type=str,   default=DEFAULTS["out"])
    p.add_argument("--seed",        type=int,   default=DEFAULTS["seed"])
    p.add_argument(
        "--noise_level",
        choices=["low", "medium", "high"],
        default=DEFAULTS["noise_level"],
        help="Scales all measurement noise sigmas by a factor.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # Apply noise scaling
    scale = NOISE_SCALE[args.noise_level]
    sigma_r    = args.sigma_r    * scale
    sigma_b    = args.sigma_b    * scale
    sigma_rdot = args.sigma_rdot * scale

    print("=" * 60)
    print("NCV Tracking Data Generator")
    print("=" * 60)
    print(f"  Trajectories : {args.n_traj}")
    print(f"  Time steps   : {args.T}  (dt = {args.dt} s)")
    print(f"  Noise level  : {args.noise_level}  (scale = {scale})")
    print(f"  σ_v          : {args.sigma_v} m/s²")
    print(f"  σ_r          : {sigma_r:.1f} m")
    print(f"  σ_b          : {math.degrees(sigma_b):.2f} deg")
    print(f"  σ_ṙ          : {sigma_rdot:.3f} m/s")
    print(f"  ρ(r, ṙ)      : {args.rho}")
    print(f"  Output file  : {args.out}")
    print("=" * 60)

    print("Simulating trajectories...", flush=True)
    true_traj, meas = simulate_trajectories(
        n_traj    = args.n_traj,
        T         = args.T,
        dt        = args.dt,
        sigma_v   = args.sigma_v,
        sigma_r   = sigma_r,
        sigma_b   = sigma_b,
        sigma_rdot= sigma_rdot,
        rho       = args.rho,
        x_range   = DEFAULTS["x_range"],
        y_range   = DEFAULTS["y_range"],
        vx_range  = DEFAULTS["vx_range"],
        vy_range  = DEFAULTS["vy_range"],
    )

    print(f"  true_trajectories : {tuple(true_traj.shape)}  (N, state_dim=4, T)")
    print(f"  measurements      : {tuple(meas.shape)}  (N, obs_dim=3, T)")

    # ── Compute statistics on training split only ────────────────────────────
    split_idx = int(args.n_traj * args.train_frac)
    x_mean, x_std = compute_stats(true_traj, split_idx)
    z_mean, z_std = compute_stats(meas,      split_idx)

    print(f"\nState statistics (training split, shape {tuple(x_mean.shape)}):")
    labels = ["x [m]", "ẋ [m/s]", "y [m]", "ẏ [m/s]"]
    for i, lbl in enumerate(labels):
        print(f"  {lbl:>10s}  μ = {x_mean[i,0]:12.2f}   σ = {x_std[i,0]:10.2f}")

    print(f"\nMeasurement statistics (training split):")
    obs_labels = ["range [m]", "bearing [rad]", "range-rate [m/s]"]
    for i, lbl in enumerate(obs_labels):
        print(f"  {lbl:>20s}  μ = {z_mean[i,0]:10.4f}   σ = {z_std[i,0]:10.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    payload = {
        "true_trajectories" : true_traj,   # [N, 4, T]
        "measurements"      : meas,        # [N, 3, T]
        "x_mean"            : x_mean,      # [4, 1]
        "x_std"             : x_std,       # [4, 1]
        "z_mean"            : z_mean,      # [3, 1]
        "z_std"             : z_std,       # [3, 1]
        # Metadata (informational, not used by dataset.py)
        "meta": {
            "n_traj"       : args.n_traj,
            "T"            : args.T,
            "dt"           : args.dt,
            "sigma_v"      : args.sigma_v,
            "sigma_r"      : sigma_r,
            "sigma_b"      : sigma_b,
            "sigma_rdot"   : sigma_rdot,
            "rho"          : args.rho,
            "noise_level"  : args.noise_level,
            "state_dim"    : 4,
            "obs_dim"      : 3,
            "state_labels" : ["x", "x_dot", "y", "y_dot"],
            "obs_labels"   : ["range", "bearing", "range_rate"],
            "train_frac"   : args.train_frac,
            "seed"         : args.seed,
        },
    }

    torch.save(payload, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nSaved → {args.out}  ({size_mb:.1f} MB)")

    # ── Quick sanity check ───────────────────────────────────────────────────
    print("\nRunning sanity check (loading file back)...")
    loaded = torch.load(args.out, weights_only=False, map_location="cpu")
    assert loaded["true_trajectories"].shape == (args.n_traj, 4, args.T)
    assert loaded["measurements"].shape      == (args.n_traj, 3, args.T)
    assert loaded["x_mean"].shape            == (4, 1)
    assert loaded["x_std"].shape             == (4, 1)
    assert loaded["z_mean"].shape            == (3, 1)
    assert loaded["z_std"].shape             == (3, 1)
    assert (loaded["x_std"] > 0).all(),  "x_std has zero entries"
    assert (loaded["z_std"] > 0).all(),  "z_std has zero entries"
    print("  All assertions passed. Dataset is compatible with dataset.py ✓")
    print("Done.")


if __name__ == "__main__":
    main()