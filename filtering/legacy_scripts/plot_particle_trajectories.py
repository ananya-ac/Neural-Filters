"""
Plots each particle's own (x, y) trajectory -- not just the mixture-weighted
point estimate -- against ground truth, for a handful of validation
trajectories. Purpose-built for num_particles>1 gaussian_nll checkpoints:
mixture_weight_max alone (see model_legacy.py's _unroll logging) can't tell
you whether two particles pinned at ~50/50 weight are spatially near-
identical or genuinely diverging -- this shows that directly.

Loads the .pt dataset file and replicates dataset.py's FileTrackingDataModule
._setup_uncontrolled normalisation exactly (80/20 split, train-only z-score
stats) rather than instantiating the full DataModule, and rolls the model
forward manually (mirroring _unroll's loop in model_legacy.py) so every
particle's own trajectory is kept instead of collapsed into the mixture-
weighted mean predictions_list normally returns.
"""
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from model_legacy import ModularNeuralODEFilter


def load_val_split(data_path):
    """
    Replicates dataset.py's FileTrackingDataModule._setup_uncontrolled
    exactly: 80/20 split (no shuffling), z-score stats computed on the
    train split only, applied to both splits.

    Returns val_true_norm, val_meas_norm [N_val, T, dim] (time-major, what
    the model expects), plus t_mean/t_std [state_dim] for denormalising
    plots back to physical units.
    """
    loaded = torch.load(data_path, weights_only=False, map_location='cpu')
    true_traj = loaded['true_trajectories']   # [N, state_dim, T]
    raw_meas  = loaded['measurements']         # [N, obs_dim,   T]
    N = true_traj.shape[0]
    split_idx = int(N * 0.8)

    train_states = true_traj[:split_idx]
    t_mean = train_states.mean(dim=(0, 2), keepdim=True)
    t_std  = train_states.std(dim=(0, 2), keepdim=True)
    t_std  = torch.where(t_std == 0, torch.ones_like(t_std), t_std)

    train_meas = raw_meas[:split_idx]
    m_mean = train_meas.mean(dim=(0, 2), keepdim=True)
    m_std  = train_meas.std(dim=(0, 2), keepdim=True)
    m_std  = torch.where(m_std == 0, torch.ones_like(m_std), m_std)

    true_norm = (true_traj - t_mean) / t_std
    meas_norm = (raw_meas - m_mean) / m_std

    val_true = true_norm[split_idx:].permute(0, 2, 1)   # [N_val, T, state_dim]
    val_meas = meas_norm[split_idx:].permute(0, 2, 1)   # [N_val, T, obs_dim]
    return val_true, val_meas, t_mean.view(-1), t_std.view(-1)


@torch.no_grad()
def rollout_particles(model, true_seq_norm, meas_seq_norm):
    """
    true_seq_norm, meas_seq_norm : [T, state_dim] / [T, obs_dim], normalised,
    single trajectory (no batch dim).

    Mirrors _unroll's loop (model_legacy.py) exactly, except every particle's
    own particles_pred is kept at each step instead of only the mixture-
    weighted mean_estimate.

    Returns particle_history [T, N, state_dim], normalised, t=0 included
    (t=0 is the true state broadcast/perturbed to all N particles, matching
    _unroll's init).
    """
    device = next(model.parameters()).device
    true_seq_norm = true_seq_norm.to(device)
    meas_seq_norm = meas_seq_norm.to(device)
    T = true_seq_norm.shape[0]
    N = model.num_particles

    particles_curr = true_seq_norm[0].unsqueeze(0).unsqueeze(1).expand(1, N, -1).clone()
    if N > 1:
        particles_curr = particles_curr + (
            torch.randn_like(particles_curr) * model.particle_init_diversity_std)

    state_context, h_state = model._gru_init(
        true_seq_norm[0].unsqueeze(0), particles_curr, batch_size=1)

    history = torch.zeros(T, N, model.state_dim)
    history[0] = particles_curr.squeeze(0).cpu()

    is_gaussian_nll = model.loss_type == 'gaussian_nll'
    prev_log_var = uncertainty_h = None
    if is_gaussian_nll:
        prev_log_var = torch.full((1, N, model.state_dim),
                                  math.log(model.min_variance), device=device)
        uncertainty_h = torch.zeros(
            1, N, model.uncertainty_gru_dim * (2 if model.uncertainty_rnn_lstm else 1),
            device=device)

    for t in range(1, T):
        current_meas = meas_seq_norm[t].unsqueeze(0)
        uncertainty_signal = (
            torch.cat([prev_log_var, model._uncertainty_h_readout(uncertainty_h)], dim=-1)
            if model.uncertainty_conditioning else None)

        particles_pred, particles_prior = model(
            particles_prev=particles_curr, state_context=state_context,
            current_meas=current_meas, uncertainty_signal=uncertainty_signal)

        if is_gaussian_nll:
            _, prev_log_var, uncertainty_h, mixture_weights, _ = model._gaussian_nll_step(
                particles_pred, state_context, current_meas,
                true_seq_norm[t].unsqueeze(0), prev_log_var, uncertainty_h)
            mean_estimate = (mixture_weights.unsqueeze(-1) * particles_pred).sum(dim=1)
        else:
            mean_estimate = particles_pred.mean(dim=1)

        history[t] = particles_pred.squeeze(0).cpu()

        particle_value = particles_pred if model.per_particle_context else mean_estimate
        state_context, h_state = model._gru_step(
            particle_value, particles_pred, h_state, batch_size=1, use_detach=False)
        particles_curr = particles_pred

    return history


def plot_trajectories(save_path, traj_indices, true_hist, particle_hist, t_mean, t_std):
    """
    true_hist     : dict[traj_idx] -> [T, state_dim] normalised true states
    particle_hist : dict[traj_idx] -> [T, N, state_dim] normalised particles
    """
    n = len(traj_indices)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    particle_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    for i, traj_idx in enumerate(traj_indices):
        ax = axes[i]
        true_phys = true_hist[traj_idx] * t_std + t_mean          # [T, state_dim]
        parts_phys = particle_hist[traj_idx] * t_std + t_mean     # [T, N, state_dim]
        N = parts_phys.shape[1]

        ax.plot(true_phys[:, 0], true_phys[:, 1], color='black', linewidth=2.5,
               label='True', zorder=5)
        ax.scatter(true_phys[0, 0], true_phys[0, 1], color='black', marker='o',
                  s=100, zorder=6, label='Start')
        ax.scatter(true_phys[-1, 0], true_phys[-1, 1], color='black', marker='*',
                  s=200, zorder=6, label='End')

        for p in range(N):
            color = particle_colors[p % len(particle_colors)]
            ax.plot(parts_phys[:, p, 0], parts_phys[:, p, 1], color=color,
                   linewidth=1.5, alpha=0.7, label=f'Particle {p}', zorder=3)
            ax.scatter(parts_phys[-1, p, 0], parts_phys[-1, p, 1], color=color,
                      marker='*', s=120, zorder=4)

        ax.set_xlabel('X position', fontsize=13)
        ax.set_ylabel('Y position', fontsize=13)
        ax.set_title(f'Trajectory {traj_idx}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='best', fontsize=9)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Per-particle trajectories vs. ground truth', fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to '{save_path}'")


def main():
    parser = argparse.ArgumentParser(
        description='Plot each particle\'s own trajectory (not just the '
                    'mixture-weighted mean) against ground truth.')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='tdoa_tracking_data.pt')
    parser.add_argument('--num_traj', type=int, default=4)
    parser.add_argument('--traj_indices', nargs='+', type=int, default=None,
                        help='Optional fixed validation-split trajectory indices '
                             '(0-indexed within the val split, not the full '
                             'dataset). Overrides --num_traj if given.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Only used for random trajectory selection when '
                             '--traj_indices is not given.')
    parser.add_argument('--save_dir', type=str, default='eval_outputs_particle_trajectories')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ModularNeuralODEFilter.load_from_checkpoint(
        args.ckpt_path, map_location=args.device, strict=False)
    model.eval().to(args.device)
    print(f"Loaded checkpoint: {args.ckpt_path}  (num_particles={model.num_particles})")

    val_true, val_meas, t_mean, t_std = load_val_split(args.data_path)
    num_val = val_true.shape[0]

    if args.traj_indices is not None:
        traj_indices = args.traj_indices
    else:
        g = torch.Generator().manual_seed(args.seed)
        traj_indices = torch.randperm(num_val, generator=g)[:args.num_traj].tolist()
    print(f"Validation-split trajectory indices: {traj_indices}")

    true_hist = {}
    particle_hist = {}
    for idx in traj_indices:
        print(f"  Rolling out trajectory {idx} ...")
        history = rollout_particles(model, val_true[idx], val_meas[idx])
        particle_hist[idx] = history
        true_hist[idx] = val_true[idx]

    os.makedirs(args.save_dir, exist_ok=True)
    plot_path = os.path.join(args.save_dir, 'particle_trajectories.png')
    data_path = os.path.join(args.save_dir, 'particle_trajectories.pt')
    torch.save({'traj_indices': traj_indices, 'true': true_hist,
               'particles': particle_hist, 't_mean': t_mean, 't_std': t_std}, data_path)
    plot_trajectories(plot_path, traj_indices, true_hist, particle_hist, t_mean, t_std)
    print(f"Saved data to '{data_path}'")


if __name__ == '__main__':
    main()
