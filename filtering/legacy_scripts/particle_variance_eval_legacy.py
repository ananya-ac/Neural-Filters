import argparse
import copy
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import baseline_edh
import baseline_edh_tdoa
from no_control_eval_legacy import (
    load_uncontrolled_data,
    prepare_models,
    parse_labels,
    run_uncontrolled_inference,
    EDH_SENTINEL,
)

DEFAULT_DATA_PATHS = {
    'acoustic': 'test_data/acoustic/no_glitch/acoustic_tracking_data_lo_noise.pt',
    'lorenz': 'lorenz_tracking_data.pt',
    'tdoa': 'tdoa_tracking_data.pt',
}

# Each EDH module supplies its own F/Q_chol/exact_daum_huang_step, matching
# that dataset's actual motion model and measurement geometry -- baseline_edh
# (acoustic) hardcodes the acoustic sensor array via config.NPFFConfig,
# baseline_edh_tdoa hardcodes the two TDOA sensor pairs from tdoa_data_gen.py.
# No EDH baseline exists for 'lorenz'.
EDH_MODULES = {
    'acoustic': baseline_edh,
    'tdoa': baseline_edh_tdoa,
}


def compute_particle_spread(particle_history, t_std, metric, dims):
    """
    particle_history : [T, N, state_dim] normalised particles
    t_std             : [state_dim] training-set std/scale (no shift — spread
                        is scale-dependent only, never shift-dependent)
    metric            : 'std' or 'var'
    dims              : 'pos' (first 2 dims) or 'all' (every state dim)

    Returns spread [T] — the chosen metric averaged over the selected dims,
    in physical units.
    """
    particles_phys = particle_history * t_std.view(1, 1, -1)
    if metric == 'std':
        per_dim = particles_phys.std(dim=1)   # [T, state_dim]
    else:
        per_dim = particles_phys.var(dim=1)   # [T, state_dim]

    if dims == 'pos':
        per_dim = per_dim[:, :2]
    return per_dim.mean(dim=-1)               # [T]


@torch.no_grad()
def run_edh_inference(edh_module, true_phys, meas_phys, num_particles, num_steps):
    """
    Exact Daum-Huang flow baseline, mirroring baseline_edh.py's main loop.
    Operates directly in physical units (no normalisation involved).

    edh_module  : baseline_edh (acoustic) or baseline_edh_tdoa (tdoa) --
                  supplies F/Q_chol/exact_daum_huang_step for the dataset's
                  own motion model and measurement geometry (see EDH_MODULES).
    true_phys : [T, 4] physical (x, y, vx, vy)
    meas_phys : [T, obs_dim] physical measurements (nSensor readings for
                  acoustic; 2 range-differences for tdoa)

    Returns particle_history [T, num_particles, 4] in physical units.
    """
    total_steps = true_phys.shape[0]
    particle_history = torch.zeros((total_steps, num_particles, 4))

    x_start = true_phys[0].unsqueeze(0).repeat(num_particles, 1)
    particles_curr = x_start + torch.randn(num_particles, 4) @ edh_module.Q_chol.T
    particle_history[0] = particles_curr

    for t in range(1, total_steps):
        particles_prior = (particles_curr @ edh_module.F.T
                            + torch.randn(num_particles, 4) @ edh_module.Q_chol.T)
        particles_post = edh_module.exact_daum_huang_step(
            particles_prior, meas_phys[t], num_steps=num_steps)
        particle_history[t] = particles_post
        particles_curr = particles_post

    return particle_history


def denormalize_uncontrolled(dm, args, true_seq_norm, meas_seq_norm):
    """Invert FileTrackingDataModule's normalisation back to physical units."""
    t_mean = dm.t_mean.squeeze().cpu()
    t_std  = dm.t_std.squeeze().cpu()
    m_mean = dm.m_mean.squeeze().cpu()
    m_std  = dm.m_std.squeeze().cpu()

    true_phys = true_seq_norm.cpu() * t_std + t_mean
    meas_denorm = meas_seq_norm.cpu() * m_std + m_mean
    if args.log_transform:
        meas_phys = torch.exp(meas_denorm) - 1e-6
    else:
        meas_phys = meas_denorm
    return true_phys, meas_phys


def plot_spread_over_time(args, eval_idx, traj_indices, histories, true_phys_by_traj,
                           labels, t_std):
    model_colors = ['red', 'blue', 'gold', 'purple', 'orange']
    ylabel = ('Variance' if args.metric == 'var' else 'Std. Dev.') + \
             (' (position)' if args.dims == 'pos' else ' (all states)')

    fig, ax = plt.subplots(figsize=(12, 7))
    prediction_data = {'eval_idx': eval_idx, 'models': {}}

    if args.include_edh:
        prediction_data['edh'] = {}
        for i, traj_idx in enumerate(traj_indices):
            edh_spread = compute_particle_spread(
                histories['edh'][traj_idx], torch.ones(4), args.metric, args.dims)
            time_axis = torch.arange(edh_spread.shape[0])
            prediction_data['edh'][traj_idx] = edh_spread
            ax.plot(
                time_axis.numpy(), edh_spread.numpy(),
                color='black', linestyle='--', linewidth=2, alpha=0.8,
                label='EDH baseline' if i == 0 else None,
            )

    for model_idx, label in enumerate(labels):
        if label not in histories['models'] or not histories['models'][label]:
            continue
        color = model_colors[model_idx % len(model_colors)]
        prediction_data['models'][label] = {}

        # Empirical ensemble spread (std/var across particles) is omitted
        # here -- with num_particles=1 (required by gaussian_nll) there is
        # only one particle, so that spread is identically zero and carries
        # no information. The predicted_std block below (var_head's learned
        # uncertainty estimate) is the meaningful signal for these models.

        # Plot predicted variance if available (gaussian_nll models)
        if 'predicted_std' in histories and label in histories['predicted_std']:
            for i, traj_idx in enumerate(traj_indices):
                if traj_idx not in histories['predicted_std'][label]:
                    continue
                pred_std_norm = histories['predicted_std'][label][traj_idx]
                pred_std_phys = pred_std_norm * t_std  # denormalize

                # Average over dims (pos or all)
                if args.dims == 'pos':
                    pred_spread = pred_std_phys[:, :2].mean(dim=-1)
                else:
                    pred_spread = pred_std_phys.mean(dim=-1)

                time_axis = torch.arange(pred_spread.shape[0])
                prediction_data['models'][label][traj_idx] = pred_spread
                ax.plot(
                    time_axis.numpy(), pred_spread.numpy(),
                    color=color, linestyle='--', linewidth=2, alpha=0.6,
                    label=f'{label} (predicted)' if i == 0 else None,
                )

    ax.set_xlabel('Time step', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(f'Particle Spread over Time ({args.dataset_type})', fontsize=20)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize='large')

    data_path = os.path.join(args.save_dir, f'particle_spread_{eval_idx}.pt')
    plot_path = os.path.join(args.save_dir, f'particle_spread_{eval_idx}.png')
    torch.save(prediction_data, data_path)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved data to '{data_path}'")
    print(f"  Saved plot to '{plot_path}'")


def plot_final_positions(args, eval_idx, traj_indices, histories, true_phys_by_traj,
                          labels, t_mean, t_std):
    traj_colors   = ['red', 'blue', 'gold', 'purple', 'orange']
    model_markers = ['o', '^', 's', 'D', 'v']

    fig, ax = plt.subplots(figsize=(10, 10))
    final_data = {'eval_idx': eval_idx, 'true': {}, 'models': {}, 'edh': {}}

    for traj_pos, traj_idx in enumerate(traj_indices):
        color = traj_colors[traj_pos % len(traj_colors)]

        true_final = true_phys_by_traj[traj_idx][-1, :2]
        final_data['true'][traj_idx] = true_final
        ax.scatter(
            true_final[0], true_final[1],
            color=color, marker='*', s=300, edgecolors='black', zorder=10,
            label='True final position' if traj_pos == 0 else None,
        )

        if args.include_edh:
            edh_final_xy = histories['edh'][traj_idx][-1, :, :2]
            final_data['edh'][traj_idx] = edh_final_xy
            ax.scatter(
                edh_final_xy[:, 0], edh_final_xy[:, 1],
                color='black', marker='+', s=80, alpha=0.7, zorder=5,
                label='EDH baseline' if traj_pos == 0 else None,
            )

        for model_idx, label in enumerate(labels):
            if label not in histories['models'] or traj_idx not in histories['models'][label]:
                continue
            marker = model_markers[model_idx % len(model_markers)]
            final_xy_norm = histories['models'][label][traj_idx][-1, :, :2]
            final_xy_phys = final_xy_norm * t_std[:2] + t_mean[:2]
            final_data['models'].setdefault(label, {})[traj_idx] = final_xy_phys
            ax.scatter(
                final_xy_phys[:, 0], final_xy_phys[:, 1],
                color=color, marker=marker, s=60, alpha=0.6, zorder=4,
                label=label if traj_pos == 0 else None,
            )

    ax.set_xlabel('X Position', fontsize=18)
    ax.set_ylabel('Y Position', fontsize=18)
    ax.set_title(f'Final Particle Positions ({args.dataset_type})', fontsize=20)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', fontsize='large')

    data_path = os.path.join(args.save_dir, f'final_particles_{eval_idx}.pt')
    plot_path = os.path.join(args.save_dir, f'final_particles_{eval_idx}.png')
    torch.save(final_data, data_path)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved data to '{data_path}'")
    print(f"  Saved plot to '{plot_path}'")


def run_variance_eval(args):
    print(f"--- Particle Spread Evaluation ({args.dataset_type}) ---")

    edh_module = None
    if args.include_edh:
        edh_module = EDH_MODULES.get(args.dataset_type)
        if edh_module is None:
            raise ValueError(
                f"--include_edh has no EDH baseline for dataset_type="
                f"{args.dataset_type} -- only {list(EDH_MODULES.keys())} are "
                f"supported (see EDH_MODULES). Drop --include_edh or add a "
                f"baseline_edh_{args.dataset_type}.py module.")

    dm = load_uncontrolled_data(args)
    # prepare_models() (no_control_eval_legacy.py) rejects --include_edh for
    # any dataset_type != 'acoustic' -- correct for that module's own EDH
    # dispatch (only run_acoustic_eval actually calls its run_edh_inference;
    # run_tdoa_eval/run_lorenz_eval never do). This file doesn't use that
    # path at all -- EDH runs entirely separately above via edh_module -- so
    # call prepare_models with include_edh forced off to load just the real
    # checkpoints, bypassing a guard that doesn't apply to this file's usage.
    # labels still uses the real args, so 'EDH' is still appended for
    # plot_spread_over_time/plot_final_positions bookkeeping.
    model_args = copy.copy(args)
    model_args.include_edh = False
    models = prepare_models(model_args)
    labels = parse_labels(args)

    # Match EDH's λ-integration resolution to the model's own, unless the
    # caller explicitly asked for a different --edh_num_steps -- see its
    # docstring. prepare_models() can't do this itself here since it's
    # called with include_edh forced off (see the comment above).
    if args.include_edh and args.edh_num_steps is None:
        steps_seen = sorted({m.integration_steps for m in models if hasattr(m, 'integration_steps')})
        args.edh_num_steps = steps_seen[0]
        possessive = "models'" if len(models) > 1 else "model's"
        print(f"  EDH λ-integration steps: {args.edh_num_steps} "
              f"(matched to the {possessive} own integration_steps)")
        if len(steps_seen) > 1:
            print(f"  WARNING: loaded checkpoints disagree on integration_steps "
                  f"{steps_seen} -- EDH is matched to only {args.edh_num_steps}, "
                  f"not all of them. Pass --integration_steps to force every "
                  f"model to the same resolution for a clean comparison.")

    t_mean = dm.t_mean.squeeze().cpu()
    t_std  = dm.t_std.squeeze().cpu()
    num_val_trajs = dm.val_dataset.measurements.shape[0]

    os.makedirs(args.save_dir, exist_ok=True)

    for eval_idx in range(args.num_eval):
        print(f"\nFigure {eval_idx + 1}/{args.num_eval}")

        if args.traj_indices is not None:
            traj_indices = [t.item() for t in torch.tensor(args.traj_indices)]
        else:
            traj_indices = torch.randperm(num_val_trajs)[:args.num_traj].tolist()

        print(f"  Trajectory indices: {traj_indices}")

        # ── Run inference once per (source, trajectory); reused by both plots.
        histories = {'edh': {}, 'models': {label: {} for label in labels}}
        true_phys_by_traj = {}

        for traj_idx in traj_indices:
            meas_seq_norm = dm.val_dataset.measurements[traj_idx]
            true_seq_norm = dm.val_dataset.true_states[traj_idx]
            true_phys, meas_phys = denormalize_uncontrolled(
                dm, args, true_seq_norm, meas_seq_norm)
            true_phys_by_traj[traj_idx] = true_phys

            if args.include_edh:
                print(f"  [EDH] trajectory {traj_idx}")
                histories['edh'][traj_idx] = run_edh_inference(
                    edh_module, true_phys, meas_phys, args.edh_num_particles, args.edh_num_steps)

            for model, label in zip(models, labels):
                if model is EDH_SENTINEL:
                    continue
                print(f"  [{label}] trajectory {traj_idx}")
                meas_seq = meas_seq_norm.to(args.device)
                true_seq_norm_dev = true_seq_norm.to(args.device)
                _, particle_history, predicted_std = run_uncontrolled_inference(
                    model, meas_seq, true_seq_norm_dev, args.num_particles, args.device
                )
                histories['models'][label][traj_idx] = particle_history
                if predicted_std is not None:
                    if 'predicted_std' not in histories:
                        histories['predicted_std'] = {}
                    if label not in histories['predicted_std']:
                        histories['predicted_std'][label] = {}
                    histories['predicted_std'][label][traj_idx] = predicted_std

        plot_spread_over_time(
            args, eval_idx, traj_indices, histories, true_phys_by_traj, labels, t_std)
        plot_final_positions(
            args, eval_idx, traj_indices, histories, true_phys_by_traj, labels,
            t_mean, t_std)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Track particle ensemble spread (std/var) over a trajectory.')
    parser.add_argument('--dataset_type', type=str, default='acoustic',
                        choices=['acoustic', 'lorenz', 'tdoa'])
    parser.add_argument('--ckpt_paths', nargs='+', required=True,
                        help='Checkpoint path(s) to evaluate.'),
    parser.add_argument('--device', type=str, required=False,
                        choices=['cuda', 'cpu']),
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Optional labels for each checkpoint.')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Dataset path. Defaults to the standard path for the selected dataset type.')
    parser.add_argument('--save_dir', type=str, default='eval_outputs')
    parser.add_argument('--num_particles', type=int, default=100)
    parser.add_argument('--integration_steps', type=int, default=None,
                        help='Override the number of λ-homotopy integration '
                             'steps used at eval time (rebuilds the lam_steps '
                             'schedule from the checkpoint\'s own lam_schedule). '
                             'Defaults to whatever the checkpoint was trained with.')
    parser.add_argument('--override_ssm_residual', type=str, default=None,
                        choices=['true', 'false'],
                        help='Force SSMStylePhysicsPredictor\'s residual mode '
                             'after loading, overriding whatever the checkpoint '
                             'was actually trained with (model.hparams.'
                             'ssm_residual_predict). Only meaningful for '
                             'ssm_style_predict=True checkpoints. Default: use '
                             'each checkpoint\'s own saved value (the correct '
                             'choice for normal evaluation) — this flag exists '
                             'for diagnosing/comparing checkpoints across the '
                             'residual/non-residual convention change, not for '
                             'routine use.')
    parser.add_argument('--num_eval', type=int, default=1,
                        help='Number of figures to generate.')
    parser.add_argument('--num_traj', type=int, default=3,
                        help='Number of trajectories to overlay per figure.')
    parser.add_argument('--traj_indices', nargs='+', type=int, default=None,
                        help='Optional fixed trajectory indices.')
    parser.add_argument('--log_transform', action='store_true')
    parser.add_argument('--equation_scaling', action='store_true',
                        help='Use min-max normalisation (yscale = ymax - ymin, '
                             'Eq. 13 arXiv:2103.15341). Must match the flag '
                             'used when the checkpoint was trained.')
    parser.add_argument('--metric', type=str, default='std', choices=['std', 'var'],
                        help='Plot particle standard deviation or variance.')
    parser.add_argument('--dims', type=str, default='pos', choices=['pos', 'all'],
                        help="'pos' averages over the first two (x,y) state "
                             "dims only; 'all' averages over every state dim.")
    parser.add_argument('--include_edh', action='store_true',
                        help='Overlay the exact Daum-Huang flow baseline. '
                             'Supported for dataset_type in EDH_MODULES '
                             '(currently acoustic, tdoa) -- each with its own '
                             'F/Q/H/R matching that dataset\'s real generative '
                             'model (baseline_edh.py / baseline_edh_tdoa.py).')
    parser.add_argument('--edh_num_steps', type=int, default=None,
                        help='EDH flow integration steps (see baseline_edh.py). '
                             'Defaults to matching the (first) model\'s own '
                             'integration_steps (its checkpoint value, or '
                             '--integration_steps if that\'s set) -- an unequal '
                             'step count silently confounds any spread/RMSE '
                             'difference with a difference in integration '
                             'resolution instead of the thing actually being '
                             'compared. Pass explicitly to deliberately test '
                             'EDH at a different resolution than the model.')
    parser.add_argument('--edh_num_particles', type=int, default=None,
                        help='Particle count for the EDH baseline. Defaults to '
                             '--num_particles.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = DEFAULT_DATA_PATHS[args.dataset_type]

    if args.edh_num_particles is None:
        args.edh_num_particles = args.num_particles

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    run_variance_eval(args)


if __name__ == '__main__':
    main()
