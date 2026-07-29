"""
Eval script for the Ackermann (controlled) dataset -- model_legacy.
ModularNeuralODEFilter checkpoints trained via `train_legacy.py --use_control`.

This is the controlled-mode counterpart to no_control_eval_legacy.py (which
covers acoustic/lorenz/tdoa). Two things are structurally different here:

  1. FileTrackingDataModule's controlled path (_setup_controlled) applies NO
     normalisation -- states/obs/controls are used at their raw physical
     scale. There is no t_mean/t_std denormalisation step anywhere below.
  2. The model's forward pass takes an extra `control` input at every step
     (ctrl_seq[t-1] driving the transition into state t, matching _unroll's
     and on_validation_epoch_end's own convention).

State layout (ackermann_dataset.py): state = (x, y, theta, v),
obs = 4 LiDAR ranges to fixed landmarks, control = (delta, a).
"""
import argparse
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from dataset import FileTrackingDataModule
from model_legacy import ModularNeuralODEFilter
from ackermann_dataset import Config as AckermannConfig

DEFAULT_DATA_PATH = 'ackermann_lidar_dataset_long.pt'


@torch.no_grad()
def run_controlled_inference(model, obs_seq, true_seq, ctrl_seq, device):
    """
    Full-trajectory rollout. state_context/h_state_rnn come from the model's
    own _gru_init/_gru_step (not a hand-rolled zero-init) -- see the
    analogous bug fix in no_control_eval_legacy.py's run_uncontrolled_
    inference, where skipping that t=0 GRU forward pass fed garbage context
    into the first prediction and inflated RMSE ~15-20x. _gru_init's h0_scale
    is 0 in eval mode (model.eval() -> self.training=False), so this is
    still fully deterministic.
    """
    state_dim   = model.state_dim
    total_steps = true_seq.shape[0]
    N           = model.num_particles
    is_gaussian_nll = model.loss_type == 'gaussian_nll'

    predictions   = torch.zeros((total_steps, state_dim))
    predicted_std = torch.zeros((total_steps, state_dim)) if is_gaussian_nll else None

    x_start = true_seq[0].unsqueeze(0).unsqueeze(1).expand(-1, N, -1).float()
    x_curr  = x_start
    predictions[0] = true_seq[0].cpu()

    state_context, h_state_rnn = model._gru_init(
        true_seq[0].unsqueeze(0), x_start, batch_size=1)

    if is_gaussian_nll:
        prev_log_var  = torch.full((1, N, state_dim), math.log(model.min_variance), device=device)
        uncertainty_h = torch.zeros(1, N, model.uncertainty_gru_dim, device=device)
        predicted_std[0] = torch.sqrt(torch.exp(prev_log_var[0, 0])).cpu()

    for t in range(1, total_steps):
        current_meas    = obs_seq[t].unsqueeze(0).float()
        current_control = ctrl_seq[t - 1].unsqueeze(0).float()

        particles_pred, _ = model(
            particles_prev = x_curr,
            state_context  = state_context,
            current_meas   = current_meas,
            control        = current_control,
        )

        mean_est = particles_pred.mean(dim=1)
        predictions[t] = mean_est.squeeze(0).cpu()

        if is_gaussian_nll:
            z_sensor   = current_meas[..., :model.sensor_obs_dim]
            z_expanded = z_sensor.unsqueeze(1).expand(-1, N, -1)
            innovation = z_expanded - model._meas_predict(particles_pred, state_context)

            raw_log_var = model.var_head(
                torch.cat([innovation, prev_log_var, uncertainty_h], dim=-1))
            var = torch.nn.functional.softplus(raw_log_var) + model.min_variance
            predicted_std[t] = torch.sqrt(var[0, 0]).cpu()

            prev_log_var  = torch.log(var)
            uncertainty_h = model.uncertainty_rnn(
                innovation.reshape(N, model.sensor_obs_dim),
                uncertainty_h.reshape(N, model.uncertainty_gru_dim),
            ).reshape(1, N, model.uncertainty_gru_dim)

        particle_value = particles_pred if model.per_particle_context else mean_est
        state_context, h_state_rnn = model._gru_step(
            particle_value, particles_pred, h_state_rnn, batch_size=1)
        x_curr = particles_pred

    return predictions, predicted_std


def compute_rmses(predictions, true_states):
    """Position (x,y) RMSE and full-state RMSE, matching on_validation_epoch_end's
    val_rmse_position / val_rmse_rollout convention (raw diff, no theta unwrap)."""
    pos_error   = predictions[1:, :2] - true_states[1:, :2]
    full_error  = predictions[1:]     - true_states[1:]
    pos_rmse    = torch.sqrt((pos_error ** 2).mean()).item()
    full_rmse   = torch.sqrt((full_error ** 2).mean()).item()
    return pos_rmse, full_rmse


def load_data(args):
    dm = FileTrackingDataModule(
        data_path    = args.data_path,
        seq_len      = 10,
        batch_size   = 256,
        dataset_type = 'ackermann',
        use_control  = True,
    )
    dm.setup()
    return dm


def override_integration_steps(model, integration_steps):
    """See no_control_eval_legacy.py -- lam_steps is a buffer baked once at
    construction time and must be rebuilt to change integration_steps post-load."""
    N = integration_steps
    lam_schedule = model.hparams.lam_schedule
    if lam_schedule == 'uniform':
        lam_steps = [1.0 / N] * N
    else:
        b  = 2.0
        s0 = (b - 1) / (b**N - 1)
        lam_steps = [s0 * (b**k) for k in range(N)]
    model.lam_steps = torch.tensor(
        lam_steps, dtype=torch.float32, device=model.lam_steps.device)
    model.integration_steps = N


def override_ssm_residual(model, residual: bool):
    if hasattr(model, 'physics_predictor') and hasattr(model.physics_predictor, 'residual'):
        model.physics_predictor.residual = residual


def prepare_models(args):
    models = []
    for ckpt in args.ckpt_paths:
        model = ModularNeuralODEFilter.load_from_checkpoint(
            ckpt, map_location=args.device, strict=False)
        if args.num_particles is not None and hasattr(model, 'num_particles'):
            model.num_particles = args.num_particles
        if args.integration_steps is not None:
            override_integration_steps(model, args.integration_steps)
        if args.override_ssm_residual is not None:
            override_ssm_residual(model, args.override_ssm_residual == 'true')
        model.eval().to(args.device)
        models.append(model)
        print(f"  Loaded: {ckpt}")
    return models


def parse_labels(args):
    if args.labels is None:
        return [f"Model {i + 1}" for i in range(len(args.ckpt_paths))]
    if len(args.labels) != len(args.ckpt_paths):
        raise ValueError("--labels must match --ckpt_paths length")
    return list(args.labels)


def run_ackermann_eval(args):
    print("--- Ackermann (controlled) Evaluation ---")
    dm     = load_data(args)
    models = prepare_models(args)
    labels = parse_labels(args)

    num_val_trajs = dm.val_dataset.states.shape[0]
    landmarks = AckermannConfig.LANDMARKS

    os.makedirs(args.save_dir, exist_ok=True)
    model_colors = ['red', 'orange', 'purple', 'magenta', 'cyan']

    for eval_idx in range(args.num_eval):
        print(f"\nFigure {eval_idx + 1}/{args.num_eval}")

        if args.traj_indices is not None:
            traj_idx = args.traj_indices[eval_idx % len(args.traj_indices)]
        else:
            traj_idx = torch.randint(0, num_val_trajs, (1,)).item()
        print(f"  Trajectory index: {traj_idx}")

        obs_seq   = dm.val_dataset.obs[traj_idx].to(args.device)
        true_seq  = dm.val_dataset.states[traj_idx].to(args.device)
        ctrl_seq  = dm.val_dataset.controls[traj_idx].to(args.device)
        true_cpu  = true_seq.cpu()
        tt        = torch.arange(true_cpu.shape[0]) * AckermannConfig.DT

        fig, (ax_map, ax_theta, ax_v) = plt.subplots(
            1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1, 1]})

        ax_map.plot(true_cpu[:, 0], true_cpu[:, 1], color='blue', linewidth=2,
                    alpha=0.6, zorder=3, label='True Trajectory')
        ax_map.scatter(true_cpu[0, 0], true_cpu[0, 1], c='green', marker='o',
                        s=100, edgecolors='black', zorder=10, label='Start')
        ax_map.scatter(true_cpu[-1, 0], true_cpu[-1, 1], c='black', marker='X',
                        s=100, edgecolors='black', zorder=10, label='End')
        ax_map.scatter(landmarks[:, 0], landmarks[:, 1], s=150, marker='^',
                        color='seagreen', zorder=6, edgecolors='black', label='Landmarks')

        ax_theta.plot(tt, true_cpu[:, 2], color='blue', alpha=0.6, linewidth=2)
        ax_v.plot(tt, true_cpu[:, 3], color='blue', alpha=0.6, linewidth=2, label='True')

        prediction_data = {'trajectory_index': traj_idx, 'models': {}}

        for model_idx, (model, label) in enumerate(zip(models, labels)):
            color = model_colors[model_idx % len(model_colors)]
            print(f"  -> '{label}' on trajectory {traj_idx}")

            predictions, _ = run_controlled_inference(
                model, obs_seq, true_seq, ctrl_seq, args.device)
            pos_rmse, full_rmse = compute_rmses(predictions, true_cpu)
            print(f"     position RMSE: {pos_rmse:.4f} m   full-state RMSE: {full_rmse:.4f}")

            prediction_data['models'][label] = {
                'mean_prediction': predictions,
                'rmse_position': pos_rmse,
                'rmse_full': full_rmse,
            }

            ax_map.plot(predictions[:, 0], predictions[:, 1], color=color,
                        linestyle='--', linewidth=2, zorder=5, label=label)
            ax_theta.plot(tt, predictions[:, 2], color=color, linestyle='--', linewidth=1.5)
            ax_v.plot(tt, predictions[:, 3], color=color, linestyle='--', linewidth=1.5, label=label)

        ax_map.set_xlim(-1, AckermannConfig.W + 1)
        ax_map.set_ylim(-1, AckermannConfig.H + 1)
        ax_map.set_xlabel('x [m]')
        ax_map.set_ylabel('y [m]')
        ax_map.set_title(f'Trajectory {traj_idx}')
        ax_map.set_aspect('equal')
        ax_map.legend(loc='best', fontsize=8)
        ax_map.grid(True, linestyle='--', alpha=0.5)

        ax_theta.set_xlabel('t [s]')
        ax_theta.set_ylabel(r'$\theta$ [rad]')
        ax_theta.set_title('Heading')
        ax_theta.grid(True, linestyle='--', alpha=0.5)

        ax_v.set_xlabel('t [s]')
        ax_v.set_ylabel('v [m/s]')
        ax_v.set_title('Speed')
        ax_v.legend(loc='best', fontsize=8)
        ax_v.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()

        data_path = os.path.join(args.save_dir, f'ackermann_eval_{eval_idx}.pt')
        plot_path = os.path.join(args.save_dir, f'ackermann_eval_{eval_idx}.png')
        torch.save(prediction_data, data_path)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    all_pos_rmses = {label: [] for label in labels}
    all_full_rmses = {label: [] for label in labels}
    for eval_idx in range(args.num_eval):
        d = torch.load(os.path.join(args.save_dir, f'ackermann_eval_{eval_idx}.pt'),
                        map_location='cpu', weights_only=False)
        for label, info in d['models'].items():
            all_pos_rmses[label].append(info['rmse_position'])
            all_full_rmses[label].append(info['rmse_full'])

    print("\n" + "=" * 50)
    print("Summary (mean over evaluated trajectories)")
    print("=" * 50)
    for label in labels:
        pr = all_pos_rmses[label]
        fr = all_full_rmses[label]
        print(f"  {label}: position RMSE = {sum(pr)/len(pr):.4f} m   "
              f"full-state RMSE = {sum(fr)/len(fr):.4f}")


def build_parser():
    parser = argparse.ArgumentParser(
        description='Eval script for the Ackermann (controlled) dataset.')
    parser.add_argument('--device', type=str, required=False, choices=['cuda', 'cpu'])
    parser.add_argument('--ckpt_paths', nargs='+', required=True,
                        help='Checkpoint path(s) to evaluate.')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Optional labels for each checkpoint.')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--save_dir', type=str, default='eval_outputs_ackermann')
    parser.add_argument('--num_particles', type=int, default=1,
                        help='gaussian_nll checkpoints require exactly 1 (matches training).')
    parser.add_argument('--integration_steps', type=int, default=None,
                        help='Override the number of lambda-homotopy integration steps '
                             'used at eval time. Defaults to the checkpoint\'s own value.')
    parser.add_argument('--override_ssm_residual', type=str, default=None,
                        choices=['true', 'false'],
                        help='Force SSMStylePhysicsPredictor\'s residual mode after '
                             'loading. Default: use the checkpoint\'s own saved value.')
    parser.add_argument('--num_eval', type=int, default=1,
                        help='Number of figures (trajectories) to generate.')
    parser.add_argument('--traj_indices', nargs='+', type=int, default=None,
                        help='Optional fixed validation-trajectory indices.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    run_ackermann_eval(args)


if __name__ == '__main__':
    main()
