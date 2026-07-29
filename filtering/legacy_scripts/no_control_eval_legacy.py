import argparse
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from dataset import FileTrackingDataModule
from model_legacy import ModularNeuralODEFilter
import baseline_edh as edh
import baseline_edh_tdoa
import baseline_edh_lorenz

# Sentinel for EDH baseline (no model needed)
EDH_SENTINEL = object()

# Each module supplies predict()/Q_chol/exact_daum_huang_step/h_func for its
# own dataset's real generative model (dynamics + measurement geometry) --
# baseline_edh (acoustic sensor array via config.NPFFConfig),
# baseline_edh_tdoa (both TDOA range-difference pairs),
# baseline_edh_lorenz (nonlinear Lorenz63 ODE + range/azimuth/elevation).
EDH_MODULES = {
    'acoustic': edh,
    'tdoa': baseline_edh_tdoa,
    'lorenz': baseline_edh_lorenz,
}


DEFAULT_DATA_PATHS = {
    'acoustic': 'var_len_train/holdout_test/acoustic_test_T100_N2500.pt',
    'lorenz': 'lorenz_tracking_data.pt',
    'tdoa': 'tdoa_tracking_data.pt',
}


def load_model_class(dataset_type: str):
    if dataset_type not in ('acoustic', 'lorenz', 'tdoa'):
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")
    return ModularNeuralODEFilter


def pinv_init_context(model, x0, device):
    """
    Analytically solve for the state_context that makes context_readout
    reconstruct EXACTLY to x0, instead of the approximate reconstruction a
    cold _gru_init pass produces (measured ~2.4km off on a real trajectory
    even though _gru_init's input was the exact true state).

    Only valid for ssm_style_predict=True with ssm_wide_context_input=False
    -- that's the only configuration with a separate context_readout:
    Linear(state_gru_dim, state_dim) module. physics_predictor consumes
    context_readout(state_context) as its input (not state_context
    directly), so exactly inverting context_readout fixes that input
    exactly, without bypassing physics_predictor itself -- unlike
    hand-computing a coast-from-known-state prediction, this keeps the
    model's own learned prediction mechanism in the loop.

    Uses only context_readout's existing trained weights -- no new
    parameters, no retraining. state_gru_dim (typically 32) > state_dim
    (typically 4) makes context_readout's weight matrix "wide" (more
    columns than rows), so the system Wh+b=x0 is underdetermined; the
    pseudo-inverse gives the minimum-norm solution. That solution is
    guaranteed correct for physics_predictor's t=1 input (which only sees
    context_readout(state_context), not state_context itself), but may be
    an atypical vector in state_gru_dim-space compared to what the GRU
    normally produces -- untested whether that affects _gru_step's
    behaviour once it's handed off as h_state for t=2.

    x0 : [state_dim] true (or best available estimate of the) initial state.
    Returns (context, h_state) matching _gru_init's return-shape contract.
    """
    if not (model.ssm_style_predict and not model.ssm_wide_context_input):
        raise ValueError(
            "pinv_init_context requires ssm_style_predict=True and "
            "ssm_wide_context_input=False -- only that configuration has a "
            "separate context_readout module to invert.")
    W = model.context_readout.weight   # [state_dim, state_gru_dim]
    b = model.context_readout.bias     # [state_dim]
    W_pinv = torch.linalg.pinv(W)      # [state_gru_dim, state_dim]
    h_star = W_pinv @ (x0.to(W.device) - b)   # [state_gru_dim]

    N = model.num_particles
    context = h_star.unsqueeze(0).unsqueeze(0).expand(1, N, -1).contiguous().to(device)
    h_state = h_star.unsqueeze(0).unsqueeze(0).expand(
        model.gru_num_layers, N, -1).contiguous().to(device)
    return context, h_state


@torch.no_grad()
def run_uncontrolled_inference(model, meas_seq, true_seq_norm, num_particles, device,
                                pinv_init=False):
    if model is EDH_SENTINEL:
        return None, None, None

    state_dim = model.state_dim
    total_steps = meas_seq.shape[0]

    predictions = torch.zeros((total_steps, state_dim))
    particle_history = torch.zeros((total_steps, num_particles, state_dim))
    predicted_std = torch.zeros((total_steps, state_dim)) if model.loss_type == 'gaussian_nll' else None

    x_start = (
        true_seq_norm[0]
        .unsqueeze(0)
        .unsqueeze(1)
        .expand(-1, num_particles, -1)
        .float()
    )
    x_curr = x_start

    predictions[0] = true_seq_norm[0].cpu()
    particle_history[0] = x_curr.squeeze(0).cpu()

    # Run the state_rnn's actual t=0 forward pass via _gru_init (deterministic
    # here since model.eval() -> h0_scale=0 inside _gru_init) rather than
    # zero-initializing state_context directly -- state_context is the GRU's
    # *output* of embedding the true initial state, not just its hidden-state
    # seed. Zero-initializing it skipped that embedding entirely, feeding
    # garbage context into the very first prediction and compounding over the
    # whole rollout. Matches _unroll's and on_validation_epoch_end's own
    # training-time convention exactly.
    N = model.num_particles
    if pinv_init:
        state_context, h_state_rnn = pinv_init_context(model, true_seq_norm[0], device)
    else:
        state_context, h_state_rnn = model._gru_init(
            true_seq_norm[0].unsqueeze(0), x_start, batch_size=1)

    # For gaussian_nll: track uncertainty state
    if model.loss_type == 'gaussian_nll':
        N = num_particles
        prev_log_var = torch.full(
            (1, N, state_dim), math.log(model.min_variance), device=device)
        uncertainty_h = torch.zeros(
            1, N, model.uncertainty_gru_dim * (2 if getattr(model, 'uncertainty_rnn_lstm', False) else 1),
            device=device)
        # t=0's state is handed to the model exactly (predictions[0] =
        # true_seq_norm[0]), so the true uncertainty here is 0 -- display
        # that directly rather than sqrt(min_variance) (a placeholder floor
        # value, never a real var_head prediction; the first actual
        # var_head-computed estimate is predicted_std[1]). prev_log_var
        # itself is untouched -- it still seeds at log(min_variance) as the
        # real input feature var_head consumes at t=1, matching what the
        # checkpoint was trained on.
        predicted_std[0] = torch.zeros(state_dim)

    for t in range(1, total_steps):
        current_meas = meas_seq[t].unsqueeze(0).float()

        # Last step's uncertainty state, fed back into physics_predictor/
        # measurement_flow when the checkpoint was trained with
        # uncertainty_conditioning=True (see model_legacy.py). Built BEFORE
        # this step's var_head call overwrites prev_log_var/uncertainty_h
        # below, same ordering as _unroll's training-time convention.
        uncertainty_signal = (
            torch.cat([prev_log_var.detach(),
                       model._uncertainty_h_readout(uncertainty_h).detach()], dim=-1)
            if getattr(model, 'uncertainty_conditioning', False) else None)

        # See model_legacy.py's direct_init_predict docstring / _unroll's
        # identical handling -- only the very first call (t=1) bypasses
        # context_readout(state_context) in favour of the raw, exact x_0.
        raw_state_override = (
            x_start if (getattr(model, 'direct_init_predict', False) and t == 1) else None)

        particles_pred, _ = model(
            particles_prev=x_curr,
            state_context=state_context,
            current_meas=current_meas,
            uncertainty_signal=uncertainty_signal,
            raw_state_override=raw_state_override,
        )

        mean_est = particles_pred.mean(dim=1)
        predictions[t] = mean_est.squeeze(0).cpu()
        particle_history[t] = particles_pred.squeeze(0).cpu()

        # Compute predicted variance for gaussian_nll
        if model.loss_type == 'gaussian_nll':
            z_sensor = current_meas[..., :model.sensor_obs_dim]
            z_expanded = z_sensor.unsqueeze(1).expand(-1, N, -1)
            innovation = z_expanded - model._meas_predict(particles_pred, state_context)

            raw_log_var = model.var_head(
                torch.cat([innovation, prev_log_var,
                           model._uncertainty_h_readout(uncertainty_h)], dim=-1))
            var = torch.nn.functional.softplus(raw_log_var) + model.min_variance
            predicted_std[t] = torch.sqrt(var[0, 0]).cpu()

            prev_log_var = torch.log(var)
            uncertainty_state_dim = uncertainty_h.shape[-1]
            uncertainty_h = model._uncertainty_rnn_step(
                innovation.reshape(1 * N, model.sensor_obs_dim),
                uncertainty_h.reshape(1 * N, uncertainty_state_dim),
            ).reshape(1, N, uncertainty_state_dim)

        # Update state
        if model.per_particle_context:
            # Per-particle GRU: each particle feeds its own value, not the
            # ensemble mean -- matches model_legacy.py's own _gru_step
            # (particle_value = particles_pred when per_particle_context),
            # and is required for the reshape below to even be shape-valid
            # (mean_est is [1, state_dim], not [1, N, state_dim]).
            N = model.num_particles
            batch_size = 1
            gru_in_dim = particles_pred.shape[-1]
            _, h_state_rnn = model.state_rnn(
                particles_pred.reshape(batch_size * N, 1, gru_in_dim), h_state_rnn)
            state_context = h_state_rnn[-1].reshape(batch_size, N, model.hparams.state_gru_dim)
        else:
            state_context = model.state_rnn(mean_est, h_state_rnn)
            h_state_rnn = state_context
        x_curr = particles_pred

    return predictions, particle_history, predicted_std


def compute_acoustic_rmse(predictions, true_physical):
    pos_error = predictions[1:, :2] - true_physical[1:, :2]
    return torch.sqrt((pos_error ** 2).mean()).item()


def compute_tdoa_rmse(predictions, true_physical):
    # Only compute RMSE on the X and Y positional states, ignoring velocities
    pos_error = predictions[1:, :2] - true_physical[1:, :2]
    return torch.sqrt((pos_error ** 2).mean()).item()


def compute_lorenz_rmse(predictions, true_physical):
    return torch.sqrt(torch.mean((predictions[1:] - true_physical[1:]) ** 2)).item()


def load_uncontrolled_data(args):
    dm = FileTrackingDataModule(
        data_path=args.data_path,
        seq_len=10,
        batch_size=256,
        dataset_type=args.dataset_type,
        use_minmax_norm=args.equation_scaling,
    )
    dm.setup()
    return dm


def override_integration_steps(model, integration_steps):
    """
    Recompute the λ-schedule buffer for a different integration_steps count.
    integration_steps itself is just bookkeeping — forward() actually walks
    model.lam_steps, a buffer baked once at construction time from
    integration_steps + lam_schedule (see ModularNeuralODEFilter.__init__),
    so it must be rebuilt here using the checkpoint's own lam_schedule.
    """
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
    """
    Force SSMStylePhysicsPredictor's residual mode post-load, overriding
    whatever was saved in the checkpoint's hparams (model.hparams.
    ssm_residual_predict). Only the forward-pass switch changes — net's
    weights/shapes are identical either way, so this is purely diagnostic:
    it lets you check whether a checkpoint's weights make more sense under
    the other convention, e.g. after the residual/non-residual default
    changed underneath an already-trained checkpoint.
    """
    if hasattr(model, 'physics_predictor') and hasattr(model.physics_predictor, 'residual'):
        model.physics_predictor.residual = residual


# Sentinel placed into `models` in place of an nn.Module when --include_edh
# is set — run_acoustic_eval dispatches to run_edh_inference() instead of
# run_uncontrolled_inference() whenever it sees this marker.
EDH_SENTINEL = object()


@torch.no_grad()
def run_edh_inference(edh_module, meas_seq, true_seq_norm, num_particles, dm, num_steps):
    """
    Classical EDH baseline — the exact closed-form Daum-Huang flow (analytical
    linearisation at the prior mean: A, b built from the measurement Jacobian
    H, particle covariance P, and measurement noise R; no learned components
    at all).

    edh_module : baseline_edh (acoustic) / baseline_edh_tdoa (tdoa) /
                 baseline_edh_lorenz (lorenz) — see EDH_MODULES. Supplies
                 predict() (deterministic prior mean -- linear CV matmul for
                 acoustic/tdoa, RK4 Lorenz63 ODE step for lorenz),
                 Q_chol (process noise), and exact_daum_huang_step() (h_func/
                 R baked in) for that dataset's own real generative model.

    NOT routed through ModularNeuralODEFilter: even with known_dynamics=True
    and known_measurement=True, that model's measurement_flow (PFFODEFunc)
    is unconditionally a learned net — a freshly-constructed instance would
    apply random, untrained weights as the "correction" step, which isn't an
    EDH update at all.

    Each edh_module's predict()/Q_chol/h_func/R operate entirely in physical
    units (and are hardcoded to device='cpu'), so this function denormalises
    at the entry point and renormalises the result at the exit point — every
    other model in this script returns normalised predictions, and the eval
    loops denormalise via `predictions * t_std + t_mean` afterwards, so EDH
    needs to match that same contract.
    """
    t_mean = dm.t_mean.squeeze().cpu()
    t_std  = dm.t_std.squeeze().cpu()
    m_mean = dm.m_mean.squeeze().cpu()
    m_std  = dm.m_std.squeeze().cpu()

    state_dim   = true_seq_norm.shape[-1]
    total_steps = meas_seq.shape[0]

    true_seq_phys = true_seq_norm.cpu() * t_std + t_mean
    meas_seq_phys = meas_seq.cpu() * m_std + m_mean

    predictions_phys      = torch.zeros((total_steps, state_dim))
    particle_history_phys = torch.zeros((total_steps, num_particles, state_dim))

    x_start        = true_seq_phys[0].unsqueeze(0).repeat(num_particles, 1)
    particles_curr = x_start + torch.randn(num_particles, state_dim) @ edh_module.Q_chol.T
    predictions_phys[0]      = true_seq_phys[0]
    particle_history_phys[0] = particles_curr

    for t in range(1, total_steps):
        particles_prior = (edh_module.predict(particles_curr)
                            + torch.randn(num_particles, state_dim) @ edh_module.Q_chol.T)
        particles_posterior = edh_module.exact_daum_huang_step(
            particles_prior, meas_seq_phys[t], num_steps=num_steps)

        predictions_phys[t]      = particles_posterior.mean(dim=0)
        particle_history_phys[t] = particles_posterior
        particles_curr = particles_posterior

    predictions      = (predictions_phys - t_mean) / t_std
    particle_history = (particle_history_phys - t_mean) / t_std
    return predictions, particle_history


def prepare_models(args):
    model_cls = load_model_class(args.dataset_type)
    models = []
    for ckpt in args.ckpt_paths:
        model = model_cls.load_from_checkpoint(ckpt, map_location=args.device, strict=False)
        if args.num_particles is not None and hasattr(model, 'num_particles'):
            model.num_particles = args.num_particles
        if args.integration_steps is not None:
            override_integration_steps(model, args.integration_steps)
        if args.override_ssm_residual is not None:
            override_ssm_residual(model, args.override_ssm_residual == 'true')
        model.eval().to(args.device)
        models.append(model)
        print(f"  Loaded: {ckpt}")
    if args.include_edh:
        if args.dataset_type not in EDH_MODULES:
            raise ValueError(
                f"--include_edh has no EDH baseline for dataset_type="
                f"{args.dataset_type} -- only {list(EDH_MODULES.keys())} are "
                f"supported (see EDH_MODULES).")
        # Match EDH's λ-integration resolution to the model's own, unless the
        # caller explicitly asked for a different --edh_num_steps -- an
        # unequal step count silently confounds any RMSE/uncertainty
        # difference with a difference in integration resolution instead of
        # the thing actually being compared. See --edh_num_steps' docstring.
        steps_seen = sorted({m.integration_steps for m in models if hasattr(m, 'integration_steps')})
        if args.edh_num_steps is None:
            args.edh_num_steps = steps_seen[0]
            possessive = "models'" if len(models) > 1 else "model's"
            print(f"  EDH λ-integration steps: {args.edh_num_steps} "
                  f"(matched to the {possessive} own integration_steps)")
        if len(steps_seen) > 1:
            print(f"  WARNING: loaded checkpoints disagree on integration_steps "
                  f"{steps_seen} -- EDH is matched to only "
                  f"{args.edh_num_steps if args.edh_num_steps in steps_seen else steps_seen[0]}, "
                  f"not all of them. Pass --integration_steps to force every "
                  f"model to the same resolution for a clean comparison.")
        models.append(EDH_SENTINEL)
        print(f"  Using: EDH (closed-form Daum-Huang baseline, "
              f"{EDH_MODULES[args.dataset_type].__name__}.py, "
              f"num_steps={args.edh_num_steps})")
    return models


def parse_labels(args):
    if args.labels is None:
        labels = [f"Model {i + 1}" for i in range(len(args.ckpt_paths))]
    else:
        if len(args.labels) != len(args.ckpt_paths):
            raise ValueError("--labels must match --ckpt_paths length")
        labels = list(args.labels)
    if args.include_edh:
        labels.append('EDH')
    return labels


def run_acoustic_eval(args):
    print("--- Acoustic Evaluation ---")
    dm = load_uncontrolled_data(args)
    models = prepare_models(args)
    labels = parse_labels(args)

    t_mean = dm.t_mean.squeeze().cpu()
    t_std = dm.t_std.squeeze().cpu()
    num_val_trajs = dm.val_dataset.measurements.shape[0]

    os.makedirs(args.save_dir, exist_ok=True)

    traj_colors = ['red', 'blue', 'gold', 'purple', 'orange']
    model_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    all_rmses = {label: [] for label in labels}

    for eval_idx in range(args.num_eval):
        print(f"\nFigure {eval_idx + 1}/{args.num_eval}")
        fig, ax = plt.subplots(figsize=(12, 10))

        if args.traj_indices is not None:
            traj_indices = torch.tensor(args.traj_indices)
        else:
            traj_indices = torch.randperm(num_val_trajs)[:args.num_traj]

        print(f"  Trajectory indices: {traj_indices.tolist()}")

        gt_plotted = set()
        prediction_data = {'eval_idx': eval_idx, 'models': {}}

        for model_idx, (model, label) in enumerate(zip(models, labels)):
            pred_style = model_styles[model_idx % len(model_styles)]
            prediction_data['models'][label] = {}

            for traj_pos, traj_idx in enumerate(traj_indices):
                traj_idx = traj_idx.item()
                color = traj_colors[traj_pos % len(traj_colors)]
                print(f"  [{label}] trajectory {traj_idx}")

                meas_seq = dm.val_dataset.measurements[traj_idx].to(args.device)
                true_seq_norm = dm.val_dataset.true_states[traj_idx].to(args.device)

                if model is EDH_SENTINEL:
                    predictions, particle_history = run_edh_inference(
                        EDH_MODULES[args.dataset_type], meas_seq, true_seq_norm,
                        args.edh_num_particles, dm, args.edh_num_steps
                    )
                else:
                    predictions, particle_history, _ = run_uncontrolled_inference(
                        model, meas_seq, true_seq_norm, args.num_particles, args.device,
                        pinv_init=args.pinv_init
                    )

                true_phys = true_seq_norm.cpu() * t_std + t_mean
                pred_phys = predictions * t_std + t_mean
                parts_phys = particle_history * t_std.unsqueeze(0) + t_mean.unsqueeze(0)

                rmse = compute_acoustic_rmse(pred_phys, true_phys)
                all_rmses[label].append(rmse)

                prediction_data['models'][label][traj_idx] = {
                    'mean_prediction': pred_phys,
                    'particles': parts_phys,
                    'rmse': rmse,
                }

                if args.show_particles:
                    ax.scatter(
                        parts_phys[:, :, 0].numpy().ravel(),
                        parts_phys[:, :, 1].numpy().ravel(),
                        color=color, s=4, alpha=0.08, zorder=2, linewidths=0,
                    )

                if traj_idx not in gt_plotted:
                    ax.plot(
                        true_phys[:, 0].numpy(), true_phys[:, 1].numpy(),
                        color=color, linestyle='-', linewidth=2, alpha=0.5, zorder=3,
                    )
                    ax.plot(
                        true_phys[0, 0], true_phys[0, 1],
                        'o', color=color, markersize=8, markeredgecolor='black', zorder=10,
                    )
                    ax.plot(
                        true_phys[-1, 0], true_phys[-1, 1],
                        'X', color=color, markersize=8, markeredgecolor='black', zorder=10,
                    )
                    gt_plotted.add(traj_idx)

                ax.plot(
                    pred_phys[:, 0].numpy(), pred_phys[:, 1].numpy(),
                    color=color, linestyle=pred_style, linewidth=2, zorder=5,
                )

        ax.set_xlabel('X Position', fontsize=20)
        ax.set_ylabel('Y Position', fontsize=20)
        ax.tick_params(labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='best', fontsize='large')

        data_path = os.path.join(args.save_dir, f'acoustic_eval_{eval_idx}.pt')
        plot_path = os.path.join(args.save_dir, f'acoustic_eval_{eval_idx}.png')
        torch.save(prediction_data, data_path)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved data to '{data_path}'")
        print(f"  Saved plot to '{plot_path}'")

    for label, values in all_rmses.items():
        if values:
            print(f"  {label}: mean position RMSE = {sum(values) / len(values):.4f} m")


def run_tdoa_eval(args):
    print("--- TDOA Tracking Evaluation ---")
    dm = load_uncontrolled_data(args)
    models = prepare_models(args)
    labels = parse_labels(args)

    t_mean = dm.t_mean.squeeze().cpu()
    t_std = dm.t_std.squeeze().cpu()
    num_val_trajs = dm.val_dataset.measurements.shape[0]

    os.makedirs(args.save_dir, exist_ok=True)

    # Attempt to extract sensor positions from the saved dataset metadata
    # Fallback to standard coordinates if missing
    L_REF = dm.norm_stats.get('L_REF', [-3.0, 0.0])
    L_RX = dm.norm_stats.get('L_RX', [3.0, 0.0])
    # Second TDOA sensor pair (perpendicular baseline) -- only present for
    # datasets generated after tdoa_data_gen.py added the second pair; older
    # single-pair datasets simply won't have these keys.
    L_REF2 = dm.norm_stats.get('L_REF2', None)
    L_RX2 = dm.norm_stats.get('L_RX2', None)

    traj_colors = ['red', 'blue', 'gold', 'purple', 'orange']
    model_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    all_rmses = {label: [] for label in labels}

    for eval_idx in range(args.num_eval):
        print(f"\nFigure {eval_idx + 1}/{args.num_eval}")
        fig, ax = plt.subplots(figsize=(12, 10))

        if args.traj_indices is not None:
            traj_indices = torch.tensor(args.traj_indices)
        else:
            traj_indices = torch.randperm(num_val_trajs)[:args.num_traj]

        print(f"  Trajectory indices: {traj_indices.tolist()}")

        gt_plotted = set()
        prediction_data = {'eval_idx': eval_idx, 'models': {}}

        # Plot Sensors
        ax.scatter(L_REF[0], L_REF[1], c='black', marker='^', s=150, label='Pair-1 Ref Sensor', zorder=6)
        ax.scatter(L_RX[0], L_RX[1], c='purple', marker='^', s=150, label='Pair-1 Rx Sensor', zorder=6)
        if L_REF2 is not None and L_RX2 is not None:
            ax.scatter(L_REF2[0], L_REF2[1], c='dimgray', marker='s', s=150, label='Pair-2 Ref Sensor', zorder=6)
            ax.scatter(L_RX2[0], L_RX2[1], c='darkorchid', marker='s', s=150, label='Pair-2 Rx Sensor', zorder=6)

        for model_idx, (model, label) in enumerate(zip(models, labels)):
            pred_style = model_styles[model_idx % len(model_styles)]
            prediction_data['models'][label] = {}

            for traj_pos, traj_idx in enumerate(traj_indices):
                traj_idx = traj_idx.item()
                color = traj_colors[traj_pos % len(traj_colors)]
                print(f"  [{label}] trajectory {traj_idx}")

                meas_seq = dm.val_dataset.measurements[traj_idx].to(args.device)
                true_seq_norm = dm.val_dataset.true_states[traj_idx].to(args.device)

                if model is EDH_SENTINEL:
                    predictions, particle_history = run_edh_inference(
                        EDH_MODULES[args.dataset_type], meas_seq, true_seq_norm,
                        args.edh_num_particles, dm, args.edh_num_steps
                    )
                else:
                    predictions, particle_history, _ = run_uncontrolled_inference(
                        model, meas_seq, true_seq_norm, args.num_particles, args.device,
                        pinv_init=args.pinv_init
                    )

                true_phys = true_seq_norm.cpu() * t_std + t_mean
                pred_phys = predictions * t_std + t_mean
                parts_phys = particle_history * t_std.unsqueeze(0) + t_mean.unsqueeze(0)

                rmse = compute_tdoa_rmse(pred_phys, true_phys)
                all_rmses[label].append(rmse)

                prediction_data['models'][label][traj_idx] = {
                    'mean_prediction': pred_phys,
                    'particles': parts_phys,
                    'rmse': rmse,
                }

                if args.show_particles:
                    ax.scatter(
                        parts_phys[:, :, 0].numpy().ravel(),
                        parts_phys[:, :, 1].numpy().ravel(),
                        color=color, s=4, alpha=0.08, zorder=2, linewidths=0,
                    )

                if traj_idx not in gt_plotted:
                    ax.plot(
                        true_phys[:, 0].numpy(), true_phys[:, 1].numpy(),
                        color=color, linestyle='-', linewidth=2, alpha=0.5, zorder=3,
                        label='True Trajectory' if len(gt_plotted) == 0 else None
                    )
                    ax.plot(
                        true_phys[0, 0], true_phys[0, 1],
                        'o', color=color, markersize=8, markeredgecolor='black', zorder=10,
                    )
                    ax.plot(
                        true_phys[-1, 0], true_phys[-1, 1],
                        'X', color=color, markersize=8, markeredgecolor='black', zorder=10,
                    )
                    gt_plotted.add(traj_idx)

                ax.plot(
                    pred_phys[:, 0].numpy(), pred_phys[:, 1].numpy(),
                    color=color, linestyle=pred_style, linewidth=2, zorder=5,
                    label=f'{label}' if traj_pos == 0 else None
                )

        ax.set_title('TDOA Tracking Evaluation', fontsize=22)
        ax.set_xlabel('X Position (km)', fontsize=20)
        ax.set_ylabel('Y Position (km)', fontsize=20)
        ax.tick_params(labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='best', fontsize='large')

        data_path = os.path.join(args.save_dir, f'tdoa_eval_{eval_idx}.pt')
        plot_path = os.path.join(args.save_dir, f'tdoa_eval_{eval_idx}.png')
        torch.save(prediction_data, data_path)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved data to '{data_path}'")
        print(f"  Saved plot to '{plot_path}'")

    for label, values in all_rmses.items():
        if values:
            print(f"  {label}: mean position RMSE = {sum(values) / len(values):.4f} km")


def run_lorenz_eval(args):
    print("--- Lorenz '63 Evaluation ---")
    dm = load_uncontrolled_data(args)
    models = prepare_models(args)
    labels = parse_labels(args)

    t_mean = dm.t_mean.squeeze().cpu()
    t_std = dm.t_std.squeeze().cpu()
    num_val_trajs = dm.val_dataset.measurements.shape[0]

    os.makedirs(args.save_dir, exist_ok=True)

    model_colors = ['red', 'orange', 'purple', 'magenta', 'cyan']

    for eval_idx in range(args.num_eval):
        print(f"\nFigure {eval_idx + 1}/{args.num_eval}")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if args.traj_indices is not None:
            traj_idx = args.traj_indices[eval_idx % len(args.traj_indices)]
        else:
            traj_idx = torch.randint(0, num_val_trajs, (1,)).item()

        print(f"  Trajectory index: {traj_idx}")

        meas_seq = dm.val_dataset.measurements[traj_idx].to(args.device)
        true_seq_norm = dm.val_dataset.true_states[traj_idx].to(args.device)
        true_physical = true_seq_norm.cpu() * t_std + t_mean

        ax.plot(
            true_physical[:, 0], true_physical[:, 1], true_physical[:, 2],
            label='True Trajectory', color='blue', linewidth=2, alpha=0.5, zorder=3,
        )
        ax.scatter(
            *[true_physical[0, i].item() for i in range(3)],
            c='green', marker='o', s=100, edgecolors='black', zorder=10, label='Start',
        )
        ax.scatter(
            *[true_physical[-1, i].item() for i in range(3)],
            c='black', marker='X', s=100, edgecolors='black', zorder=10, label='End',
        )

        prediction_data = {'evaluation_group': eval_idx, 'models': {}}

        for model_idx, (model, label) in enumerate(zip(models, labels)):
            color = model_colors[model_idx % len(model_colors)]
            print(f"  -> '{label}' on trajectory {traj_idx}")

            if model is EDH_SENTINEL:
                preds_norm, _ = run_edh_inference(
                    EDH_MODULES[args.dataset_type], meas_seq, true_seq_norm,
                    args.edh_num_particles, dm, args.edh_num_steps
                )
            else:
                preds_norm, _, _ = run_uncontrolled_inference(
                    model, meas_seq, true_seq_norm, args.num_particles, args.device,
                    pinv_init=args.pinv_init
                )
            pred_physical = preds_norm * t_std + t_mean
            rmse = compute_lorenz_rmse(pred_physical, true_physical)
            print(f"     RMSE: {rmse:.4f}")

            prediction_data['models'][label] = {
                'trajectory_index': traj_idx,
                'mean_prediction': pred_physical,
                'rmse': rmse,
            }

            ax.plot(
                pred_physical[:, 0], pred_physical[:, 1], pred_physical[:, 2],
                label=label, color=color, linestyle='--', linewidth=2, zorder=5,
            )

        sensor_pos = [6 * math.sqrt(2), 6 * math.sqrt(2), 27.0]
        ax.scatter(*sensor_pos, marker='^', color='green', s=150,
                   edgecolors='black', label='Sensor', zorder=6)
        ax.set_xlabel('$X_1$', fontsize=14)
        ax.set_ylabel('$X_2$', fontsize=14)
        ax.set_zlabel('$X_3$', fontsize=14, labelpad=20, rotation=90)
        handles, labels_leg = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels_leg, loc='best', fontsize='large')
        for lh in leg.legend_handles:
            lh.set_alpha(1)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        data_path = os.path.join(args.save_dir, f'lorenz_eval_{eval_idx}.pt')
        plot_path = os.path.join(args.save_dir, f'lorenz_eval_{eval_idx}.png')
        torch.save(prediction_data, data_path)
        plt.savefig(plot_path, dpi=300, bbox_inches=None, pad_inches=0.3)
        plt.close(fig)

def build_parser():
    parser = argparse.ArgumentParser(description='Unified eval script for acoustic, lorenz, and tdoa datasets.')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['acoustic', 'lorenz', 'tdoa']),
    parser.add_argument('--device', type=str, required=False,
                        choices=['cuda', 'cpu']),
    parser.add_argument('--ckpt_paths', nargs='+', required=True,
                        help='Checkpoint path(s) to evaluate.')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Optional labels for each checkpoint.')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Dataset path. Defaults to the standard path for the selected dataset type.')
    parser.add_argument('--save_dir', type=str, default='eval_outputs')
    parser.add_argument('--num_particles', type=int, default=100)
    parser.add_argument('--pinv_init', action='store_true',
                        help='Seed state_context at t=0 by analytically inverting '
                             'context_readout against the true initial state, instead of '
                             'the approximate reconstruction a cold _gru_init pass '
                             'produces (measured ~2.4km off on a real trajectory despite '
                             'exact input). No retraining -- uses context_readout\'s '
                             'existing weights via pseudo-inverse. Requires '
                             'ssm_style_predict=True, ssm_wide_context_input=False.')
    parser.add_argument('--edh_num_particles', type=int, default=None,
                        help='Particle count for EDH baseline. Defaults to --num_particles.')
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
                        help='Number of trajectories to plot per figure (for 2D setups).')
    parser.add_argument('--traj_indices', nargs='+', type=int, default=None,
                        help='Optional fixed trajectory indices.')
    parser.add_argument('--show_particles', action='store_true',
                        help='Overlay the raw particle cloud (every particle, '
                             'every timestep) beneath the mean prediction and '
                             'ground truth, in the same color as that '
                             'trajectory. Acoustic and TDOA only.')
    parser.add_argument('--log_transform', action='store_true')
    parser.add_argument('--equation_scaling', action='store_true',
                        help='Use min-max normalisation (yscale = ymax - ymin, '
                             'Eq. 13 arXiv:2103.15341). Must match the flag '
                             'used when the checkpoint was trained.')
    parser.add_argument('--include_edh', action='store_true',
                        help='Add the classical EDH baseline as an extra '
                             'model in the comparison, labeled "EDH" — the '
                             'exact closed-form Daum-Huang flow (analytical '
                             'linearisation at the prior mean; no checkpoint, '
                             'no learned parameters). Supported for '
                             'dataset_type in EDH_MODULES (acoustic, tdoa, '
                             'lorenz), each with its own dynamics/measurement '
                             'model matching that dataset\'s real generative '
                             'process (baseline_edh.py / baseline_edh_tdoa.py '
                             '/ baseline_edh_lorenz.py).')
    parser.add_argument('--edh_num_steps', type=int, default=None,
                        help='Number of λ-integration steps for the EDH '
                             'baseline\'s closed-form flow (exact_daum_huang_'
                             'step). Defaults to matching the (first) model\'s '
                             'own integration_steps (its checkpoint value, or '
                             '--integration_steps if that\'s set) -- an '
                             'apples-to-apples resolution comparison against '
                             'the model\'s own λ-homotopy solve requires the '
                             'same step count on both sides; a mismatch here '
                             'silently confounds any RMSE/uncertainty '
                             'difference between the two with a difference in '
                             'integration resolution instead. Pass explicitly '
                             'to deliberately test EDH at a different '
                             'resolution than the model.')
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

    if args.dataset_type == 'acoustic':
        run_acoustic_eval(args)
    elif args.dataset_type == 'lorenz':
        run_lorenz_eval(args)
    elif args.dataset_type == 'tdoa':
        run_tdoa_eval(args)
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")


if __name__ == '__main__':
    main()