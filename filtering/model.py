"""
Legacy model wrapper - checkpoint from ssm_style_run_gaussian_nll_mlp_varhead.
Trained with uncertainty_rnn + var_head for heteroscedastic Gaussian NLL.
Config: loss_type=gaussian_nll, ssm_style_predict=true, per_particle_context=true,
        uncertainty_gru_dim=16, state_gru_dim=32, num_particles=1

Fully learned dynamics + measurement only -- known_dynamics/known_measurement
(analytical F/Q, analytical h(x)) have been moved to model_known_node.py,
which pairs known/exact dynamics and measurement with a learned residual on
top of the analytical Daum-Huang particle-flow update, for direct comparison
against this file's fully-learned counterpart.
"""
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from solver import build_measurement_ode_solver


# ─────────────────────────────────────────────────────────────────────────────
#  Physics Predictor
# ─────────────────────────────────────────────────────────────────────────────
class DiscretePhysicsPredictor(nn.Module):
    """
    Predicts a residual state delta from GRU context (and optionally control).

    control_dim=0  — lightweight 2-layer MLP on context alone (acoustic)
    control_dim>0  — deeper 3-layer MLP on [context ‖ control] (Ackermann)

    motion_hidden_dim=None (default) — width defaults to hidden_dim
                          (control_dim>0) or context_dim (control_dim=0),
                          matching pre-existing behaviour exactly.
    motion_hidden_dim=int — overrides the hidden width for either branch,
                          decoupling motion-model capacity from hidden_dim.

    per_particle_context=False (default) — state_context is [B, context_dim],
                          shared across particles; broadcast via .expand().
    per_particle_context=True  — state_context is already [B, N, context_dim]
                          (its own vector per particle); used directly, no
                          broadcast. control (if any) is still per-batch and
                          always broadcast regardless of this flag.

    Used when ssm_style_predict=False (see ModularNeuralODEFilter docstring).
    """

    def __init__(self, state_dim, hidden_dim, context_dim, control_dim=0,
                 motion_hidden_dim=None, per_particle_context=False):
        super().__init__()
        self.use_control          = control_dim > 0
        self.per_particle_context = per_particle_context

        if self.use_control:
            width  = motion_hidden_dim if motion_hidden_dim is not None else hidden_dim
            in_dim = context_dim + control_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.LayerNorm(width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.SiLU(),
                nn.Linear(width, state_dim),
            )
        else:
            width = motion_hidden_dim if motion_hidden_dim is not None else context_dim
            self.input_norm = nn.LayerNorm(context_dim)
            self.fc1  = nn.Linear(context_dim, width)
            self.fc2  = nn.Linear(width, state_dim)
            self.act  = nn.SiLU()
            self.skip = nn.Linear(context_dim, state_dim)

    def forward(self, x, state_context, control=None):
        ctx = (state_context if self.per_particle_context
               else state_context.unsqueeze(1).expand(-1, x.shape[1], -1))
        if self.use_control:
            ctrl_exp = control.unsqueeze(1).expand(-1, x.shape[1], -1)
            return x + self.net(torch.cat([ctx, ctrl_exp], dim=-1))
        else:
            ctx   = self.input_norm(ctx)
            delta = self.fc2(self.act(self.fc1(ctx))) + self.skip(ctx)
            return x + delta


# ─────────────────────────────────────────────────────────────────────────────
#  SSM-style Physics Predictor (EXPERIMENTAL)
# ─────────────────────────────────────────────────────────────────────────────
class SSMStylePhysicsPredictor(nn.Module):
    """
    Transition function operating only on the GRU's recurrent context h_t
    (+ control) — never on particles_prev. Two modes, selected by `residual`:

    residual=False — direct: f(h_t, u_t) = delta(...). The output IS
                the predicted next state, mirroring a textbook nonlinear SSM
                transition x_{t+1} = f(x_t, u_t). (Original ssm_style_predict
                behaviour.)
    residual=True  — f(h_t, u_t) = x_t + delta(...). Mirrors the
                near-identity structure of a linear SSM transition (F @ x,
                whose diagonal blocks would carry state forward by default
                for a near-identity F), giving delta an easy "stay near x_t"
                default instead of needing to learn the whole transition from
                scratch. delta's final layer is zero-initialised when this
                mode is active (see ModularNeuralODEFilter._init_weights),
                so f(h_t, u) = x_t exactly at the start of training —
                passive-start, same convention as innov_skip/out_layer/
                meas_model.

    x_t — the per-particle readout of h_t down to the literal physical
                state_dim that the rest of the filter (meas_model, the ODE
                flow) operates on. Always present as the residual anchor
                when residual=True; never the particle's own previous
                value.

    wide_context_input=False (default) — original architecture: the readout
                Linear(state_gru_dim, state_dim) lives in
                ModularNeuralODEFilter.context_readout and runs *before*
                this module; only x_t (state_dim-wide) is passed in, and
                delta is computed from x_t alone — i.e. delta only ever
                sees the same lossy state_dim-wide projection as the
                residual anchor. (Original ssm_style_predict behaviour;
                checkpoints trained before this knob existed load
                unchanged — the in_dim of `net` and the absence of
                `self.readout` here exactly reproduce the old structure.)
    wide_context_input=True — the readout moves into this module
                (self.readout) and h_t (state_gru_dim-wide, unprojected) is
                passed in instead of x_t. x_t is still derived internally
                via self.readout(h_t) and used as the residual anchor, but
                delta is now computed from the *full* h_t — recovering
                whatever information context_readout would otherwise have
                discarded before delta could use it (trend, multi-step
                shape, etc. that doesn't fit in a single state_dim point
                estimate). Requires state_gru_dim.

    Only valid together with per_particle_context=True: x_t must already be
    per-particle for the ensemble to retain any deterministic-path diversity
    (there is no `+particles_prev` skip here to fall back on) — enforced in
    ModularNeuralODEFilter.__init__.

    control_dim=0  — MLP on x_t (or h_t) alone.
    control_dim>0  — MLP on [x_t ‖ control] (or [h_t ‖ control]).

    uncertainty_dim=0 (default) — no uncertainty conditioning (original
                architecture exactly).
    uncertainty_dim>0  — MLP additionally sees [... ‖ uncertainty_signal],
                a [B, N, uncertainty_dim] tensor the caller builds from the
                previous step's gaussian_nll uncertainty state (prev_log_var
                ‖ uncertainty_h — see ModularNeuralODEFilter's
                uncertainty_conditioning flag). Lets the dynamics prediction
                see how uncertain the filter was last step, closing the loop
                that var_head's output otherwise never feeds back into.

    num_layers=1 (default) — single hidden block (Linear→LayerNorm→SiLU)
                followed by the output Linear, i.e. the original
                ssm_style_predict architecture exactly (checkpoints trained
                before this knob existed load unchanged).
    num_layers>1 — num_layers-1 extra (Linear(width,width)→LayerNorm→SiLU)
                blocks are inserted before the output Linear, increasing
                depth without changing width.

    input_layernorm=False (default) — original architecture: `base` (what
                delta/`net` consumes -- x_t, or h_t under wide_context_input)
                is fed to net's first Linear unnormalised.
    input_layernorm=True  — a LayerNorm(state_dim) (or LayerNorm(state_gru_dim)
                under wide_context_input) is applied to `base` before it
                reaches net's first Linear -- NOT to `x_t` when it's used as
                the residual anchor (`x_t + net(inp)`), which stays in its
                original scale so the skip connection still means "predict
                no change" in physical units. Targets a specific, measured
                problem: `base`'s distribution at t=1 (from _gru_init, a
                once-per-trajectory cold GRU pass) differs measurably from
                its distribution at t>=2 (from _gru_step, ~N_steps real
                recurrent updates) -- e.g. a 0.47 std-dev average per-
                dimension shift, up to 2.9 std devs in the worst dimension,
                measured directly on a trained checkpoint. Every fix tried
                so far that touches *what* reaches delta (raw x_0 substitution,
                an auxiliary reconstruction loss) has failed or only
                partially helped, because delta itself stays sensitive to
                which distribution its input came from. LayerNorm instead
                makes delta robust to *that* it can shift, independent of why
                (Ba, Kiros & Hinton, "Layer Normalization", arXiv:1607.06450,
                2016) -- no new loss term, no assumption about what the
                correct t=1 representation should look like.
    """

    def __init__(self, state_dim, hidden_dim, control_dim=0, motion_hidden_dim=None,
                 residual=False, num_layers=1, wide_context_input=False,
                 state_gru_dim=None, uncertainty_dim=0, input_layernorm=False):
        super().__init__()
        self.use_control        = control_dim > 0
        self.use_uncertainty    = uncertainty_dim > 0
        self.residual            = residual
        self.wide_context_input = wide_context_input
        self.use_input_layernorm = input_layernorm
        width  = motion_hidden_dim if motion_hidden_dim is not None else hidden_dim

        extra_dim = (control_dim if self.use_control else 0) + uncertainty_dim
        if wide_context_input:
            self.readout = nn.Linear(state_gru_dim, state_dim)
            net_in_dim = state_gru_dim + extra_dim
            base_dim   = state_gru_dim
        else:
            net_in_dim = state_dim + extra_dim
            base_dim   = state_dim
        if input_layernorm:
            self.input_norm = nn.LayerNorm(base_dim)

        layers = [nn.Linear(net_in_dim, width), nn.LayerNorm(width), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU()]
        layers.append(nn.Linear(width, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state_input, control=None, uncertainty_signal=None):
        # state_input is x_t (state_dim-wide) when wide_context_input=False,
        # h_t (state_gru_dim-wide) when wide_context_input=True — see class
        # docstring. x_t is always what the residual anchors to; `base` is
        # always what delta consumes.
        if self.wide_context_input:
            x_t  = self.readout(state_input)
            base = state_input
        else:
            x_t  = state_input
            base = state_input
        if self.use_input_layernorm:
            base = self.input_norm(base)
        parts = [base]
        if self.use_control:
            parts.append(control.unsqueeze(1).expand(-1, base.shape[1], -1))
        if self.use_uncertainty:
            parts.append(uncertainty_signal)
        inp = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return x_t + self.net(inp) if self.residual else self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
#  Measurement Model  h_θ(x)
# ─────────────────────────────────────────────────────────────────────────────
class LearnedMeasurementModel(nn.Module):
    """
    Learns h(x): state → measurement space.

    context_dim=0  — maps x alone: h(x), no trajectory-history conditioning
                      at all. `context` is accepted but ignored.
    context_dim>0  — maps [x ‖ state_context] (standard mode).

    per_particle_context=False (default) — state_context is [B, context_dim],
                      shared across particles; broadcast via .expand().
                      Ignored entirely when context_dim=0.
    per_particle_context=True  — state_context is already [B, N, context_dim];
                      used directly, no broadcast.

    num_layers=1 (default) — single hidden block (Linear→SiLU) followed by
                      the output Linear, i.e. the original architecture
                      exactly (checkpoints trained before this knob existed
                      load unchanged).
    num_layers>1 — num_layers-1 extra (Linear(hidden,hidden)→SiLU) blocks
                      are inserted before the output Linear, increasing
                      depth without changing width.

    use_layernorm=False (default) — no LayerNorm, original architecture
                      exactly (checkpoints trained before this knob existed
                      load unchanged).
    use_layernorm=True  — inserts LayerNorm(hidden_dim) after every hidden
                      Linear, before SiLU: [Linear, LayerNorm, SiLU] blocks,
                      matching SSMStylePhysicsPredictor's convention. Added
                      because this module is the only one of the four
                      learned submodules with no normalisation anywhere —
                      its first hidden layer's gradient norm was observed to
                      sit flat/non-decaying for an entire 26k-step training
                      run while every analogous hidden layer elsewhere
                      (including this module's own output layer) showed a
                      clear multi-x decay over the same run.
    """

    def __init__(self, state_dim, obs_dim, hidden_dim, context_dim,
                 per_particle_context=False, num_layers=1, use_layernorm=False):
        super().__init__()
        self.context_dim          = context_dim
        self.per_particle_context = per_particle_context
        if use_layernorm:
            layers = [nn.Linear(state_dim + context_dim, hidden_dim),
                      nn.LayerNorm(hidden_dim), nn.SiLU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim),
                           nn.LayerNorm(hidden_dim), nn.SiLU()]
        else:
            layers = [nn.Linear(state_dim + context_dim, hidden_dim), nn.SiLU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, context=None):
        # x       : [B, N, state_dim]
        # context : [B, context_dim] or [B, N, context_dim] (per_particle_context)
        if self.context_dim == 0:
            return self.net(x)
        ctx = (context if self.per_particle_context
               else context.unsqueeze(1).expand(-1, x.shape[1], -1))
        return self.net(torch.cat([x, ctx], dim=-1))   # [B, N, obs_dim]


# ─────────────────────────────────────────────────────────────────────────────
#  PFF ODE Function
# ─────────────────────────────────────────────────────────────────────────────
class PFFODEFunc(nn.Module):
    """
    Neural ODE vector field  dx/dλ = f_θ(x, λ, innovation).
    Driven purely by the explicit innovation  z − h_θ(x, state_context).

    uncertainty_dim=0 (default) — no uncertainty conditioning (original
                architecture exactly).
    uncertainty_dim>0  — vector field additionally conditions on
                uncertainty_signal (see SSMStylePhysicsPredictor's docstring
                for the same convention — this is the measurement-update
                side of the same feedback path). Held fixed across the whole
                λ=0→1 integration for a given real timestep, same as the
                frozen innovation_0 convention (see forward()'s ssm_style
                branch) — it reflects last step's uncertainty, not something
                that evolves during this step's homotopy solve.

    Optional techniques from Kim et al., "Stiff Neural Ordinary Differential
    Equations" (arXiv:2103.15341):

    deep_gelu_ode=False (default) — 2-hidden-layer SiLU MLP of width hidden_dim.
    deep_gelu_ode=True  — deep/narrow GELU stack (ode_depth layers of ode_width
                          nodes), mirroring the paper's successful ROBER
                          architecture (Sec. IV.A.2), which mitigates gradient
                          pathologies from scale-separated dynamics.

    ode_residual=False (default, deep_gelu_ode only) — hidden blocks are plain
                          feedforward.
    ode_residual=True  (deep_gelu_ode only) — each hidden block becomes a
                          residual block (h = h + block(h)), improving gradient
                          flow at depth.

    Equation scaling (Eq. 13, arXiv:2103.15341): dy/dt = NN(y,t) * yscale.
    Implemented here as a fixed (non-learnable) per-channel buffer,
    `correction_scale`, multiplying the raw network output before the
    innovation skip connection. Defaults to all-ones (no-op) and is set
    post-construction via ModularNeuralODEFilter.set_equation_scaling(),
    using the SAME value the loss divides by (Eq. 14) — see calculate_loss.
    This is a model-side mechanism, distinct from (and on top of) dataset-
    level input normalisation, which solves the unrelated problem of
    cross-network gradient-scale disparity.
    """

    def __init__(self, state_dim, obs_dim, hidden_dim,
                 deep_gelu_ode=False, ode_depth=6, ode_width=5,
                 ode_residual=False, uncertainty_dim=0):
        super().__init__()
        self.uncertainty_dim = uncertainty_dim
        in_dim = state_dim + obs_dim + 1 + uncertainty_dim
        self.innov_skip       = nn.Linear(obs_dim, state_dim, bias=False)
        self.deep_gelu_ode    = deep_gelu_ode
        self.ode_residual     = ode_residual
        self.register_buffer('correction_scale', torch.ones(state_dim))

        if deep_gelu_ode:
            self.in_proj = nn.Linear(in_dim, ode_width)
            self.in_act  = nn.GELU()

            blocks = []
            for _ in range(ode_depth - 1):
                blocks.append(nn.Sequential(
                    nn.Linear(ode_width, ode_width), nn.GELU(),
                ))
            self.hidden_blocks = nn.ModuleList(blocks)

            self.out_proj = nn.Linear(ode_width, state_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, state_dim),
            )

    @property
    def out_layer(self):
        """Final Linear layer, for passive-start zero-init regardless of architecture."""
        return self.out_proj if self.deep_gelu_ode else self.net[-1]

    def forward(self, lam, x, innovation, uncertainty_signal=None):
        lam_t = (lam if isinstance(lam, torch.Tensor)
                 else torch.tensor(lam, dtype=x.dtype, device=x.device))
        lam_vec = lam_t.expand(x.shape[0], x.shape[1], 1)
        if self.uncertainty_dim > 0:
            inp = torch.cat([x, innovation, lam_vec, uncertainty_signal], dim=-1)
        else:
            inp = torch.cat([x, innovation, lam_vec], dim=-1)

        if self.deep_gelu_ode:
            h = self.in_act(self.in_proj(inp))
            for block in self.hidden_blocks:
                h = h + block(h) if self.ode_residual else block(h)
            out = self.out_proj(h)
        else:
            out = self.net(inp)

        out = out * self.correction_scale
        return out + self.innov_skip(innovation)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Filter
# ─────────────────────────────────────────────────────────────────────────────
class ModularNeuralODEFilter(pl.LightningModule):
    """
    Neural ODE particle filter with fully learned dynamics and measurement --
    unified model covering both operational modes (use_control). Known/exact
    dynamics and measurement (analytical F/Q, analytical h(x)) live in the
    companion model_known_node.py instead, paired there with a learned
    residual on top of the analytical Daum-Huang particle-flow update.

    ── Control mode ──────────────────────────────────────────────────────────
    use_control=False  batch: (obs, states, mask)
    use_control=True   batch: (obs, states, controls, mask)

    ── Dynamics / measurement ────────────────────────────────────────────────
    physics_predictor   DiscretePhysicsPredictor (ssm_style_predict=False) or
                         SSMStylePhysicsPredictor (ssm_style_predict=True) --
                         always learned.
    meas_model           LearnedMeasurementModel — always learned.

    ── Phase training ────────────────────────────────────────────────────────
    training_phase=1   skip_update=True; loss on prior only (predictor warmup)
    training_phase=2   full filter; loss on posterior (default)

    ── Augmented Neural ODE ─────────────────────────────────────────────────
    aug_dim=0   no augmentation (default)
    aug_dim>0   pads the state with `aug_dim` zero-initialised channels before
                the homotopy ODE solve (Dupont et al., "Augmented Neural
                ODEs"), giving the flow's vector field extra room to represent
                trajectories that can't cross in state_dim-only space. The
                measurement model never sees the augmented channels; they are
                dropped after integration (forward() slices back to
                state_dim before returning).

    ── Equation scaling (Eq. 13/14, arXiv:2103.15341) ───────────────────────
    equation_scaling=False  correction_scale stays all-ones (no-op).
    equation_scaling=True   main.py computes a fixed, data-derived, per-
                state-dimension scale from consecutive-timestep deltas of
                the (already-normalised) training trajectories
                (FileTrackingDataModule.correction_scale) and registers it
                via set_equation_scaling(). The SAME value multiplies the
                ODE flow's raw output (Eq. 13, in PFFODEFunc.forward) and
                divides predictions/targets before the loss (Eq. 14, in
                calculate_loss) — distinct from dataset-level input
                normalisation, which addresses cross-network gradient-scale
                disparity rather than the ODE's output magnitude.

    ── Process noise (removed) / particle diversity ──────────────────────────
    No process-noise term of any kind (fixed or learned) is injected during
                propagation, ever, regardless of num_particles -- forward()'s
                particles_prev -> x_prior (== _predict's raw output) is
                always a fully deterministic chain, every step, every
                particle. Q/R modelling was deliberately dropped
                project-wide, not just the learnable variant: both are
                "system identification" in the classical Kalman sense, and
                if the system were well-identified enough to support that, a
                classical/EKF/UKF filter (or KalmanNet, which assumes known
                f/h and only learns the gain) would likely be the right tool
                rather than this fully end-to-end neural model -- see
                session history.
    particle_init_diversity_std (default 0.0) is NOT a process-noise/Q
                substitute, and is deliberately NOT injected every step the
                way process_noise_std used to be -- doing so would just be
                Q under a new name. Instead it perturbs ONLY the initial
                particle cloud, once, in _unroll (particles_curr at t=0):
                every particle otherwise starts from the identical true
                state and propagation is fully deterministic, so without
                this num_particles>1 would collapse to N identical
                trajectories for the entire sequence (see session history).
                This is closer to a bootstrap particle filter's prior/
                proposal initialisation (start from a spread-out initial
                hypothesis set, then evolve deterministically) than to a
                per-step noise model. No-op whenever num_particles=1,
                regardless of its value. Fixed, non-learned, for the same
                reason process_noise_std_init was before removal: nothing in
                the loss ties it to any "correct" value -- it's an
                engineering knob for initial proposal spread, not a claim
                about physical noise.
    ── EXPERIMENTAL: per-particle GRU context ────────────────────────────────
    per_particle_context=False (default) — state_rnn is a single nn.GRUCell;
                one context vector [B, state_gru_dim] shared by every
                particle (matches pre-existing behaviour exactly).
    per_particle_context=True  — state_rnn becomes an nn.GRU with
                gru_num_layers layers; every particle carries its OWN
                hidden state [B, N, state_gru_dim], evolved from its own
                history (particles keep a fixed identity across time here —
                no resampling — so this is well-defined). Implemented as a
                single shared-weight GRU applied with (batch, particle)
                folded into one dimension, not N separate networks.
                physics_predictor/meas_model receive this per-particle
                context directly (no broadcast).
    gru_num_layers=1 (default) — stacked GRU depth, only meaningful when
                per_particle_context=True.

    ── EXPERIMENTAL: context-free measurement model ─────────────────────────
    meas_use_context=True (default) — LearnedMeasurementModel sees
                [x ‖ state_context] (matches pre-existing behaviour).
    meas_use_context=False — LearnedMeasurementModel sees x alone (h(x) with
                no trajectory-history conditioning).

    ── EXPERIMENTAL: SSM-style physics prediction ───────────────────────────
    ssm_style_predict=False (default) — physics_predictor is
                DiscretePhysicsPredictor: x_next = particles_prev +
                delta(state_context [, control]) (residual, conditioned on
                context, matches pre-existing behaviour).
    ssm_style_predict=True  — physics_predictor becomes
                SSMStylePhysicsPredictor: x_hat_{t+1} = f(x_t [, control]),
                depending only on x_t (NOT particles_prev) — where x_t is
                the per-particle readout of state_context through a new
                Linear(state_gru_dim, state_dim) layer, context_readout.
                Requires per_particle_context=True (raises ValueError
                otherwise): x_t must already be per-particle, since there is
                no +particles_prev skip left to carry diversity if context
                were shared. Forces meas_use_context=False: h must see only
                f's output (x_hat), never state_context directly. The
                measurement innovation is frozen once at x_prior (post-noise,
                the λ-transport's entry point) rather than recomputed at the
                evolving x(λ) — NOT the continuous learned-measurement
                default (see ssm_continuous_innovation below).
    ssm_residual_predict=False (default) — f(x_t, u) = delta(x_t, u)
                directly (no skip). Matches the original ssm_style_predict
                behaviour exactly — kept as the default so checkpoints
                trained before this flag existed keep loading correctly
                (it's simply absent from their saved hparams, and the
                constructor default reproduces their exact forward pass).
    ssm_residual_predict=True  — f(x_t, u) = x_t + delta(x_t, u). delta's
                final layer is zero-initialised in this mode (see
                _init_weights), so f(x_t, u) = x_t exactly at init —
                identity/passive start, same convention as innov_skip/
                out_layer/meas_model. Recommended for new ssm_style_predict
                runs (see SSMStylePhysicsPredictor docstring for rationale).
                No effect when ssm_style_predict=False.
    ssm_predictor_layernorm=False (default) — physics_predictor's delta
                network consumes its state input (x_t, or h_t under
                ssm_wide_context_input) unnormalised.
    ssm_predictor_layernorm=True — a LayerNorm is applied to that input
                before delta's first layer (not to x_t when used as the
                residual anchor, which stays in physical-state scale). See
                SSMStylePhysicsPredictor's input_layernorm docstring for the
                full rationale and citation. No effect when
                ssm_style_predict=False.
    ssm_continuous_innovation=False (default) — ssm_style_predict's
                innovation is frozen, evaluated once at x_prior and reused
                at every λ-integration step. Original ssm_style_predict
                behaviour — kept as the default so checkpoints trained
                before this flag existed keep loading/evaluating correctly.
    ssm_continuous_innovation=True — innovation is recomputed at every
                λ-step from the evolving x(λ) instead (z − h(x(λ),
                state_context)), matching the default learned-measurement
                branch's convention (see forward()'s final `else`). Makes
                the λ-homotopy closed-loop: as h(x(λ)) approaches z, the
                ODE's driving force self-attenuates, instead of being
                pushed toward a fixed target for the whole integration —
                a plausible fix for overshoot/oscillation at low
                integration_steps. One extra cheap meas_model forward per
                integration step versus the frozen default. No effect when
                ssm_style_predict=False.

    ── EXPERIMENTAL: adjoint / adaptive-step ODE integration ─────────────────
    adaptive_steps=False (default)  measurement_solver is the fixed-step
                rk4/euler stepper (see integration_type), unaffected by
                adjoint. Also True whenever integration_type='adaptive',
                regardless of this flag's own value.
    adaptive_steps=True   measurement_solver instead uses torchdiffeq's
                adaptive Dopri5 integrator (solver.py's
                _integrate_torchdiffeq) — lazily imports torchdiffeq, so
                this raises ImportError only if actually used without it
                installed. integration_steps/lam_schedule are then unused
                (torchdiffeq chooses its own step sizes to hit rtol/atol).
    adjoint=False (default)  when adaptive_steps=True, backprop stores the
                full forward computational graph (torchdiffeq.odeint) —
                more memory, exact gradients.
    adjoint=True   when adaptive_steps=True, backprop instead uses
                torchdiffeq's adjoint sensitivity method (odeint_adjoint) —
                O(1) memory in the number of solver steps at the cost of a
                second (backward-time) ODE solve for the gradient, and
                gradients that are only as accurate as that second solve.
                No effect when adaptive_steps=False (ported from
                model_particles.py's identical convention, see its
                docstring — model.py didn't support this at all
                until now).
    inference_rtol=1e-4, inference_atol=1e-5 — torchdiffeq's error
                tolerances, only consulted when adaptive_steps=True.

    ── EXPERIMENTAL: uncertainty feedback into state estimation ─────────────
    uncertainty_conditioning=True (default, changed from False -- see
                session history) — var_head's AND prior_var_head's
                uncertainty estimates are fed forward as extra conditioning
                input to measurement_flow (the homotopy measurement update)
                only -- physics_predictor never sees either, keeping this
                scoped to the measurement update alone. In spirit (not
                literal Kalman-gain math -- neither head implements a real
                gain equation) this mimics Q and R both acting at the
                correction step: prior_var_head's output is computed on
                particles_prior (post-propagation, pre-measurement -- "how
                much did the motion model just inject", Q-flavoured) and
                var_head's is computed on the innovation ("how surprising
                was the last measurement", R-flavoured); measurement_flow
                now sees both. uncertainty_signal is [prev_log_var
                (state_dim) ‖ uncertainty_h (uncertainty_gru_dim) ‖
                prior_log_var (state_dim, computed inside forward() from
                prior_var_head applied to the first two channels)]. NOT
                detached -- gradients from the state-tracking loss reach
                both var_head's and prior_var_head's weights through this
                path. This deliberately reopens a previously-documented
                failure mode (see _unroll's uncertainty_signal construction
                comment and session history): an earlier, detached version
                of this same mechanism was found to collapse var_head's
                predicted variance to the min_variance floor within a few
                epochs when NOT detached, because the tracking loss could
                reach var_head's weights through exactly this kind of path.
                Not detaching now is a deliberate experiment, not an
                oversight -- watch mean_pred_var/min_pred_var (and their
                prior_var_head equivalents once added) for the same
                monotonic-collapse signature if this doesn't work out.
    uncertainty_conditioning=False  — both heads are read-only side-channels
                (original architecture exactly): they score their
                respective quantities but never influence how those
                quantities were computed.
                Requires loss_type='gaussian_nll' (raises ValueError
                otherwise -- prev_log_var/uncertainty_h only exist in that
                mode) and ssm_style_predict=True (the only physics predictor
                wired up to accept uncertainty_dim so far, though
                physics_predictor's own uncertainty_dim is always 0 now
                regardless) -- since this is now the DEFAULT, constructing
                with other defaults (loss_type='huber', ssm_style_predict=
                False) will raise immediately unless uncertainty_conditioning
                is explicitly set to False, or loss_type/ssm_style_predict
                are set to satisfy the requirement.

    ── EXPERIMENTAL: LSTM cell for uncertainty_rnn ───────────────────────────
    uncertainty_rnn_lstm=False (default) — uncertainty_rnn is a
                nn.GRUCell(sensor_obs_dim, uncertainty_gru_dim), a single
                hidden state carrying its cross-step memory (original
                architecture exactly).
    uncertainty_rnn_lstm=True  — uncertainty_rnn is an
                nn.LSTMCell(sensor_obs_dim, uncertainty_gru_dim) instead,
                giving it a separate cell state with its own additive,
                forget-gated update path (rather than GRU's more tightly
                coupled reset/update gating) -- forget-gate bias initialised
                to 1 (Jozefowicz, Zaremba & Sutskever, "An Empirical
                Exploration of Recurrent Network Architectures", ICML 2015),
                so the cell starts by defaulting to remembering rather than
                forgetting. Externally, uncertainty_h stays a single tensor
                at every call site (_unroll, on_validation_epoch_end,
                no_control_eval.py) -- internally it packs [h ‖ c]
                ([B, N, 2*uncertainty_gru_dim] instead of
                [B, N, uncertainty_gru_dim]) via _uncertainty_rnn_step, and
                only the h half is ever exposed to var_head or
                uncertainty_signal via _uncertainty_h_readout, so
                var_head_in_dim and uncertainty_conditioning's
                _uncertainty_dim are completely unaffected by this flag.

    ── Prior-belief NLL (loss_type='gaussian_nll' only) ──────────────────────
    A second, independent heteroscedastic-NLL head, prior_var_head, scores
                particles_prior (x_prior, the pre-measurement prediction)
                against the true state -- var_head above only ever scored
                the final, measurement-corrected particle, so prior belief
                was previously untrained/unpenalised entirely. Structurally
                identical to var_head (Linear -> SiLU -> Linear, output
                state_dim + 1 -- see var_head's construction comment for the
                K-component-mixture logit convention, which applies
                identically here and is likewise dead weight at
                num_particles=1). Conditioned ONLY on prev_log_var and
                uncertainty_h -- the same two cross-step feedback channels
                var_head uses, minus innovation (doesn't exist yet -- no
                measurement has been consulted) and minus state_context (for
                the same capacity/gradient-competition reason var_head
                excludes it, see _gaussian_nll_step's docstring). Reads
                prev_log_var/uncertainty_h as a snapshot rather than
                maintaining its own recurrent state -- see
                _prior_gaussian_nll_step's docstring.

                This is deliberately NOT a Q/R substitute and does not
                attempt to recover any "true" physical noise value -- like
                var_head, it's a direct heteroscedastic regression head (Nix
                & Weigend 1994) scored by its own proper NLL, so there's no
                indirection for training to fail to identify (contrast with
                the removed learnable Q, which fed an unrelated head with no
                structural tie back to it -- see session history). A
                structural coupling between prior_var_head and var_head
                (e.g. forcing posterior variance <= prior variance) was
                considered and rejected: it would bake in "the measurement
                can only ever reduce uncertainty," which only holds under
                classical well-specified linear-Gaussian assumptions this
                project has deliberately opted out of (see session history)
                -- the two heads are fully independent.
    prior_nll_weight=1.0 — the two NLL terms are combined as
                nll_post + prior_nll_weight * nll_prior (train_step/
                validation_step). No principled derivation exists for this
                weight (that would need a known noise ratio, exactly what
                dropping Q/R gives up) -- 1.0 (unweighted sum) is a
                starting point, not a tuned value; adjust empirically if
                train_nll_prior/train_nll_post logs show one term
                dominating and starving the other of gradient signal.

    ── EXPERIMENTAL: per-gate orthogonal init for state_rnn ──────────────────
    state_rnn_per_gate_orthogonal=False (default) — _init_weights applies
                nn.init.orthogonal_ to state_rnn's whole weight_hh matrix
                ([3*state_gru_dim, state_gru_dim], GRU's three stacked
                reset/update/candidate gates) as a single flat block. This
                gives the stacked matrix orthonormal columns as a whole, but
                NOT an orthogonal transform for each individual gate --
                original architecture exactly (also still applied to
                uncertainty_rnn regardless of this flag).
    state_rnn_per_gate_orthogonal=True — state_rnn's weight_hh is instead
                split into its 3 [state_gru_dim, state_gru_dim] gate blocks
                (reset, update, candidate) and each is orthogonally
                initialised independently. This is the norm-preservation
                property "orthogonal RNN init" is actually meant to
                guarantee (Saxe, McClelland & Ganguli, "Exact solutions to
                the nonlinear dynamics of learning in deep linear neural
                networks", ICLR 2014 workshop track / arXiv:1312.6120) --
                each gate's hidden-to-hidden map is what's applied
                *repeatedly* across timesteps, so its individual spectral
                properties (not the stacked block's) govern whether
                _gru_init's single cold pass over x_0 lands in a
                predictably-scaled region of state_context space, rather
                than an arbitrarily-scaled one. Motivation: _gru_init's cold
                pass is the one place state_rnn's initial-condition
                sensitivity actually matters (every other call already has a
                real, non-arbitrary hidden state carried forward from
                _gru_step) -- see direct_init_predict's and
                ssm_predictor_layernorm's docstrings for the broader
                cold-start diagnosis this targets a different facet of.
                Only affects state_rnn, not uncertainty_rnn (kept on the
                original whole-matrix scheme for a clean single-variable
                comparison).

    ── EXPERIMENTAL: learned initial-hidden-state encoder ───────────────────
    learned_init_encoder=False (default) — _gru_init seeds state_rnn's h0
                from noise (std=0.1 during training, exactly 0 at eval, see
                _gru_init) -- original architecture exactly. h0 carries no
                information about the true initial state either way
                (verified: recreating training's exact noise distribution at
                eval time changed nothing, t=1 prediction error identical to
                4 decimal places) -- the network has to reconstruct the
                initial state through a single cold recurrent step every
                time, which is what actually causes the large first-step
                error (measured ~2.4km off on a real trajectory despite
                _gru_init's *input* being the exact true state).
    learned_init_encoder=True  — h0 is instead produced by a small learned
                Linear(state_dim, state_gru_dim), self.init_encoder, applied
                to the true initial state: h0 = init_encoder(x_0). This is
                the standard encoder-conditioned initial-hidden-state
                pattern from seq2seq models (Sutskever, Vinyals & Le 2014;
                Cho et al. 2014's GRU paper) — seeding a recurrent state from
                a learned encoding of available conditioning information,
                rather than zero/noise. Trained jointly with the rest of the
                network via the normal state-tracking loss, so (unlike
                analytically inverting context_readout, which was tried and
                made t=1 error ~3x worse) init_encoder learns to produce an
                h0 that physics_predictor/context_readout actually know how
                to use, not just one that's numerically exact.
    init_recon_weight=0.0 (default) — weight lambda_init on an auxiliary
                reconstruction term added directly to the training loss:

                    L_init = || context_readout(init_encoder(x_0)) - x_0 ||^2
                    L_total = L_main + lambda_init * L_init

                computed once per trajectory (t=0 only, inside _unroll), no
                effect when learned_init_encoder=False. Only x_0 gets this
                treatment: it is the one point in the whole rollout where the
                target is known exactly and is exactly init_encoder's own
                input, so asking context_readout(init_encoder(x_0)) to
                reproduce it is a well-posed identity constraint, unlike
                every other timestep's context_readout(state_context_t),
                whose "correct" output legitimately incorporates process/
                measurement uncertainty and shouldn't be pinned to a single
                exact target.
                Motivation: init_encoder fires once per trajectory (vs.
                state_rnn's ~N_steps firings), so the ordinary state-tracking
                loss reaches it only through a long, diluted path (through
                every subsequent _predict/measurement_flow call in the
                rollout) — measured via grad_norm/init_encoder.* sitting
                ~100-230x smaller than the dominant layers, and empirically,
                200 epochs of training under only that diluted signal left
                context_readout(init_encoder(x_0)) no more accurate than at
                epoch 0. A first attempt at fixing this by boosting
                init_encoder's LR (~15x, see git history) changed the raw
                gradient-magnitude ratio a lot (~1:230 -> ~1:15) but left the
                actual reconstruction error and the eval-time cold-start
                variance spike essentially unchanged — confirming the
                problem is that the gradient reaching init_encoder isn't
                specifically *about* x_0-reconstruction accuracy (a magnitude
                problem would have responded to a larger step; this didn't).
                L_init fixes that directly by giving init_encoder its own
                short, undiluted path to the loss. This is the same "deep
                supervision" idea as GoogLeNet's auxiliary classifier heads
                (Szegedy et al., "Going Deeper with Convolutions", CVPR 2015,
                Sec. 4: extra loss heads attached to intermediate layers
                specifically so those layers get a stronger, more direct
                gradient than what survives backpropagating through the rest
                of the network), formalized as "companion objectives" by Lee,
                Xie, Gallagher, Zhang & Tu, "Deeply-Supervised Nets", AISTATS
                2015. Recommended starting point: lambda_init=1.0 (same
                order of magnitude as the main per-step state loss; increase
                if grad_norm/init_encoder.* is still small relative to
                grad_norm/measurement_flow.* once training is underway).

    ── EXPERIMENTAL: bypass context_readout for the very first prediction ───
    direct_init_predict=False (default) — unchanged: physics_predictor's
                first call (t=1) consumes context_readout(state_context),
                where state_context comes from _gru_init's single cold pass
                over x_0 — the same path used at every later timestep.
    direct_init_predict=True — physics_predictor's first call instead
                consumes x_0 directly (the raw, exact initial state, already
                available as particles_prev but previously discarded --
                see _predict's "particles is NOT consulted" comment).
                context_readout/state_context are only ever asked to
                reconstruct a state value starting at t=2 (the corrected
                posterior from t=1, via the ordinary _gru_step -- no
                different from what it already does at every t>=2). No
                effect when ssm_style_predict=False or ssm_wide_context_input=
                True (see __init__ validation).
                Rationale: under ssm_style_predict=True, state_context has
                exactly one functional consumer -- physics_predictor via
                context_readout (meas_use_context is force-disabled in this
                mode, see __init__, and _gaussian_nll_step's state_context
                argument is likewise unused by _meas_predict here) -- so the
                only way state_context's t=1 value (from _gru_init, a
                once-per-trajectory computation no other GRU call resembles)
                can hurt anything is through that one call. Removing it there
                removes the defect at its root, rather than trying to make
                _gru_init's cold pass reconstruct x_0 accurately (the
                learned_init_encoder / init_recon_weight approach above) --
                the GRU ends up doing the *same kind* of update at every
                single step (real corrected state + previous hidden state ->
                new hidden state), t=1 included, with no special round-trip
                for context_readout to fail at. This is also just how a
                recursive filter is supposed to start: x_0 is known exactly,
                so the first prediction should use it directly, and a
                learned recurrent summary should only start accumulating
                once posterior estimates exist to summarize.
                Caveat: physics_predictor now sees two different kinds of
                input across a trajectory -- raw x_0 at t=1, and
                context_readout(state_context) (a GRU-processed summary,
                not necessarily identical to the raw corrected state) at
                every t>=2. If context_readout already tracks the corrected
                state closely at t>=2 (plausible given reasonable
                val_rmse_position numbers) this gap is small; if not,
                physics_predictor's very first prediction could behave
                differently from its steady-state behaviour. Worth checking
                directly against a trained checkpoint (compare
                context_readout(state_context_t) to particles_pred_{t-1}
                for t>=2) rather than assuming either way.
    """

    def __init__(self,
                 # ── Core dims ─────────────────────────────────────────────
                 state_dim=4,
                 obs_dim=4,
                 hidden_dim=64,
                 state_gru_dim=16,
                 motion_hidden_dim=None,
                 meas_hidden_dim=None,
                 # ── EXPERIMENTAL architecture changes ────────────────────
                 per_particle_context=False,
                 gru_num_layers=1,
                 state_rnn_per_gate_orthogonal=False,
                 meas_use_context=True,
                 ssm_style_predict=False,
                 ssm_residual_predict=False,
                 ssm_predictor_layernorm=False,
                 ssm_predictor_layers=1,
                 meas_model_layers=1,
                 meas_model_layernorm=False,
                 ssm_wide_context_input=False,
                 ssm_continuous_innovation=False,
                 uncertainty_conditioning=True,
                 learned_init_encoder=False,
                 init_recon_weight=0.0,
                 direct_init_predict=False,
                 # ── Control ───────────────────────────────────────────────
                 use_control=False,
                 control_dim=2,
                 training_phase=2,
                 # ── Particles & training ──────────────────────────────────
                 num_particles=50,
                 lr=1e-3,
                 integration_steps=4,
                 loss_type='huber',
                 huber_delta=1.0,
                 min_variance=1e-2,
                 prior_nll_weight=1.0,
                 uncertainty_gru_dim=16,
                 uncertainty_rnn_lstm=False,
                 skip_update=False,
                 integration_type='rk4',
                 adjoint=False,
                 adaptive_steps=False,
                 inference_rtol=1e-4,
                 inference_atol=1e-5,
                 lam_schedule='exponential',
                 # ── Particle diversity (num_particles>1 only) ─────────────
                 particle_init_diversity_std=0.0,
                 # ── Stiff-ODE techniques (arXiv:2103.15341) ──────────────────
                 equation_scaling=False,
                 deep_gelu_ode=False,
                 ode_depth=6,
                 ode_width=5,
                 ode_residual=False,
                 # ── Augmented Neural ODE ──────────────────────────────────
                 aug_dim=0,
                 # ── Sensor obs slice ──────────────────────────────────────
                 sensor_obs_dim=None,
                 # ── Legacy / unused ───────────────────────────────────────
                 context_dim=32,
                 ode_func_type='deep'):
        super().__init__()
        self.save_hyperparameters()

        self.state_dim        = state_dim
        self.num_particles    = num_particles
        self.lr               = lr
        self.integration_steps = integration_steps
        self.loss_type        = loss_type.lower()
        self.huber_delta      = huber_delta
        self.min_variance     = min_variance
        self.skip_update      = skip_update
        self.integration_type = integration_type
        self.adjoint          = adjoint
        self.adaptive_steps   = adaptive_steps or integration_type == 'adaptive'
        self.aug_dim          = aug_dim
        self.per_particle_context = per_particle_context
        self.gru_num_layers   = gru_num_layers
        self.state_rnn_per_gate_orthogonal = state_rnn_per_gate_orthogonal
        self.ssm_style_predict = ssm_style_predict
        self.ssm_residual_predict = ssm_residual_predict
        self.ssm_predictor_layernorm = ssm_predictor_layernorm
        self.ssm_wide_context_input = ssm_wide_context_input
        self.ssm_continuous_innovation = ssm_continuous_innovation
        self.uncertainty_conditioning = uncertainty_conditioning
        self.learned_init_encoder = learned_init_encoder
        self.init_recon_weight = init_recon_weight
        self.direct_init_predict = direct_init_predict
        if ssm_style_predict and not per_particle_context:
            raise ValueError("ssm_style_predict requires per_particle_context=True "
                              "(x_t must already be per-particle, since there is no "
                              "+particles_prev skip to fall back on for diversity)")
        if uncertainty_conditioning and loss_type.lower() != 'gaussian_nll':
            raise ValueError("uncertainty_conditioning requires loss_type='gaussian_nll' "
                              "-- prev_log_var/uncertainty_h (the fed-back uncertainty "
                              "state) only exist in that mode.")
        if uncertainty_conditioning and not ssm_style_predict:
            raise ValueError("uncertainty_conditioning requires ssm_style_predict=True "
                              "-- SSMStylePhysicsPredictor is the only physics predictor "
                              "wired up to accept uncertainty_dim so far (DiscretePhysics"
                              "Predictor is unchanged).")
        if init_recon_weight > 0 and not learned_init_encoder:
            raise ValueError("init_recon_weight > 0 requires learned_init_encoder=True "
                              "-- there is no init_encoder to supervise otherwise.")
        if (init_recon_weight > 0 and
                not (ssm_style_predict and not ssm_wide_context_input)):
            raise ValueError("init_recon_weight > 0 requires self.context_readout to exist, "
                              "i.e. ssm_style_predict=True, ssm_wide_context_input=False -- "
                              "context_readout is only constructed under that combination "
                              "(see its construction site).")
        if (direct_init_predict and
                not (ssm_style_predict and not ssm_wide_context_input)):
            raise ValueError("direct_init_predict=True requires ssm_style_predict=True, "
                              "ssm_wide_context_input=False -- it only has meaning for "
                              "_predict's context_readout(h_state) branch (see "
                              "direct_init_predict's docstring).")
        if ssm_style_predict:
            # h must see only f's output (x_hat), never the raw context
            # directly — same contract as meas_use_context=False.
            meas_use_context = False
        self.meas_use_context = meas_use_context

        # Eq. 13/14 "yscale" analog: fixed, data-derived per-channel scale,
        # shared between the ODE flow's output and the loss. Defaults to
        # all-ones (no-op); set via set_equation_scaling() when
        # equation_scaling=True (see main.py).
        self.register_buffer('correction_scale', torch.ones(state_dim))

        # Perturbs ONLY the initial particle cloud (t=0, in _unroll) when
        # num_particles>1 -- see __init__'s "Process noise (removed) /
        # particle diversity" docstring section for why this is not a
        # process-noise substitute.
        self.register_buffer(
            'particle_init_diversity_std',
            torch.tensor(particle_init_diversity_std, dtype=torch.float32))

        if self.loss_type not in ['l1', 'l2', 'huber', 'mse', 'gaussian_nll']:
            raise ValueError("loss_type must be 'l1', 'l2'/'mse', 'huber', or 'gaussian_nll'")

        # ── Sensor obs slice ──────────────────────────────────────────────────
        _sensor_obs_dim     = sensor_obs_dim if sensor_obs_dim is not None else obs_dim
        self.sensor_obs_dim = _sensor_obs_dim

        # ── Uncertainty-feedback conditioning dim ─────────────────────────────
        # measurement_flow's uncertainty_signal is now [prev_log_var
        # (state_dim) ‖ uncertainty_h (uncertainty_gru_dim) ‖ prior_log_var
        # (state_dim, computed inside forward() from prior_var_head using the
        # first two channels -- see forward())] -- available regardless of
        # loss_type since uncertainty_gru_dim is always a constructor arg;
        # only actually built/passed at call time when loss_type=
        # 'gaussian_nll' (enforced above). Only measurement_flow ever
        # receives it -- physics_predictor's own uncertainty_dim is always
        # 0, see uncertainty_conditioning's docstring. NOT detached (see
        # uncertainty_conditioning's docstring) -- deliberately reopens a
        # previously-documented collapse risk, see session history.
        _uncertainty_dim = (2 * state_dim + uncertainty_gru_dim) if uncertainty_conditioning else 0

        # ── State GRU (always present) ────────────────────────────────────────
        gru_in_dim = state_dim
        if per_particle_context:
            # Per-particle hidden state: one shared-weight GRU, applied with
            # (batch, particle) folded into a single dimension — see
            # _gru_init/_gru_step. nn.GRU (not the Cell variant) for built-in
            # multi-layer support; called one step (seq_len=1) at a time.
            self.state_rnn = nn.GRU(gru_in_dim, state_gru_dim, num_layers=gru_num_layers,
                                    batch_first=True)
        else:
            self.state_rnn = nn.GRUCell(gru_in_dim, state_gru_dim)

        # ── Learned initial-hidden-state encoder (see class docstring) ────────
        if learned_init_encoder:
            self.init_encoder = nn.Linear(state_dim, state_gru_dim)

        # ── Physics Predictor ──────────────────────────────────────────────────
        if ssm_style_predict:
            if not ssm_wide_context_input:
                # x_t readout: projects the GRU's recurrent context
                # (state_gru_dim, decoupled capacity) down to the
                # literal physical state_dim that f/g operate on. Lives
                # here (not inside the predictor) only in this mode —
                # see SSMStylePhysicsPredictor docstring.
                self.context_readout = nn.Linear(state_gru_dim, state_dim)
            self.physics_predictor = SSMStylePhysicsPredictor(
                state_dim          = state_dim,
                hidden_dim         = hidden_dim,
                control_dim        = control_dim if use_control else 0,
                motion_hidden_dim  = motion_hidden_dim,
                residual           = ssm_residual_predict,
                num_layers         = ssm_predictor_layers,
                wide_context_input = ssm_wide_context_input,
                state_gru_dim      = state_gru_dim,
                input_layernorm    = ssm_predictor_layernorm,
            )
        else:
            self.physics_predictor = DiscretePhysicsPredictor(
                state_dim            = state_dim,
                hidden_dim           = hidden_dim,
                context_dim          = state_gru_dim,
                control_dim          = control_dim if use_control else 0,
                motion_hidden_dim    = motion_hidden_dim,
                per_particle_context = per_particle_context,
            )

        # ── Measurement Model ─────────────────────────────────────────────────
        _meas_hidden_dim = meas_hidden_dim if meas_hidden_dim is not None else hidden_dim // 2
        self.meas_model = LearnedMeasurementModel(
            state_dim            = state_dim,
            obs_dim              = _sensor_obs_dim,
            hidden_dim           = _meas_hidden_dim,
            context_dim          = state_gru_dim if meas_use_context else 0,
            per_particle_context = per_particle_context,
            num_layers           = meas_model_layers,
            use_layernorm        = meas_model_layernorm,
        )

        # ── ODE Flow ──────────────────────────────────────────────────────────
        self.measurement_flow = PFFODEFunc(
            state_dim     = state_dim + aug_dim,
            obs_dim       = obs_dim,
            hidden_dim    = hidden_dim,
            deep_gelu_ode = deep_gelu_ode,
            ode_depth     = ode_depth,
            ode_width     = ode_width,
            ode_residual  = ode_residual,
            uncertainty_dim = _uncertainty_dim,
        )

        # ── λ schedule ────────────────────────────────────────────────────────
        N = integration_steps
        if lam_schedule == 'uniform':
            lam_steps = [1.0 / N] * N
        else:
            b  = 2.0
            s0 = (b - 1) / (b**N - 1)
            lam_steps = [s0 * (b**k) for k in range(N)]
        self.register_buffer('lam_steps',
                             torch.tensor(lam_steps, dtype=torch.float32))

        # Initialize the ODE solver once; integration path is selected by flags.
        self.measurement_solver = build_measurement_ode_solver(
            integration_type = self.integration_type,
            adaptive_steps   = self.adaptive_steps,
            adjoint          = self.adjoint,
            rtol             = inference_rtol,
            atol             = inference_atol,
        )

        if self.loss_type == 'gaussian_nll':
            # Heteroscedastic-NLL variance head (Lakshminarayanan et al. 2017,
            # Section 2.2.1 / Nix & Weigend 1994): predicts log-variance for
            # the FINAL, measurement-corrected particle (not the pre-update
            # prior). Deliberately NOT conditioned on state_context (h_{t-1})
            # -- that tensor is shared with, and dominated by, the
            # mean-estimation path (physics predictor / measurement model),
            # so reading it here would reintroduce exactly the capacity/
            # gradient competition uncertainty_rnn below exists to avoid.
            # (In the current ssm_style_predict=True configuration this also
            # costs no information: meas_use_context is force-overridden to
            # False at construction time -- see the "h must see only f's
            # output" branch above -- so _meas_predict already ignores
            # state_context when forming `innovation`; the position/geometry
            # signal that matters for heteroscedasticity still reaches
            # var_head via innovation's dependence on `prediction`. This
            # would need revisiting if ever trained with meas_use_context=True
            # and ssm_style_predict=False, where context does carry
            # information `prediction` alone doesn't.)
            #
            # Dedicated uncertainty-tracking GRUCell: a second, narrow
            # recurrent channel driven only by the innovation each step --
            # var_head's only cross-step memory is its own hidden state and
            # the explicit prev_log_var feedback below, not state_context.
            self.uncertainty_gru_dim = uncertainty_gru_dim
            self.uncertainty_rnn_lstm = uncertainty_rnn_lstm
            self.uncertainty_rnn = (
                nn.LSTMCell(self.sensor_obs_dim, uncertainty_gru_dim) if uncertainty_rnn_lstm
                else nn.GRUCell(self.sensor_obs_dim, uncertainty_gru_dim))
            # var_head input: innovation [S] + previous-step log-variance
            # feedback [D] (explicit scalar propagation channel) +
            # uncertainty_rnn hidden state [U] (dedicated recurrent
            # propagation channel) -- the two feedback mechanisms are
            # complementary, not exclusive.
            #
            # A single Linear here can only blend these inputs along one
            # fixed direction -- training data showed mean_pred_var and
            # pred_var_time_std both shrinking monotonically for the full
            # 300 epochs while uncertainty_h_std plateaued early and stayed
            # rich, i.e. the linear readout was learning to discard growing
            # upstream structure rather than use it. One hidden layer gives
            # it room to combine the three inputs nonlinearly instead of
            # being forced into a single fixed linear combination.
            # Output is state_dim (per-dim log-variance) + 1 (unnormalised
            # mixture logit) -- num_particles>1 particles are scored as a
            # K-component Gaussian mixture (see _gaussian_nll_step), each
            # particle independently predicting its own variance AND weight
            # from the same [innovation, prev_log_var, uncertainty_h] inputs
            # already used for N=1. At N=1 the logit is a no-op (softmax of
            # a single value is always 1), so this exactly reduces to the
            # original single-Gaussian NLL -- verified numerically, see
            # session history.
            var_head_in_dim = self.sensor_obs_dim + state_dim + uncertainty_gru_dim
            self.var_head = nn.Sequential(
                nn.Linear(var_head_in_dim, 32),
                nn.SiLU(),
                nn.Linear(32, state_dim + 1),
            )

            # Prior-belief NLL head -- see __init__'s "Prior-belief NLL"
            # docstring section. Same [prev_log_var, uncertainty_h] inputs
            # var_head reads, minus innovation (no measurement yet) and
            # minus state_context (same exclusion reason as var_head).
            self.prior_nll_weight = prior_nll_weight
            prior_var_head_in_dim = state_dim + uncertainty_gru_dim
            self.prior_var_head = nn.Sequential(
                nn.Linear(prior_var_head_in_dim, 32),
                nn.SiLU(),
                nn.Linear(32, state_dim + 1),
            )

        self._init_weights()

    # ── Checkpoint backward-compat ───────────────────────────────────────────
    def on_load_checkpoint(self, checkpoint):
        """
        Checkpoints saved before the ode_residual refactor of PFFODEFunc
        (deep_gelu_ode=True) stored the deep stack as a flat
        `measurement_flow.net.<i>` Sequential instead of
        in_proj/hidden_blocks/out_proj. Those old checkpoints always had
        ode_residual=False, so the computation is identical — only the
        attribute names changed. Remap the keys so old checkpoints still load.

        Also drops measurement_flow.output_scale / loss_dim_weights, an
        earlier (learnable, independent) attempt at equation scaling that
        predates the current fixed/shared correction_scale buffers — no
        corresponding parameter exists in this model anymore.

        Checkpoints saved before correction_scale existed (including ones
        saved with equation_scaling already removed in between) lack the
        `correction_scale` / `measurement_flow.correction_scale` buffer
        keys entirely. Fill them in with all-ones (the correct no-op
        default) so old checkpoints still load under strict=True.

        Checkpoints saved before the `net` Sequential refactor of the
        non-deep_gelu_ode PFFODEFunc stored its 3-layer MLP as separate
        `measurement_flow.fc1/fc2/fc3` Linear submodules instead of a single
        `net` Sequential. Remap fc1->net.0, fc2->net.2, fc3->net.4 (the
        Linear layers at those Sequential indices; net.1/3 are SiLU, no
        parameters) so those checkpoints still load.

        process_noise_std/process_noise_raw was removed entirely (no process-
        noise injection of any kind, fixed or learned -- see session
        history). Checkpoints saved before this removal have a
        `process_noise_raw` key with no corresponding buffer in this model
        anymore -- drop it so those checkpoints still load under strict=True.

        Checkpoints saved before particle_init_diversity_std was introduced
        lack the `particle_init_diversity_std` buffer key entirely. Fill it
        in with this model's own init value (particle_init_diversity_std,
        default 0.0) so old checkpoints still load.
        """
        state_dict = checkpoint['state_dict']
        for obsolete_key in ('measurement_flow.output_scale', 'loss_dim_weights',
                              'process_noise_raw'):
            state_dict.pop(obsolete_key, None)

        if 'correction_scale' not in state_dict:
            state_dict['correction_scale'] = self.correction_scale.clone()
        if 'measurement_flow.correction_scale' not in state_dict:
            state_dict['measurement_flow.correction_scale'] = \
                self.measurement_flow.correction_scale.clone()
        if 'particle_init_diversity_std' not in state_dict:
            state_dict['particle_init_diversity_std'] = \
                self.particle_init_diversity_std.detach().clone()

        if self.hparams.deep_gelu_ode:
            old_prefix = 'measurement_flow.net.'
            if any(k.startswith(old_prefix) for k in state_dict):
                ode_depth = self.hparams.ode_depth
                last_idx  = 2 * ode_depth
                remapped  = {}
                for k, v in state_dict.items():
                    if not k.startswith(old_prefix):
                        remapped[k] = v
                        continue
                    idx_str, suffix = k[len(old_prefix):].split('.', 1)
                    idx = int(idx_str)
                    if idx == 0:
                        new_key = f'measurement_flow.in_proj.{suffix}'
                    elif idx == last_idx:
                        new_key = f'measurement_flow.out_proj.{suffix}'
                    else:
                        block_i = idx // 2 - 1
                        new_key = f'measurement_flow.hidden_blocks.{block_i}.0.{suffix}'
                    remapped[new_key] = v
                checkpoint['state_dict'] = remapped
        else:
            fc_to_net_idx = {'fc1': 0, 'fc2': 2, 'fc3': 4}
            if any(k.startswith('measurement_flow.fc1.') for k in state_dict):
                remapped = {}
                for k, v in state_dict.items():
                    moved = False
                    for fc_name, net_idx in fc_to_net_idx.items():
                        old_prefix = f'measurement_flow.{fc_name}.'
                        if k.startswith(old_prefix):
                            suffix = k[len(old_prefix):]
                            remapped[f'measurement_flow.net.{net_idx}.{suffix}'] = v
                            moved = True
                            break
                    if not moved:
                        remapped[k] = v
                checkpoint['state_dict'] = remapped

    # ── Registration helpers ───────────────────────────────────────────────────
    def set_equation_scaling(self, correction_scale: torch.Tensor):
        """
        Register the fixed, data-derived Eq. 13/14 scale (equation_scaling=
        True configs). `correction_scale` is [state_dim], computed from
        consecutive-timestep deltas of the normalised training trajectories
        (FileTrackingDataModule.correction_scale). The same value multiplies
        the ODE flow's output (Eq. 13) and divides predictions/targets in
        calculate_loss (Eq. 14).

        Augmented channels (aug_dim>0) have no corresponding data statistic,
        so they are padded with ones (no scaling) when sizing the flow's
        buffer, which spans state_dim + aug_dim.
        """
        correction_scale = correction_scale.to(self.correction_scale)
        self.correction_scale.copy_(correction_scale)

        flow_scale = self.measurement_flow.correction_scale
        flow_scale.copy_(torch.cat([
            correction_scale.to(flow_scale),
            torch.ones(self.aug_dim, dtype=flow_scale.dtype, device=flow_scale.device),
        ]))

    # ── Weight init ───────────────────────────────────────────────────────────
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0,
                                         mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.GRUCell):
                nn.init.kaiming_uniform_(module.weight_ih, nonlinearity='relu')
                nn.init.orthogonal_(module.weight_hh)
                nn.init.zeros_(module.bias_ih)
                nn.init.zeros_(module.bias_hh)
            elif isinstance(module, nn.LSTMCell):
                # Gate order (PyTorch convention): [input, forget, cell, output],
                # each a hidden_size-wide chunk of the 4*hidden_size weight/bias.
                nn.init.kaiming_uniform_(module.weight_ih, nonlinearity='relu')
                nn.init.orthogonal_(module.weight_hh)
                nn.init.zeros_(module.bias_ih)
                nn.init.zeros_(module.bias_hh)
                # Forget-gate bias = 1 (bias_hh's forget chunk only, bias_ih's
                # left at 0 -- the two are additive, so this is equivalent to
                # splitting it across both) so the cell defaults to
                # remembering rather than forgetting at initialisation
                # (Jozefowicz, Zaremba & Sutskever, ICML 2015 -- see
                # uncertainty_rnn_lstm's docstring).
                hidden = module.hidden_size
                nn.init.ones_(module.bias_hh[hidden:2 * hidden])
            elif isinstance(module, nn.GRU):
                # per_particle_context=True path: nn.GRU names weights per
                # layer (weight_ih_l0, weight_ih_l1, ...) instead of GRUCell's
                # flat weight_ih/weight_hh.
                for layer in range(module.num_layers):
                    nn.init.kaiming_uniform_(
                        getattr(module, f'weight_ih_l{layer}'), nonlinearity='relu')
                    nn.init.orthogonal_(getattr(module, f'weight_hh_l{layer}'))
                    nn.init.zeros_(getattr(module, f'bias_ih_l{layer}'))
                    nn.init.zeros_(getattr(module, f'bias_hh_l{layer}'))
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # innov_skip and the final ODE-func layer start passive so the filter
        # begins as prior-only
        nn.init.zeros_(self.measurement_flow.innov_skip.weight)
        nn.init.zeros_(self.measurement_flow.out_layer.weight)
        nn.init.zeros_(self.measurement_flow.out_layer.bias)

        # Learned meas_model output starts at zero → innovation ≈ z at init
        nn.init.zeros_(self.meas_model.net[-1].weight)
        nn.init.zeros_(self.meas_model.net[-1].bias)

        # Residual physics predictor starts as identity (x_hat = x_t) — same
        # passive-start convention as innov_skip/out_layer/meas_model above.
        # Only zeroed when residual=True: for residual=False, the net's
        # output IS x_hat directly, so zero-init would make f(x_t,u)=0
        # always — a degenerate, not passive, starting point.
        if self.ssm_style_predict and self.ssm_residual_predict:
            nn.init.zeros_(self.physics_predictor.net[-1].weight)
            nn.init.zeros_(self.physics_predictor.net[-1].bias)

        # state_rnn_per_gate_orthogonal=True -- overwrite state_rnn's
        # weight_hh (already whole-matrix-orthogonal from the generic loop
        # above) with a properly per-gate orthogonal init instead. See
        # state_rnn_per_gate_orthogonal's docstring. uncertainty_rnn is
        # deliberately left on the generic whole-matrix scheme.
        if self.state_rnn_per_gate_orthogonal:
            self._orthogonal_init_per_gate(self.state_rnn)

    def _orthogonal_init_per_gate(self, rnn_module):
        """
        Split a GRU/GRUCell's weight_hh ([3*hidden, hidden], the stacked
        reset/update/candidate gates) into its [hidden, hidden] blocks and
        orthogonally initialise each independently, in place -- unlike
        nn.init.orthogonal_ applied to the whole stacked matrix (the generic
        _init_weights loop's convention), which only makes the stacked
        columns orthonormal as a whole, not each individual gate's own
        hidden-to-hidden transform. Only ever called with self.state_rnn
        (always GRU/GRUCell), see state_rnn_per_gate_orthogonal's docstring.
        """
        def per_gate(weight_hh):
            for chunk in weight_hh.data.chunk(3, dim=0):
                nn.init.orthogonal_(chunk)

        if isinstance(rnn_module, nn.GRUCell):
            per_gate(rnn_module.weight_hh)
        elif isinstance(rnn_module, nn.GRU):
            for layer in range(rnn_module.num_layers):
                per_gate(getattr(rnn_module, f'weight_hh_l{layer}'))

    def on_train_start(self):
        if hasattr(self, '_lr_override') and self._lr_override is not None:
            for pg in self.trainer.optimizers[0].param_groups:
                pg['lr'] = self._lr_override
            print(f"[on_train_start] LR reset to {self._lr_override}")

    # ── Predict step ──────────────────────────────────────────────────────────
    def _predict(self, particles, h_state, control=None, uncertainty_signal=None,
                 raw_state_override=None):
        """
        Deterministic state transition (noise is added in forward).
        ssm_style_predict=True → direct:      x_next = f(readout(h_state) [, ctrl] [, unc])
                                  (particles is NOT consulted — see
                                  SSMStylePhysicsPredictor -- UNLESS
                                  raw_state_override is given, see below)
        otherwise (learned)    → residual:    x_next = x + delta(h_state [, ctrl])
                                  (uncertainty_signal ignored -- DiscretePhysicsPredictor
                                  isn't wired up to accept it; see
                                  uncertainty_conditioning's docstring)

        raw_state_override : [B, N, state_dim] or None — direct_init_predict=
                         True only (see its docstring). When given, used
                         directly as physics_predictor's x_t input instead
                         of context_readout(h_state) -- the t=1 call site is
                         the only caller that ever passes this.
        """
        if self.ssm_style_predict:
            if self.ssm_wide_context_input:
                return self.physics_predictor(h_state, control, uncertainty_signal)
            x_t = raw_state_override if raw_state_override is not None else self.context_readout(h_state)
            return self.physics_predictor(x_t, control, uncertainty_signal)
        return self.physics_predictor(particles, h_state, control)

    # ── Measurement prediction ────────────────────────────────────────────────
    def _meas_predict(self, x, state_context):
        """Apply the learned measurement model h_θ(x, state_context) to particles."""
        return self.meas_model(x, state_context)

    # ── uncertainty_rnn step / readout (GRUCell or LSTMCell) ──────────────────
    def _uncertainty_rnn_step(self, innovation_flat, uncertainty_h_flat):
        """
        innovation_flat, uncertainty_h_flat : [B*N, sensor_obs_dim],
            [B*N, uncertainty_gru_dim] (GRU) or [B*N, 2*uncertainty_gru_dim]
            (LSTM, packed [h ‖ c]) -- see uncertainty_rnn_lstm's docstring.
        Returns the new packed state, same shape as uncertainty_h_flat.
        """
        if self.uncertainty_rnn_lstm:
            h, c = uncertainty_h_flat.split(self.uncertainty_gru_dim, dim=-1)
            new_h, new_c = self.uncertainty_rnn(innovation_flat, (h, c))
            return torch.cat([new_h, new_c], dim=-1)
        return self.uncertainty_rnn(innovation_flat, uncertainty_h_flat)

    def _uncertainty_h_readout(self, uncertainty_h):
        """
        The [..., uncertainty_gru_dim] slice actually exposed to var_head /
        uncertainty_signal -- for LSTM this is just the h half of the packed
        [h ‖ c] state (c is internal cell-state bookkeeping only, never
        surfaced outside _uncertainty_rnn_step). No-op for GRU.
        """
        if self.uncertainty_rnn_lstm:
            return uncertainty_h[..., :self.uncertainty_gru_dim]
        return uncertainty_h

    # ── Heteroscedastic Gaussian NLL (loss_type='gaussian_nll' only) ──────────
    def _gaussian_nll_step(self, prediction, state_context, current_meas, true_state,
                           prev_log_var, uncertainty_h):
        """
        Heteroscedastic Gaussian-MIXTURE NLL (Lakshminarayanan et al. 2017,
        Eq. 1; Nix & Weigend 1994, extended to a K-component mixture via
        log-sum-exp, matching the ffjord_nll mixture convention this
        originally replaced -- see also model_experimental.py's
        mixture_head/mixture_nll_loss for the same idea applied to an
        explicit num_modes rather than num_particles directly), scored
        against the FINAL, measurement-corrected particles (`prediction` —
        the same quantity calculate_loss scores for every other loss_type),
        not the pre-update prior. Each of the num_particles particles IS one
        mixture component -- its mean already exists (that's what a
        particle is); var_head additionally predicts that component's own
        variance AND an unnormalised mixture weight (logit) from the same
        inputs used at num_particles=1. Exactly reduces to the original
        single-Gaussian NLL when num_particles=1: log_softmax of a single
        logit is identically 0 regardless of the logit's value (softmax of
        one element is always 1), so the mixture term vanishes and the
        extra logit output is simply untrained dead weight in that regime
        -- verified numerically, see session history.

        Unlike per-particle MSE's bias^2 + Var(particles) decomposition
        (which rewards collapsing predictive spread to zero), this loss has
        the model directly predict its own variance and is scored via a
        strictly proper scoring rule: too-small sigma^2 is punished by the
        blown-up residual term, too-large sigma^2 is punished by the
        log-variance term — so there is no in-loss incentive to shrink
        variance toward zero. The mixture weight adds a genuinely separate
        escape valve at num_particles>1: a poorly-tracking particle can be
        down-weighted (logit -> -inf) instead of being forced to inflate
        its own variance to explain a large residual.

        var_head is conditioned on the POST-measurement innovation residual
        (current_meas - h(prediction, state_context)), not the pre-update
        prior — it must see the same information available at the point
        it's accompanying, since `prediction` is post-measurement-
        correction. It is deliberately NOT conditioned on state_context
        directly (see __init__ comment by self.var_head's construction) --
        only innovation plus two complementary cross-step feedback
        channels, since neither alone forces var_head to track uncertainty
        over time (otherwise it just recomputes from scratch every step
        with no memory of the past):
          - prev_log_var  — the previous step's realised log(var) fed back
                             directly, an explicit scalar propagation signal.
          - uncertainty_h — the hidden state of a dedicated GRUCell driven
                             only by the innovation each step, decoupled
                             from state_context (which the mean-estimation
                             path also conditions on and dominates) -- a
                             richer, purpose-built recurrent memory for
                             uncertainty specifically.

        prediction    : [B, N, D]  final particle states (N = number of
                         mixture components).
        state_context : [B, N, C]  still needed for _meas_predict (and the
                         non-ssm_style_predict/meas_use_context=True path),
                         just no longer fed into var_head directly.
        current_meas  : [B, obs_dim]
        true_state    : [B, D]
        prev_log_var  : [B, N, D]  previous step's log(var), or the
                         log(min_variance) floor at t=1 (no prior step yet).
        uncertainty_h : [B, N, U]  previous step's uncertainty_rnn hidden
                         state, or zeros at t=1 (no prior step yet).
        Returns: nll [B], log_var [B, N, D] (-> prev_log_var next call),
                 new_uncertainty_h [B, N, U] (-> uncertainty_h next call),
                 mixture_weights [B, N] (normalised, sums to 1 over N),
                 sq_resid [B, N, D] (squared, correction_scale-normalised
                 residual -- same quantity var is scored against, i.e. the
                 numerator of log_gauss's data term; exposed purely as a
                 diagnostic for calibration_ratio, see _unroll).
        """
        N          = prediction.shape[1]
        B          = prediction.shape[0]
        z_sensor   = current_meas[..., :self.sensor_obs_dim]
        z_expanded = z_sensor.unsqueeze(1).expand(-1, N, -1)
        innovation = z_expanded - self._meas_predict(prediction, state_context)

        raw = self.var_head(
            torch.cat([innovation, prev_log_var, self._uncertainty_h_readout(uncertainty_h)],
                      dim=-1))
        raw_log_var = raw[..., :self.state_dim]                    # [B, N, D]
        logit       = raw[..., self.state_dim]                     # [B, N]
        var         = torch.nn.functional.softplus(raw_log_var) + self.min_variance
        logpi       = torch.log_softmax(logit, dim=1)               # mixture weights, normalised over particles

        pred_scaled = prediction / self.correction_scale
        true_scaled = (true_state / self.correction_scale).unsqueeze(1).expand_as(prediction)
        sq_resid    = (true_scaled - pred_scaled) ** 2               # [B, N, D]

        log_gauss = -0.5 * (torch.log(var) + sq_resid / var).sum(dim=-1)  # [B, N]
        nll       = -torch.logsumexp(logpi + log_gauss, dim=1)      # [B]

        uncertainty_state_dim = uncertainty_h.shape[-1]
        new_uncertainty_h = self._uncertainty_rnn_step(
            innovation.reshape(B * N, self.sensor_obs_dim),
            uncertainty_h.reshape(B * N, uncertainty_state_dim),
        ).reshape(B, N, uncertainty_state_dim)

        # Normalised mixture weights [B, N] -- NOT used by the NLL itself
        # (that combines per-component log-likelihoods via logsumexp, not
        # a weighted-mean residual, see docstring above); returned purely
        # so callers can report E[x] = sum_k pi_k * x_k as a point estimate.
        mixture_weights = logpi.exp()

        return nll, torch.log(var), new_uncertainty_h, mixture_weights, sq_resid

    # ── Heteroscedastic Gaussian NLL for the PRE-measurement prior ────────────
    def _prior_gaussian_nll_step(self, prior_prediction, true_state,
                                 prev_log_var, uncertainty_h):
        """
        Same K-component Gaussian-mixture NLL as _gaussian_nll_step (see its
        docstring for the general form and the num_particles=1 reduction),
        but scored against particles_prior (x_prior, pre-measurement) via
        prior_var_head instead of var_head -- see __init__'s "Prior-belief
        NLL" docstring section for why this head is conditioned the way it
        is and why it's a separate, uncoupled head rather than sharing
        var_head's weights.

        Reads prev_log_var/uncertainty_h as a snapshot of "what the filter
        knew about its uncertainty one step ago" -- it does NOT produce a
        new recurrent state of its own; the actual state update (this
        step's real prev_log_var/uncertainty_h for the NEXT call) still
        happens in _gaussian_nll_step immediately after, in _unroll's loop.
        Call this one first, with the same (not-yet-updated) prev_log_var/
        uncertainty_h _gaussian_nll_step is about to consume.

        prior_prediction : [B, N, D]  particles_prior (x_prior).
        true_state        : [B, D]
        prev_log_var       : [B, N, D]
        uncertainty_h      : [B, N, U]
        Returns: nll [B].
        """
        raw = self.prior_var_head(
            torch.cat([prev_log_var, self._uncertainty_h_readout(uncertainty_h)],
                      dim=-1))
        raw_log_var = raw[..., :self.state_dim]
        logit       = raw[..., self.state_dim]
        var         = torch.nn.functional.softplus(raw_log_var) + self.min_variance
        logpi       = torch.log_softmax(logit, dim=1)

        pred_scaled = prior_prediction / self.correction_scale
        true_scaled = (true_state / self.correction_scale).unsqueeze(1).expand_as(prior_prediction)

        log_gauss = -0.5 * (torch.log(var)
                            + (true_scaled - pred_scaled) ** 2 / var).sum(dim=-1)
        nll       = -torch.logsumexp(logpi + log_gauss, dim=1)

        return nll

    # ── State GRU step (unifies shared-context and per-particle-context) ─────
    def _gru_init(self, init_value, init_particles, batch_size):
        """
        First state_rnn call of a rollout (t=0).

        init_value     : [B, state_dim] — the (possibly noisy) seed value fed
                          to every particle's GRU input.
        init_particles : [B, N, state_dim] — the initialised particle
                          ensemble.

        Returns (context, recurrent_state):
          context         — what physics_predictor/meas_model consume this
                             step: [B, state_gru_dim] or [B, N, state_gru_dim].
          recurrent_state — what to pass into the next _gru_step call.
        """
        N = self.num_particles
        # Stochastic h0 only during actual training -- self.training is False
        # for validation_step (Lightning runs it in eval mode) and for any
        # external eval script that calls model.eval(), giving reproducible
        # rollouts there instead of a fresh random h0 (and therefore a
        # different t=1 prediction) on every invocation. Irrelevant when
        # learned_init_encoder=True -- h0 comes from init_encoder(x_0)
        # instead, deterministic given the input either way (verified this
        # noise choice made literally no difference to t=1 error when it was
        # the only source of h0 -- see learned_init_encoder's docstring).
        h0_scale = 0.1 if self.training else 0.0
        if self.per_particle_context:
            # Each particle feeds its OWN noisy init value (init_particles,
            # already [B, N, state_dim] with independent per-particle noise)
            # rather than a single shared init_value broadcast to every
            # particle — otherwise cross-particle diversity in context at
            # t=0 would come only from each particle's random h0 draw, not
            # from any real positional information.
            h_in = init_particles
            gru_in_dim = h_in.shape[-1]
            if self.learned_init_encoder:
                # init_encoder maps each particle's own init value to its own
                # h0 -- [B, N, state_dim] -> [B, N, state_gru_dim] -> folded
                # to [B*N, state_gru_dim] and broadcast across GRU layers
                # (no principled way to differentiate per-layer initial
                # state from a single input encoding, so all layers start
                # from the same encoding).
                h0_single = self.init_encoder(init_particles).reshape(
                    batch_size * N, self.hparams.state_gru_dim)
                h0 = h0_single.unsqueeze(0).expand(
                    self.gru_num_layers, -1, -1).contiguous()
            else:
                h0 = torch.randn(self.gru_num_layers, batch_size * N,
                                 self.hparams.state_gru_dim, device=self.device) * h0_scale
            out, h_state = self.state_rnn(
                h_in.reshape(batch_size * N, 1, gru_in_dim), h0)
            context = out.reshape(batch_size, N, self.hparams.state_gru_dim)
            return context, h_state
        else:
            if self.learned_init_encoder:
                h0 = self.init_encoder(init_value)
            else:
                h0 = torch.randn(batch_size, self.hparams.state_gru_dim,
                                 device=self.device) * h0_scale
            h_state = self.state_rnn(init_value, h0)
            return h_state, h_state

    def _gru_step(self, particle_value, particles_full, h_state, batch_size,
                  use_detach=False):
        """
        Subsequent state_rnn calls (t>0).

        particle_value : [B, state_dim] (mean estimate) when not
                          per_particle_context, else [B, N, state_dim] (the
                          full particle set — each particle feeds its own
                          GRU input).
        particles_full : [B, N, state_dim] — unused (kept for call-site
                          symmetry with _gru_init).
        h_state        : the recurrent_state returned by the previous
                          _gru_init/_gru_step call.

        Returns (context, recurrent_state) — see _gru_init.
        """
        N = self.num_particles
        h_in = particle_value.detach() if use_detach else particle_value
        if self.per_particle_context:
            gru_in_dim = h_in.shape[-1]
            out, h_state = self.state_rnn(
                h_in.reshape(batch_size * N, 1, gru_in_dim), h_state)
            context = out.reshape(batch_size, N, self.hparams.state_gru_dim)
            return context, h_state
        else:
            h_state = self.state_rnn(h_in, h_state)
            return h_state, h_state

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, particles_prev, state_context, current_meas,
                control=None, uncertainty_signal=None, raw_state_override=None):
        """
        particles_prev : [B, N, state_dim]
        state_context  : [B, state_gru_dim]
        current_meas   : [B, obs_dim]
        control        : [B, control_dim]    — use_control=True only
        uncertainty_signal : [B, N, state_dim + uncertainty_gru_dim] —
                         uncertainty_conditioning=True only (required, not
                         optional, when that flag is set -- see its
                         docstring). Last step's [prev_log_var ‖
                         uncertainty_h] -- passed to physics_predictor as-is
                         (though it structurally ignores it, see __init__).
                         measurement_flow instead receives a WIDENED signal
                         built below (see meas_uncertainty_signal), with
                         this step's prior_var_head output appended.
        raw_state_override : see _predict -- direct_init_predict=True only.
        """
        # STEP 1: PREDICT (deterministic transition -- no process noise; see
        # __init__'s "Process noise" docstring section for why).
        x_prior = self._predict(particles_prev, state_context, control, uncertainty_signal,
                                 raw_state_override)

        if self.skip_update:
            return x_prior[..., :self.state_dim], x_prior[..., :self.state_dim]

        # Prior-belief uncertainty, computed fresh from this step's
        # prev_log_var/uncertainty_h (the SAME inputs _prior_gaussian_nll_step
        # reads in _unroll for the loss -- recomputed here, not threaded
        # through, see that method's docstring) -- see
        # uncertainty_conditioning's docstring for why this is fed forward
        # into measurement_flow specifically. Only the log-variance channels
        # are forwarded, not the mixture logit (an N-particle weighting
        # concept, not a per-step uncertainty-magnitude signal, and dead
        # weight at num_particles=1 same as var_head's own logit).
        meas_uncertainty_signal = uncertainty_signal
        if self.uncertainty_conditioning:
            raw_prior = self.prior_var_head(uncertainty_signal)
            prior_var = (torch.nn.functional.softplus(raw_prior[..., :self.state_dim])
                        + self.min_variance)
            meas_uncertainty_signal = torch.cat(
                [uncertainty_signal, torch.log(prior_var)], dim=-1)

        # STEP 2: ODE UPDATE  (λ ∈ [0,1] homotopy)
        x = x_prior
        if self.aug_dim > 0:
            # Augmented Neural ODE (Dupont et al.): pad the state with extra
            # zero-initialised channels before the homotopy solve, giving the
            # ODE flow's vector field more room to represent trajectories that
            # would otherwise need to cross in state_dim-only space. The
            # measurement model never sees these channels; they are dropped
            # after integration.
            aug_zeros = torch.zeros(*x.shape[:-1], self.aug_dim,
                                    dtype=x.dtype, device=x.device)
            x = torch.cat([x, aug_zeros], dim=-1)

        z_sensor   = current_meas[..., :self.sensor_obs_dim]
        z_expanded = z_sensor.unsqueeze(1).expand(-1, x.shape[1], -1)

        if self.ssm_style_predict and not self.ssm_continuous_innovation:
            # Frozen innovation, evaluated once at x_prior (post-noise, the
            # actual entry point of the λ-transport) — NOT the continuous
            # recompute-at-x(λ) default below. No torch.no_grad() here:
            # h is learned, so its parameters need gradients from this path.
            z_pred_0     = self._meas_predict(x[..., :self.state_dim], state_context)
            innovation_0 = z_expanded - z_pred_0

            def wrapper_flow(lam, x_):
                return self.measurement_flow(lam, x_, innovation_0, meas_uncertainty_signal)
        else:
            # Continuous (recompute-at-x(λ)) innovation — the default
            # learned-measurement branch's convention, also used by
            # ssm_style_predict when ssm_continuous_innovation=True.
            def wrapper_flow(lam, x_):
                innov = z_expanded - self._meas_predict(x_[..., :self.state_dim], state_context)
                return self.measurement_flow(lam, x_, innov, meas_uncertainty_signal)

        # adjoint_params only actually consulted when adaptive_steps=True
        # AND adjoint=True (see solver.py's build_measurement_ode_solver
        # docstring) -- computed lazily to avoid the tuple(...) materialise
        # cost on every call in the (default, common) non-adjoint path.
        adjoint_params = tuple(self.parameters()) if self.adjoint else None
        x = self.measurement_solver(wrapper_flow, x, self.lam_steps, adjoint_params=adjoint_params)

        return x[..., :self.state_dim], x_prior[..., :self.state_dim]

    # ── Loss ──────────────────────────────────────────────────────────────────
    def calculate_loss(self, predictions, targets, mask=None):
        # Eq. 14: divide by the same scale used in Eq. 13 before computing
        # the elementwise loss. No-op when correction_scale is all-ones.
        predictions = predictions / self.correction_scale
        targets     = targets     / self.correction_scale

        if self.loss_type == 'l1':
            elementwise = torch.abs(predictions - targets)
        elif self.loss_type in ['l2', 'mse']:
            elementwise = (predictions - targets) ** 2
        elif self.loss_type == 'huber':
            elementwise = torch.nn.functional.huber_loss(
                predictions, targets, reduction='none', delta=self.huber_delta
            )

        elementwise = elementwise.mean(dim=-1)

        if mask is not None:
            active = mask.float()
            return (elementwise * active).sum() / active.sum().clamp(min=1)
        return elementwise.mean()

    # ── Unroll ────────────────────────────────────────────────────────────────
    def _unroll(self, obs_window, true_states, controls=None, use_detach=False):
        seq_len    = true_states.shape[1]
        batch_size = true_states.shape[0]

        particles_curr = true_states[:, 0, :].unsqueeze(1).expand(-1, self.num_particles, -1)
        if self.num_particles > 1:
            # See __init__'s "particle_init_diversity_std" docstring --
            # seeds diversity ONCE, here, not per propagation step.
            particles_curr = particles_curr + (
                torch.randn_like(particles_curr) * self.particle_init_diversity_std)

        state_context, h_state = self._gru_init(
            true_states[:, 0, :], particles_curr, batch_size)

        # Auxiliary init_encoder reconstruction loss -- see init_recon_weight's
        # docstring for the math and motivation. x_0 is broadcast identically
        # across all N particles when particle_init_diversity_std=0 (the
        # default) or num_particles=1, so context_readout's per-particle
        # output should also match it identically; no need to weight by
        # particle diversity the way the state-tracking loss does. When
        # particle_init_diversity_std>0 and num_particles>1, each particle's
        # context instead starts from a perturbed x_0 (particles_curr
        # above) but is still scored against the same true x_0 here --
        # deliberately so: the target is "recover the true t=0 state",
        # not "reconstruct whatever noisy input you were given".
        init_recon_loss = None
        if self.learned_init_encoder and self.init_recon_weight > 0:
            x0_recon = self.context_readout(state_context)
            init_recon_loss = torch.nn.functional.mse_loss(
                x0_recon, true_states[:, 0, :].unsqueeze(1).expand_as(x0_recon))

        predictions_list  = []
        priors_list       = []
        is_gaussian_nll   = self.loss_type == 'gaussian_nll'
        nll_list          = [] if is_gaussian_nll else None
        nll_prior_list    = [] if is_gaussian_nll else None
        # log(var) per timestep, detached -- diagnostic only (see
        # mean_pred_var/pred_var_time_std/uncertainty_h_std logging below),
        # not part of the loss, so no need to retain the autograd graph.
        log_var_list      = [] if is_gaussian_nll else None
        # Squared residual matching var_head's own scoring target each step
        # (see _gaussian_nll_step's sq_resid) -- diagnostic only, feeds
        # calibration_ratio below (mean actual sq_resid / mean reported var;
        # ~1.0 is well-calibrated, >>1 overconfident, <<1 underconfident).
        sq_resid_list     = [] if is_gaussian_nll else None
        # Per-step mixture weights [B, N] -- diagnostic only, feeds
        # mixture_weight_std/mixture_weight_max below (num_particles>1 only;
        # both are exactly 0/1 respectively at N=1, since softmax of a
        # single logit is always 1 -- not useful there, still logged for
        # consistency).
        mixture_weights_list = [] if is_gaussian_nll else None
        # Feedback channel for _gaussian_nll_step's temporal propagation of
        # uncertainty (see var_head docstring) — no previous step exists at
        # t=1, so seed it at the log of the variance floor (the "no
        # information yet" point).
        prev_log_var = (torch.full(
            (batch_size, self.num_particles, self.state_dim),
            math.log(self.min_variance), device=true_states.device)
            if is_gaussian_nll else None)
        # Dedicated uncertainty-tracking GRUCell/LSTMCell's hidden state (see
        # var_head/uncertainty_rnn docstrings) — zeros at t=1, no prior
        # innovation to have driven it yet. 2*uncertainty_gru_dim when
        # uncertainty_rnn_lstm=True (packed [h ‖ c], see its docstring).
        uncertainty_h = (torch.zeros(
            batch_size, self.num_particles,
            self.uncertainty_gru_dim * (2 if self.uncertainty_rnn_lstm else 1),
            device=true_states.device)
            if is_gaussian_nll else None)
        # Particle spread immediately before (prior, λ=0) and after (pred,
        # λ=1) the ODE solve, collected every timestep and averaged over the
        # sequence — not the per-λ-step spread *during* the solve.
        particle_prior_std_list = []
        particle_std_list       = []
        # Mean pairwise Euclidean distance between particles -- more directly
        # interpretable than std at small num_particles (e.g. exactly
        # ||particle_0 - particle_1|| at N=2). Undefined (no pairs) at N=1,
        # only collected when num_particles>1.
        particle_prior_dist_list = [] if self.num_particles > 1 else None
        particle_dist_list       = [] if self.num_particles > 1 else None

        for t in range(1, seq_len):
            current_meas    = obs_window[:, t, :]
            current_control = controls[:, t-1, :] if controls is not None else None

            # Last step's uncertainty state (or the t=1 floor/zero init
            # above) -- built BEFORE this step's _gaussian_nll_step call
            # overwrites prev_log_var/uncertainty_h, so this really is
            # "what the filter knew about its own uncertainty one step ago."
            # NOT detached (deliberately, for now -- see session history):
            # the state-tracking loss now has a live gradient path back into
            # var_head's AND prior_var_head's weights (loss -> forward()'s
            # meas_uncertainty_signal -> measurement_flow's uncertainty
            # branch, and loss -> prior_var_head directly via
            # meas_uncertainty_signal's prior_log_var channel -- see
            # forward()). An earlier, DETACHED version of the var_head-only
            # half of this same mechanism was found to collapse predicted
            # variance to the min_variance floor within a few epochs
            # (pred_var_time_std declining monotonically, min_pred_var
            # pinned at the floor from epoch 0) when NOT detached, because
            # the tracking loss could reach var_head's weights through
            # exactly this kind of path -- detaching was the fix. Removing
            # that fix here is a deliberate experiment, not an oversight;
            # watch mean_pred_var/min_pred_var for the same signature.
            uncertainty_signal = (
                torch.cat([prev_log_var,
                           self._uncertainty_h_readout(uncertainty_h)], dim=-1)
                if self.uncertainty_conditioning else None)

            # direct_init_predict: only the very first call (t=1) bypasses
            # context_readout(state_context) in favour of the raw x_0 --
            # see direct_init_predict's docstring. Exactly x_0 only when
            # num_particles=1 or particle_init_diversity_std=0; otherwise
            # this is x_0 plus each particle's own init-diversity
            # perturbation (see particles_curr above), same value each
            # particle already uses as its physics_predictor input via the
            # ordinary (non-override) path at every other t, so this stays
            # consistent with that, not with a literal ground-truth input.
            # From t=2 onward, state_context is always built by the
            # ordinary _gru_step below from a real corrected posterior,
            # same as without this flag.
            raw_state_override = (
                particles_curr if (self.direct_init_predict and t == 1) else None)

            particles_pred, particles_prior = self(
                particles_prev = particles_curr,
                state_context  = state_context,
                current_meas   = current_meas,
                control        = current_control,
                uncertainty_signal = uncertainty_signal,
                raw_state_override = raw_state_override,
            )

            prior_mean = particles_prior.mean(dim=1)
            priors_list.append(prior_mean)

            if is_gaussian_nll:
                # Prior NLL first -- reads prev_log_var/uncertainty_h as the
                # (not-yet-updated) snapshot from last step; _gaussian_nll_step
                # right after is what actually advances them. See
                # _prior_gaussian_nll_step's docstring.
                nll_prior_t = self._prior_gaussian_nll_step(
                    particles_prior, true_states[:, t, :], prev_log_var, uncertainty_h)
                nll_prior_list.append(nll_prior_t)

                nll_t, prev_log_var, uncertainty_h, mixture_weights, sq_resid = \
                    self._gaussian_nll_step(
                        particles_pred, state_context, current_meas, true_states[:, t, :],
                        prev_log_var, uncertainty_h)
                nll_list.append(nll_t)
                # Mixture-weighted mean E[x] = sum_k pi_k * x_k -- the actual
                # first moment of the fitted mixture, not a plain particle
                # average (which would ignore what var_head learned about
                # which particles to trust). Exactly equals the plain mean
                # at num_particles=1 (single weight is always 1).
                mean_estimate = (mixture_weights.unsqueeze(-1) * particles_pred).sum(dim=1)
            else:
                mean_estimate = particles_pred.mean(dim=1)

            predictions_list.append(mean_estimate)

            with torch.no_grad():
                particle_prior_std_list.append(particles_prior.std(dim=1).mean())
                particle_std_list.append(particles_pred.std(dim=1).mean())
                if self.num_particles > 1:
                    N = self.num_particles
                    n_pairs = N * (N - 1)
                    particle_prior_dist_list.append(
                        torch.cdist(particles_prior, particles_prior).sum(dim=(-2, -1))
                        .div(n_pairs).mean())
                    particle_dist_list.append(
                        torch.cdist(particles_pred, particles_pred).sum(dim=(-2, -1))
                        .div(n_pairs).mean())
                if is_gaussian_nll:
                    log_var_list.append(prev_log_var.detach())
                    sq_resid_list.append(sq_resid.detach())
                    mixture_weights_list.append(mixture_weights.detach())

            particle_value = particles_pred if self.per_particle_context else mean_estimate
            state_context, h_state = self._gru_step(
                particle_value, particles_pred, h_state, batch_size, use_detach=use_detach)
            particles_curr = particles_pred.detach() if use_detach else particles_pred

        with torch.no_grad():
            correction         = (particles_pred - particles_prior).abs().mean()
            h_state_std        = state_context.std(dim=-1).mean()
            # Mean over the whole sequence, not just the last timestep.
            particle_std       = torch.stack(particle_std_list).mean()
            particle_prior_std = torch.stack(particle_prior_std_list).mean()
            if self.num_particles > 1:
                particle_dist       = torch.stack(particle_dist_list).mean()
                particle_prior_dist = torch.stack(particle_prior_dist_list).mean()
            if is_gaussian_nll:
                var_stack = torch.exp(torch.stack(log_var_list, dim=1))  # [B, T-1, N, D]
                mean_pred_var      = var_stack.mean()
                min_pred_var       = var_stack.min()
                max_pred_var       = var_stack.max()
                # Std across the TIME axis (dim=1), averaged over batch/
                # particle/state-dim -- directly answers "does the model's
                # own predicted variance actually fluctuate over the course
                # of a trajectory, or has it collapsed to a constant?" (the
                # flatness seen in eval-script plots so far). Near-zero here
                # confirms flatness numerically during training itself,
                # without waiting for a checkpoint + separate eval run.
                pred_var_time_std  = var_stack.std(dim=1).mean()
                # Final-step hidden state spread of the dedicated
                # uncertainty-tracking GRUCell/LSTMCell, across its own
                # hidden dim (readout half only, for LSTM) -- near-zero
                # would mean uncertainty_rnn itself has collapsed to a
                # near-constant output, independent of var_head's mapping
                # from it.
                uncertainty_h_std  = self._uncertainty_h_readout(uncertainty_h).std(dim=-1).mean()
                # Calibration check: mean actual squared residual / mean
                # reported variance, both over the same [B, T-1, N, D]
                # population var_head is scored against. ~1.0 means the
                # reported uncertainty matches actual error on average
                # (well-calibrated); >>1 means overconfident (variance too
                # small for the real errors); <<1 means underconfident.
                # Unlike mean_pred_var/min_pred_var alone (which can decline
                # for entirely legitimate reasons, e.g. the model getting
                # genuinely more accurate, or just reflect a worst-case
                # single value rather than a population average), this
                # metric directly answers whether the SCALE of that decline
                # is actually justified by real prediction error -- see
                # session history for a case where reading mean_pred_var/
                # min_pred_var alone gave a misleading impression that a
                # direct calibration check then corrected.
                sq_resid_stack   = torch.stack(sq_resid_list, dim=1)  # [B, T-1, N, D]
                calibration_ratio = sq_resid_stack.mean() / var_stack.mean()
                # Mixture weight spread [B, T-1, N] -- how much var_head is
                # actually differentiating between particles by trust, vs
                # leaving them near-uniform (1/N each). Both are exactly 0
                # (std) / 1/N (max) at num_particles=1 -- not useful there,
                # not logged in that case.
                if self.num_particles > 1:
                    mw_stack = torch.stack(mixture_weights_list, dim=1)  # [B, T-1, N]
                    mixture_weight_std = mw_stack.std(dim=-1).mean()
                    mixture_weight_max = mw_stack.max(dim=-1).values.mean()

        self.log('mean_ode_correction', correction,          prog_bar=True,  on_step=True, on_epoch=True)
        self.log('h_state_std',         h_state_std,         prog_bar=False, on_step=True, on_epoch=True)
        self.log('particle_std',        particle_std,        prog_bar=False, on_step=True, on_epoch=True)
        self.log('particle_prior_std',  particle_prior_std,  prog_bar=False, on_step=True, on_epoch=True)
        if self.num_particles > 1:
            self.log('particle_dist',       particle_dist,       prog_bar=False, on_step=True, on_epoch=True)
            self.log('particle_prior_dist', particle_prior_dist, prog_bar=False, on_step=True, on_epoch=True)
        if is_gaussian_nll:
            self.log('mean_pred_var',      mean_pred_var,      prog_bar=False, on_step=True, on_epoch=True)
            self.log('min_pred_var',       min_pred_var,       prog_bar=False, on_step=True, on_epoch=True)
            self.log('max_pred_var',       max_pred_var,       prog_bar=False, on_step=True, on_epoch=True)
            self.log('pred_var_time_std',  pred_var_time_std,  prog_bar=False, on_step=True, on_epoch=True)
            self.log('uncertainty_h_std',  uncertainty_h_std,  prog_bar=False, on_step=True, on_epoch=True)
            self.log('calibration_ratio',  calibration_ratio,  prog_bar=True,  on_step=True, on_epoch=True)
            if self.num_particles > 1:
                self.log('mixture_weight_std', mixture_weight_std, prog_bar=False, on_step=True, on_epoch=True)
                self.log('mixture_weight_max', mixture_weight_max, prog_bar=False, on_step=True, on_epoch=True)
        loss_extras = {}
        if is_gaussian_nll:
            loss_extras['nll']       = torch.stack(nll_list, dim=1)
            loss_extras['nll_prior'] = torch.stack(nll_prior_list, dim=1)
        if init_recon_loss is not None:
            loss_extras['init_recon'] = init_recon_loss
        return (torch.stack(predictions_list, dim=1),
                torch.stack(priors_list,      dim=1),
                loss_extras or None)

    # ── Batch unpacking ───────────────────────────────────────────────────────
    def _unpack_batch(self, batch):
        if self.hparams.use_control:
            obs_window, true_states, controls, mask = batch
        else:
            obs_window, true_states, mask = batch
            controls = None
        return obs_window, true_states, controls, mask

    @staticmethod
    def _masked_mean(series, active_mask):
        active = active_mask.float()
        return (series * active).sum() / active.sum().clamp(min=1)

    # ── Auxiliary init_encoder loss ─────────────────────────────────────────────
    def _add_init_recon(self, loss, loss_extras, log_prefix):
        # See init_recon_weight's docstring for the math. No-op unless
        # init_recon_weight > 0 (loss_extras won't contain 'init_recon'
        # otherwise -- see _unroll).
        if loss_extras is not None and 'init_recon' in loss_extras:
            init_recon_loss = loss_extras['init_recon']
            self.log(f'{log_prefix}_init_recon_loss', init_recon_loss, prog_bar=False)
            loss = loss + self.init_recon_weight * init_recon_loss
        return loss

    # ── Training ──────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        obs_window, true_states, controls, mask = self._unpack_batch(batch)
        active_mask = mask[:, 1:]

        if batch_idx == 0:
            print(f"[Epoch {self.current_epoch} phase={self.hparams.training_phase}] "
                  f"seq={true_states.shape[1]} batch={true_states.shape[0]}",
                  flush=True)

        all_predictions, all_priors, loss_extras = self._unroll(
            obs_window, true_states, controls, use_detach=False)

        if self.hparams.training_phase == 1:
            # Phase 1: supervise the predict step only (predictor warm-up)
            loss = self.calculate_loss(
                all_priors, true_states[:, 1:, :], mask=active_mask)
            self.log('train_loss',       loss, prog_bar=True)
            self.log('train_prior_loss', loss, prog_bar=False)
        elif self.loss_type == 'gaussian_nll':
            nll_post  = self._masked_mean(loss_extras['nll'],       active_mask)
            nll_prior = self._masked_mean(loss_extras['nll_prior'], active_mask)
            loss = nll_post + self.prior_nll_weight * nll_prior
            self.log('train_loss',       loss,      prog_bar=True)
            self.log('train_nll_post',   nll_post,  prog_bar=False)
            self.log('train_nll_prior',  nll_prior, prog_bar=False)
        else:
            main_loss = self.calculate_loss(
                all_predictions, true_states[:, 1:, :], mask=active_mask)
            aux_loss  = self.calculate_loss(
                all_priors,      true_states[:, 1:, :], mask=active_mask)
            loss = main_loss
            self.log('train_loss',      loss,      prog_bar=True)
            self.log('train_main_loss', main_loss, prog_bar=False)
            self.log('train_aux_loss',  aux_loss,  prog_bar=False)

        loss = self._add_init_recon(loss, loss_extras, 'train')

        # Instability counters -- logged as per-epoch SUMS (reduce_fx='sum')
        # of a per-step 0/1 indicator, so e.g. train_loss_spikes=3 means "3
        # batches this epoch had |loss| > 50", not an average. 50 is a fixed,
        # generous threshold -- comfortably above any value seen in a
        # healthy run so far (roughly -10..25) but well below a genuine
        # blowup (e.g. 434 seen on a real run -- see session history).
        # train_loss_nonfinite (NaN/Inf) is the more severe, unambiguous
        # case -- doesn't depend on the threshold being well-chosen.
        with torch.no_grad():
            is_nonfinite = (~torch.isfinite(loss)).float()
            is_spike     = (loss.detach().abs() > 50).float()
        self.log('train_loss_nonfinite', is_nonfinite, on_step=False, on_epoch=True,
                 reduce_fx='sum', prog_bar=False)
        self.log('train_loss_spikes',    is_spike,     on_step=False, on_epoch=True,
                 reduce_fx='sum', prog_bar=False)

        return loss

    # ── Validation ────────────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        obs_window, true_states, controls, mask = self._unpack_batch(batch)
        all_predictions, _, loss_extras = self._unroll(
            obs_window, true_states, controls, use_detach=False)
        active_mask = mask[:, 1:]
        if self.loss_type == 'gaussian_nll':
            nll_post  = self._masked_mean(loss_extras['nll'],       active_mask)
            nll_prior = self._masked_mean(loss_extras['nll_prior'], active_mask)
            loss = nll_post + self.prior_nll_weight * nll_prior
            self.log('val_nll_post',  nll_post,  prog_bar=False, sync_dist=True)
            self.log('val_nll_prior', nll_prior, prog_bar=False, sync_dist=True)
        else:
            loss = self.calculate_loss(
                all_predictions, true_states[:, 1:, :], mask=active_mask)
        loss = self._add_init_recon(loss, loss_extras, 'val')
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    # ── Full-trajectory rollout RMSE ──────────────────────────────────────────
    def on_validation_epoch_end(self):
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            return

        dataset = self.val_dataset
        # UncontrolledTrajectoryDataset stores .measurements/.true_states;
        # ControlledTrajectoryDataset (use_control=True, e.g. Ackermann)
        # stores .obs/.states instead -- resolve whichever this dataset
        # actually has rather than assuming the uncontrolled names.
        meas_tensor  = dataset.measurements if hasattr(dataset, 'measurements') else dataset.obs
        state_tensor = dataset.true_states  if hasattr(dataset, 'true_states')  else dataset.states
        t_mean  = self.val_t_mean.to(self.device)
        t_std   = self.val_t_std.to(self.device)
        n_traj  = min(5, meas_tensor.shape[0])
        if not hasattr(self, '_val_indices'):
            self._val_indices = torch.randperm(
                meas_tensor.shape[0])[:n_traj]

        all_rmses = []
        pos_rmses = []
        self.eval()
        with torch.no_grad():
            for idx in self._val_indices:
                meas_seq      = meas_tensor[idx].to(self.device)
                true_seq_norm = state_tensor[idx].to(self.device)
                total_steps   = meas_seq.shape[0]

                ctrl_seq = (dataset.controls[idx].to(self.device)
                            if self.hparams.use_control else None)

                x_curr = (true_seq_norm[0].unsqueeze(0).unsqueeze(1)
                          .expand(-1, self.num_particles, -1).float())
                if self.num_particles > 1:
                    # Matches _unroll's t=0 particle_init_diversity_std
                    # injection exactly -- see _unroll and __init__'s
                    # docstring. Without this, num_particles>1 here would
                    # show zero particle diversity even when the model was
                    # actually trained with some, misrepresenting its
                    # real inference-time behaviour.
                    x_curr = x_curr + (torch.randn_like(x_curr)
                                        * self.particle_init_diversity_std)

                state_context, h_state = self._gru_init(
                    true_seq_norm[0].unsqueeze(0), x_curr, batch_size=1)

                predictions = [true_seq_norm[0].cpu()]

                # Same uncertainty-state bookkeeping as _unroll. var_head's
                # own recurrent feedback (prev_log_var/uncertainty_h) is used
                # whenever loss_type='gaussian_nll', independent of
                # uncertainty_conditioning (that flag only gates the SEPARATE
                # feed into physics_predictor/measurement_flow) -- gating
                # this init on uncertainty_conditioning instead would silently
                # skip both the uncertainty-tracking update AND the mixture-
                # weighted point estimate below whenever gaussian_nll is
                # trained with uncertainty_conditioning=False, so this
                # diagnostic wouldn't reflect the model's actual inference-
                # time behaviour.
                if self.loss_type == 'gaussian_nll':
                    prev_log_var = torch.full(
                        (1, self.num_particles, self.state_dim),
                        math.log(self.min_variance), device=self.device)
                    uncertainty_h = torch.zeros(
                        1, self.num_particles,
                        self.uncertainty_gru_dim * (2 if self.uncertainty_rnn_lstm else 1),
                        device=self.device)

                for t in range(1, total_steps):
                    current_meas    = meas_seq[t].unsqueeze(0).float()
                    current_control = (ctrl_seq[t-1].unsqueeze(0).float()
                                       if ctrl_seq is not None else None)

                    # .detach() not needed here (whole loop runs under
                    # torch.no_grad(), see above) -- kept undetached to
                    # match _unroll's construction exactly, not because it
                    # matters functionally in this no-grad context.
                    uncertainty_signal = (
                        torch.cat([prev_log_var,
                                   self._uncertainty_h_readout(uncertainty_h)], dim=-1)
                        if self.uncertainty_conditioning else None)

                    # See _unroll's identical raw_state_override handling.
                    raw_state_override = (
                        x_curr if (self.direct_init_predict and t == 1) else None)

                    particles_pred, _ = self(
                        particles_prev = x_curr,
                        state_context  = state_context,
                        current_meas   = current_meas,
                        control        = current_control,
                        uncertainty_signal = uncertainty_signal,
                        raw_state_override = raw_state_override,
                    )

                    if self.loss_type == 'gaussian_nll':
                        _, prev_log_var, uncertainty_h, mixture_weights, _ = self._gaussian_nll_step(
                            particles_pred, state_context, current_meas,
                            true_seq_norm[t].unsqueeze(0), prev_log_var, uncertainty_h)
                        # See _unroll's identical mixture-weighted-mean comment.
                        mean_est = (mixture_weights.unsqueeze(-1) * particles_pred).sum(dim=1)
                    else:
                        mean_est = particles_pred.mean(dim=1)

                    predictions.append(mean_est.squeeze(0).cpu())
                    particle_value = particles_pred if self.per_particle_context else mean_est
                    state_context, h_state = self._gru_step(
                        particle_value, particles_pred, h_state, batch_size=1)
                    x_curr  = particles_pred

                predictions   = torch.stack(predictions, dim=0)
                true_physical = true_seq_norm.cpu() * t_std.cpu() + t_mean.cpu()
                pred_physical = predictions          * t_std.cpu() + t_mean.cpu()
                all_rmses.append(
                    torch.sqrt(((pred_physical[1:] - true_physical[1:]) ** 2).mean()).item()
                )
                pos_error = pred_physical[1:, :2] - true_physical[1:, :2]
                pos_rmses.append(torch.sqrt((pos_error ** 2).mean()).item())

        self.log('val_rmse_rollout', sum(all_rmses) / len(all_rmses), prog_bar=True)
        self.log('val_rmse_position', sum(pos_rmses) / len(pos_rmses), prog_bar=True)

    # ── Optimizer hooks ───────────────────────────────────────────────────────
    def on_before_optimizer_step(self, optimizer):
        self.log('lr', optimizer.param_groups[0]['lr'],
                 prog_bar=True, on_step=True, on_epoch=False)

    def on_after_backward(self):
        # Global (all-parameter) gradient norm, computed BEFORE Lightning's
        # own automatic clipping (gradient_clip_val, set on the Trainer in
        # train.py) runs. clip_grad_norm_ both returns this pre-clip
        # value AND clips in-place to it as a side effect -- harmless
        # (idempotent) here since Lightning's own clip runs again right
        # after this hook and is then a no-op (already <= clip_val). This
        # is the actual quantity gradient_clip_val=5.0 is bounding -- the
        # per-parameter grad_norm/* below are informative but don't show
        # how hard clipping is working overall, or how often it engages.
        clip_val = self.trainer.gradient_clip_val
        if clip_val is not None:
            global_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)
            self.log('grad_norm_global_preclip', global_norm,
                     on_step=True, on_epoch=True, prog_bar=False)
            # 1.0 = no clipping needed this step; -> 0 the harder clipping
            # had to scale gradients down (e.g. ~1e-13 for a 1e13 spike
            # against clip_val=5.0 -- see session history for a case where
            # this ratio would have made an otherwise-invisible instability
            # immediately visible in a single scalar).
            clip_ratio = clip_val / torch.clamp(global_norm, min=clip_val)
            self.log('grad_clip_ratio', clip_ratio,
                     on_step=True, on_epoch=True, prog_bar=False)

        if self.trainer.global_step % 5 == 0:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.log(f'grad_norm/{name}', param.grad.norm().item(),
                             on_step=True, on_epoch=False, add_dataloader_idx=False)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        # Only include parameters that require gradients.
        trainable = [(n, p) for n, p in self.named_parameters() if p.requires_grad]

        meas_params  = [p for n, p in trainable if 'meas_model'  in n]
        other_params = [p for n, p in trainable if 'meas_model' not in n]

        param_groups = [{'params': other_params, 'lr': self.lr}]
        if meas_params:
            # Boost learned meas_model LR so h(x) keeps pace with the ODE flow
            lr_mult = 1.0
            param_groups.append({'params': meas_params, 'lr': self.lr * lr_mult})

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
