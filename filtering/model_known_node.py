"""
Known-dynamics, known-measurement TDOA filter whose ONLY learned component
is a small residual correction on top of the analytical Daum-Huang
particle-flow homotopy (Bell & Stone; see baseline_edh_tdoa.py's
exact_daum_huang_step) -- F, Q, h(x), and R are all exact/known, ported
from tdoa_data_gen.py / baseline_edh_tdoa.py.

Built to directly test whether a learned residual improves tracking beyond
EDH's own closed-form flow, holding fixed: the linearisation point (H at
the particle-ensemble mean, matching EDH's convention exactly, not a
per-particle Jacobian), the covariance estimate (empirical, from particle
spread -- exactly as EDH computes it, which is why this model requires
num_particles > 1: a single particle has no spread to measure), and the
frozen-innovation convention already used elsewhere in this codebase. The
residual's final layer is zero-initialised, so at construction this
model's measurement update is numerically the analytical Daum-Huang flow
alone (see test_known_node.py for a direct check against
baseline_edh_tdoa.py.exact_daum_huang_step).

Operates entirely in PHYSICAL units (km, km/s), not the z-scored space
model.py's fully-learned filter uses -- F/Q/h(x)/R are all taken
directly from tdoa_data_gen.py's real generative process with no rescaling,
exactly matching baseline_edh_tdoa.py. Denormalisation happens once, at the
top of _unroll, using the dataset's own t_mean/t_std/m_mean/m_std (set via
set_norm_stats below); every downstream computation, including the loss,
stays in physical space.

Companion to model.py's ModularNeuralODEFilter, the fully LEARNED
counterpart (learned dynamics via a GRU-based recurrent context, learned
measurement model) -- that file no longer contains any known_dynamics /
known_measurement code; all of it has been moved here.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

from solver import build_measurement_ode_solver


# ─────────────────────────────────────────────────────────────────────────────
#  Known TDOA Measurement  h(x) = range-difference over 2 sensor pairs
# ─────────────────────────────────────────────────────────────────────────────
class KnownTDOAMeasurement(nn.Module):
    """
    Analytical TDOA range-difference measurement function, ported from
    baseline_edh_tdoa.py's h_func, extended with a closed-form Jacobian
    (used by KnownDaumHuangODEFunc's analytical flow term). Operates
    entirely in physical units -- no learnable parameters, no
    normalisation. Sensor positions are registered as buffers via
    set_params() so they follow .to(device) automatically.
    """

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim   = obs_dim
        # Placeholder buffers (real values come from set_params) -- must
        # exist at __init__ time, not just after set_params, so a fresh
        # instance built by load_from_checkpoint() has slots for
        # load_state_dict() to populate. Re-registering the same name in
        # set_params below just overwrites the value, not an error -- see
        # KnownNodeFilter's identical pattern for F/Q_chol/R/etc.
        for name in ('L_REF', 'L_RX', 'L_REF2', 'L_RX2'):
            self.register_buffer(name, torch.zeros(2))

    def set_params(self,
                   L_REF: torch.Tensor, L_RX: torch.Tensor,    # [2] each, pair 1
                   L_REF2: torch.Tensor, L_RX2: torch.Tensor): # [2] each, pair 2
        self.register_buffer('L_REF',  L_REF.float())
        self.register_buffer('L_RX',   L_RX.float())
        self.register_buffer('L_REF2', L_REF2.float())
        self.register_buffer('L_RX2',  L_RX2.float())

    def _ranges(self, pos):
        """pos: [..., 2] physical. Returns (r_ref1, r_rx1, r_ref2, r_rx2), each [...]."""
        r_ref1 = torch.sqrt(((pos - self.L_REF)  ** 2).sum(dim=-1) + 1e-8)
        r_rx1  = torch.sqrt(((pos - self.L_RX)   ** 2).sum(dim=-1) + 1e-8)
        r_ref2 = torch.sqrt(((pos - self.L_REF2) ** 2).sum(dim=-1) + 1e-8)
        r_rx2  = torch.sqrt(((pos - self.L_RX2)  ** 2).sum(dim=-1) + 1e-8)
        return r_ref1, r_rx1, r_ref2, r_rx2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [..., state_dim] physical -> [..., obs_dim] physical."""
        pos = x[..., :2]
        r_ref1, r_rx1, r_ref2, r_rx2 = self._ranges(pos)
        return torch.stack([r_rx1 - r_ref1, r_rx2 - r_ref2], dim=-1)

    def jacobian(self, x_bar: torch.Tensor) -> torch.Tensor:
        """
        Closed-form dh/dx at x_bar : [B, state_dim] physical.
        Returns [B, obs_dim, state_dim].

        Derived by hand rather than torch.autograd.functional.jacobian
        (used by baseline_edh_tdoa.py, for a single unbatched trajectory)
        since h_func is simple enough to differentiate directly, and this
        needs to run batched, every timestep, inside a training loop:
            d(||pos-s||)/d(pos) = (pos-s)/||pos-s||   (unit vector away from s)
            d(h_pair)/d(pos)    = (pos-Rx)/r_Rx - (pos-Ref)/r_Ref
            d(h)/d(vx,vy)       = 0  (range-difference doesn't depend on velocity)
        """
        pos = x_bar[..., :2]                                    # [B, 2]
        r_ref1, r_rx1, r_ref2, r_rx2 = self._ranges(pos)

        d_pair1 = (pos - self.L_RX)  / r_rx1.unsqueeze(-1)  - (pos - self.L_REF)  / r_ref1.unsqueeze(-1)
        d_pair2 = (pos - self.L_RX2) / r_rx2.unsqueeze(-1)  - (pos - self.L_REF2) / r_ref2.unsqueeze(-1)

        B = x_bar.shape[0]
        H = x_bar.new_zeros(B, self.obs_dim, self.state_dim)
        H[:, 0, :2] = d_pair1
        H[:, 1, :2] = d_pair2
        # velocity columns (2:4) stay zero -- h doesn't depend on velocity.
        return H


# ─────────────────────────────────────────────────────────────────────────────
#  Neural ODE vector field: analytical Daum-Huang flow + learned residual
# ─────────────────────────────────────────────────────────────────────────────
class KnownDaumHuangODEFunc(nn.Module):
    """
    matrix_free=False (default):
        dx/dlam = A(lam) @ x + b(lam) + delta_theta(x, lam, innov_at_mean)

        A(lam) = -0.5 * P_lam @ H^T @ R_inv @ H
        b(lam) = P_lam @ H^T @ R_inv @ innov_at_mean - A(lam) @ x_bar
        P_lam  = (P_inv + lam * H^T @ R_inv @ H)^-1

        H, P_inv, R_inv, innov_at_mean, x_bar are frozen for the whole
        lam=0->1 solve of a given real timestep (computed once, from the
        empirical particle-ensemble covariance and the closed-form
        measurement Jacobian at the ensemble mean -- see
        KnownNodeFilter.forward) -- this exactly mirrors
        baseline_edh_tdoa.py's exact_daum_huang_step, with delta_theta as
        the one addition. delta_theta's final layer is zero-initialised
        (see KnownNodeFilter._init_weights), so at construction dx/dlam is
        the analytical Daum-Huang flow alone.

    matrix_free=True:
        dx/dlam = f_theta(x, lam, innov_at_mean [, cov_signal])
        f_theta = net(x, innov, lam [, cov_signal]) + innov_skip(innov)

        Structurally IDENTICAL to model.py's PFFODEFunc (same net
        architecture, same effective inputs) -- H/P_inv/R_inv are never
        consulted in this mode, so KnownNodeFilter.forward() skips
        computing the empirical-covariance inverse entirely (real compute
        savings, not just unused args). This isolates whether the
        closed-form analytical term itself is what helps, vs. just having
        exact F/Q/h(x) available for the predict step and the innovation.
        Passive start still holds: both net's out_layer and innov_skip are
        zero-initialised, so dx/dlam=0 at construction -- matching
        PFFODEFunc's own passive start.

        cov_conditioning=True (matrix_free only) additionally concatenates
        cov_signal -- the upper triangle of the analytical Kalman P_prior
        (see KnownNodeFilter._kalman_cov_step) -- into net's input, playing
        the same role legacy's uncertainty_signal (prev_log_var ‖
        uncertainty_h) plays for PFFODEFunc: an extra per-timestep
        conditioning input, frozen for the whole lam=0->1 sweep (computed
        from *last* step's posterior, not updated mid-integration -- see
        KnownNodeFilter.forward's docstring for the exact analogy). Unlike
        uncertainty_h, cov_signal is not learned/recurrent -- it's a
        closed-form function of the known F/Q/H/R, decoupled from
        num_particles entirely (see session history for why this replaces
        the empirical particle-spread covariance, which degenerates to a
        constant at num_particles=1 and was the original motivation, even
        though num_particles=50 is what's actually used).
    """

    def __init__(self, state_dim, obs_dim, hidden_dim, matrix_free=False,
                 cov_conditioning=False):
        super().__init__()
        self.matrix_free      = matrix_free
        self.cov_conditioning = cov_conditioning
        cov_dim = (state_dim * (state_dim + 1)) // 2 if cov_conditioning else 0
        in_dim = state_dim + obs_dim + 1 + cov_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Only used when matrix_free=True -- mirrors PFFODEFunc's innov_skip
        # exactly, standing in for the analytical term this mode omits.
        if matrix_free:
            self.innov_skip = nn.Linear(obs_dim, state_dim, bias=False)

    @property
    def out_layer(self):
        """Final Linear layer, for passive (zero-residual) start."""
        return self.net[-1]

    def forward(self, lam, x, innov_expanded, H=None, P_inv=None, R_inv=None,
                innov_at_mean=None, x_bar=None, cov_signal=None):
        """
        x              : [B, N, state_dim]  particle positions along the flow
        innov_expanded : [B, N, obs_dim]    innov_at_mean broadcast to every
                          particle -- the net's own input feature, not used
                          by the analytical term
        H, P_inv, R_inv, innov_at_mean, x_bar : only required (non-None)
                          when matrix_free=False -- see class docstring.
        cov_signal     : [B, N, state_dim*(state_dim+1)//2]  only required
                          (non-None) when cov_conditioning=True.
        """
        lam_t = (lam if isinstance(lam, torch.Tensor)
                 else torch.tensor(lam, dtype=x.dtype, device=x.device))
        lam_vec = lam_t.expand(x.shape[0], x.shape[1], 1)
        net_in_parts = [x, innov_expanded, lam_vec]
        if self.cov_conditioning:
            net_in_parts.append(cov_signal)
        net_out = self.net(torch.cat(net_in_parts, dim=-1))

        if self.matrix_free:
            return net_out + self.innov_skip(innov_expanded)

        HT = H.transpose(-1, -2)                                       # [B, D, obs_dim]
        HT_Rinv_H = HT @ R_inv @ H                                     # [B, D, D]
        P_lam = torch.linalg.inv(P_inv + lam_t * HT_Rinv_H)            # [B, D, D]
        A = -0.5 * P_lam @ HT_Rinv_H                                   # [B, D, D]
        Ax_bar = (A @ x_bar.unsqueeze(-1)).squeeze(-1)                 # [B, D]
        b = (P_lam @ HT @ R_inv @ innov_at_mean.unsqueeze(-1)).squeeze(-1) - Ax_bar  # [B, D]

        analytical = (A.unsqueeze(1) @ x.unsqueeze(-1)).squeeze(-1) + b.unsqueeze(1)  # [B, N, D]

        return analytical + net_out


# ─────────────────────────────────────────────────────────────────────────────
#  Main filter
# ─────────────────────────────────────────────────────────────────────────────
class KnownNodeFilter(pl.LightningModule):
    """
    Known-F/Q/h/R TDOA filter with a Daum-Huang-plus-learned-residual
    measurement update. See module docstring for the full rationale.

    num_particles > 1 is required -- P is estimated empirically from
    particle spread (matching EDH exactly), which is meaningless for a
    single particle. Loss is a point-estimate loss (huber/l2/l1) against
    the particle-ensemble mean, matching how EDH itself is scored in the
    eval scripts -- there is no learned uncertainty head in this model
    (EDH has none either, so there's nothing to compare against).

    matrix_free=False (default)  measurement_flow adds the closed-form
                Daum-Huang term (built from H/P_inv/R_inv) to the learned
                residual -- see KnownDaumHuangODEFunc's docstring.
    matrix_free=True   measurement_flow is structurally identical to
                model.py's PFFODEFunc (net + innov_skip, no
                analytical term) -- isolates whether the closed-form term
                itself matters, vs. just having exact F/Q/h(x) for
                prediction and the innovation. forward() skips computing
                the empirical-covariance inverse in this mode (H is still
                computed when cov_conditioning=True, see below).

    cov_conditioning=False (default)  no effect.
    cov_conditioning=True  (matrix_free only -- raises ValueError otherwise)
                measurement_flow additionally conditions on cov_signal, built
                from an analytically-propagated Kalman covariance
                P_prior = F @ P_post_prev @ F.T + Q, with P_post_prev
                obtained via a standard Kalman gain update using the known H
                (Jacobian at x_bar) and R -- see _kalman_cov_step. This is
                deliberately NOT the empirical particle-ensemble covariance:
                that quantity degenerates to a constant at num_particles=1
                (the original motivation for this feature), and unlike the
                analytical version, an open-loop F/Q-only propagation (no
                Kalman update) would grow unboundedly over a rollout.
                P_post is threaded through _unroll/on_validation_epoch_end
                as explicit per-trajectory state, playing the same
                structural role as model.py's uncertainty_h, just
                computed rather than learned, and -- like uncertainty_signal
                there -- frozen for the whole lam=0->1 sweep of a given real
                timestep (it reflects last step's Kalman posterior, not
                something that evolves during this step's homotopy solve).

                cov_signal is NOT P_prior's raw upper triangle -- position
                and velocity variances differ by ~3 orders of magnitude in
                normalised space (P_norm = P_phys / outer(t_std, t_std)
                scales as 1/(std_i*std_j), roughly squaring any existing
                position/velocity std disparity), which produced volatile,
                spiky gradients at measurement_flow's output layer in
                practice (see session history). Instead: diagonal entries
                are log-variance (log(diag(P_prior_norm)), matching
                model.py's prev_log_var convention exactly --
                compresses the multiplicative range into a compact additive
                one) and off-diagonal entries are correlation coefficients
                (cov_ij / sqrt(var_i*var_j), bounded in [-1,1], scale-
                invariant regardless of the position/velocity std
                disparity) -- see forward()'s cov_conditioning branch.
    """

    def __init__(self,
                 state_dim=4,
                 obs_dim=2,
                 hidden_dim=64,
                 num_particles=50,
                 lr=1e-3,
                 integration_steps=4,
                 integration_type='rk4',
                 lam_schedule='exponential',
                 loss_type='huber',
                 huber_delta=1.0,
                 process_noise_scale=1.0,
                 matrix_free=False,
                 cov_conditioning=False):
        super().__init__()
        if num_particles <= 1:
            raise ValueError(
                "KnownNodeFilter requires num_particles > 1 -- P is estimated "
                "empirically from particle spread (matching baseline_edh_tdoa.py "
                "exactly), which needs at least 2 particles to be meaningful.")
        if cov_conditioning and not matrix_free:
            raise ValueError(
                "cov_conditioning requires matrix_free=True -- the non-matrix-free "
                "measurement_flow already consumes H/P_inv/R directly via the "
                "analytical A(lam)/b(lam) term, so feeding their derived Kalman "
                "P_prior in again as a raw side-signal would be redundant.")
        self.save_hyperparameters()

        self.state_dim         = state_dim
        self.obs_dim            = obs_dim
        self.num_particles      = num_particles
        self.lr                 = lr
        self.loss_type          = loss_type.lower()
        self.huber_delta        = huber_delta
        self.integration_type   = integration_type
        self.integration_steps  = integration_steps
        self.process_noise_scale = process_noise_scale
        self.matrix_free        = matrix_free
        self.cov_conditioning   = cov_conditioning

        self.meas_model        = KnownTDOAMeasurement(state_dim, obs_dim)
        self.measurement_flow  = KnownDaumHuangODEFunc(
            state_dim, obs_dim, hidden_dim, matrix_free=matrix_free,
            cov_conditioning=cov_conditioning)

        if cov_conditioning:
            # Upper-triangle (incl. diagonal) indices for flattening the
            # symmetric P_prior into cov_signal -- precomputed once and
            # registered so they move with .to(device) automatically.
            rows, cols = torch.triu_indices(state_dim, state_dim)
            self.register_buffer('_cov_rows', rows, persistent=False)
            self.register_buffer('_cov_cols', cols, persistent=False)
            self.register_buffer('_cov_is_diag', rows == cols, persistent=False)

        # Placeholder buffers for set_ssm/set_measurement/set_norm_stats'
        # real values (registered post-construction, from train_known_node.py)
        # -- must exist at __init__ time, not just after those calls, so a
        # fresh instance built by load_from_checkpoint() has slots for
        # load_state_dict() to populate (otherwise the saved F/Q_chol/R/etc.
        # are silently dropped -- PyTorch logs "keys not in the model state
        # dict" but strict=False swallows it). Re-registering the same name
        # in set_ssm/etc. below just overwrites the value, not an error.
        self.register_buffer('F',          torch.zeros(state_dim, state_dim))
        self.register_buffer('Q_chol',     torch.zeros(state_dim, state_dim))
        self.register_buffer('R',          torch.zeros(obs_dim, obs_dim))
        self.register_buffer('R_inv',      torch.zeros(obs_dim, obs_dim))
        self.register_buffer('R_inv_norm', torch.zeros(obs_dim, obs_dim))
        self.register_buffer('t_mean',     torch.zeros(state_dim))
        self.register_buffer('t_std',      torch.ones(state_dim))
        self.register_buffer('m_mean',     torch.zeros(obs_dim))
        self.register_buffer('m_std',      torch.ones(obs_dim))
        self._measurement_set = False

        # ── λ schedule (same convention as model.py) ────────────────────
        N = integration_steps
        if lam_schedule == 'uniform':
            lam_steps = [1.0 / N] * N
        else:
            b  = 2.0
            s0 = (b - 1) / (b**N - 1)
            lam_steps = [s0 * (b**k) for k in range(N)]
        self.register_buffer('lam_steps',
                             torch.tensor(lam_steps, dtype=torch.float32))

        self.measurement_solver = build_measurement_ode_solver(
            integration_type=self.integration_type)

        self._init_weights()

    # ── Registration hooks (called post-construction, from train_known_node.py) ─
    def set_ssm(self, F: torch.Tensor, Q_chol: torch.Tensor):
        """Register the (physical-space) CV transition matrix and process-noise
        Cholesky factor -- see tdoa_data_gen.py / baseline_edh_tdoa.py."""
        self.register_buffer('F', F)
        self.register_buffer('Q_chol', Q_chol)

    def set_measurement(self, L_REF, L_RX, L_REF2, L_RX2, measurement_std: float):
        """Register sensor positions and R (physical-space, diagonal)."""
        self.meas_model.set_params(L_REF, L_RX, L_REF2, L_RX2)
        R = torch.eye(self.obs_dim) * (measurement_std ** 2)
        self.register_buffer('R', R)
        self.register_buffer('R_inv', torch.linalg.inv(R))
        self._measurement_set = True

    def set_norm_stats(self, t_mean, t_std, m_mean, m_std):
        """
        Dataset normalisation stats. Used by forward() to move into
        normalised space for the correction-step integration and back to
        physical space around it (see forward()'s docstring) -- NOT to
        denormalise _unroll's input batches, which are already physical
        (FileTrackingDataModule constructed with normalize=False, see
        train_known_node.py).

        Must be called AFTER set_measurement() -- R_inv_norm is derived
        here from self.R (registered by set_measurement) and m_std.
        """
        self.register_buffer('t_mean', t_mean.squeeze())
        self.register_buffer('t_std',  t_std.squeeze())
        self.register_buffer('m_mean', m_mean.squeeze())
        self.register_buffer('m_std',  m_std.squeeze())

        if not self._measurement_set:
            raise RuntimeError(
                "set_norm_stats() must be called after set_measurement() -- "
                "R_inv_norm is derived from self.R and m_std.")
        m_std_flat = self.m_std
        R_norm = self.R / (m_std_flat.unsqueeze(0) * m_std_flat.unsqueeze(1))
        self.register_buffer('R_inv_norm', torch.linalg.inv(R_norm))

    # ── Init ─────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0,
                                         mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
        # Passive start: the learned residual begins at exactly zero, so the
        # vector field starts as the analytical Daum-Huang flow alone (see
        # KnownDaumHuangODEFunc's docstring / test_known_node.py).
        nn.init.zeros_(self.measurement_flow.out_layer.weight)
        nn.init.zeros_(self.measurement_flow.out_layer.bias)
        if self.matrix_free:
            nn.init.zeros_(self.measurement_flow.innov_skip.weight)

    # ── Predict step (known CV dynamics) ────────────────────────────────────
    def _predict(self, particles: torch.Tensor) -> torch.Tensor:
        """particles: [B, N, state_dim] physical -> [B, N, state_dim] physical."""
        return (self.F @ particles.transpose(-1, -2)).transpose(-1, -2)

    # ── Analytical Kalman covariance recursion (cov_conditioning only) ──────
    def _kalman_cov_step(self, P_post_prev: torch.Tensor, H_phys: torch.Tensor):
        """
        Standard EKF predict+update covariance recursion, physical units,
        entirely decoupled from the particle ensemble -- see
        cov_conditioning's docstring for why this replaces the empirical
        particle-spread covariance. Joseph-form update for numerical
        stability (guaranteed PSD under rounding error, unlike the short
        form) -- cheap to afford here since this is one obs_dim x obs_dim
        inverse per real timestep, nothing like the per-lambda-step
        state_dim x state_dim inverses matrix_free=True already avoids.

        P_post_prev : [B, state_dim, state_dim] physical -- previous step's
                      posterior (0 at the first real step, since _unroll
                      starts particles at the true state exactly, so there
                      is no uncertainty yet to propagate).
        H_phys      : [B, obs_dim, state_dim] physical Jacobian at x_bar.

        Returns (P_prior, P_post), both [B, state_dim, state_dim] physical.
        P_prior is this step's cov_signal source (frozen for the whole
        lam=0->1 sweep); P_post is threaded to the next step's call.
        """
        Q = self.Q_chol @ self.Q_chol.T
        P_prior = self.F @ P_post_prev @ self.F.T + Q                 # [B, D, D]

        HT = H_phys.transpose(-1, -2)                                 # [B, D, obs_dim]
        S  = H_phys @ P_prior @ HT + self.R                           # [B, obs_dim, obs_dim]
        K  = P_prior @ HT @ torch.linalg.inv(S)                       # [B, D, obs_dim]

        I   = torch.eye(self.state_dim, device=P_prior.device, dtype=P_prior.dtype)
        IKH = I - K @ H_phys                                          # [B, D, D]
        P_post = (IKH @ P_prior @ IKH.transpose(-1, -2)
                  + K @ self.R @ K.transpose(-1, -2))                 # [B, D, D]
        return P_prior, P_post

    # ── cov_signal reparameterisation (cov_conditioning only) ───────────────
    def _build_cov_signal(self, P_norm: torch.Tensor) -> torch.Tensor:
        """
        P_norm's raw upper triangle is badly conditioned as a net input --
        position and velocity variances differ by ~3 orders of magnitude in
        normalised space (see cov_conditioning's docstring), which produced
        volatile output-layer gradients in practice. Reparameterise instead:
        diagonal (variance) entries -> log-variance, matching
        model.py's prev_log_var convention exactly (compresses the
        multiplicative range into a compact additive one); off-diagonal
        (covariance) entries -> correlation coefficients, bounded in [-1,1]
        and scale-invariant regardless of the position/velocity std
        disparity that made the raw covariance so poorly scaled.

        P_norm : [B, state_dim, state_dim], symmetric, PSD (up to solver
                 rounding error).
        Returns cov_signal : [B, state_dim*(state_dim+1)//2].
        """
        diag      = torch.diagonal(P_norm, dim1=-2, dim2=-1).clamp(min=1e-8)  # [B, D]
        log_var   = torch.log(diag)                                          # [B, D]
        std       = torch.sqrt(diag)                                        # [B, D]
        denom     = std.unsqueeze(-1) * std.unsqueeze(-2)                    # [B, D, D]
        corr      = P_norm / denom.clamp(min=1e-8)                          # [B, D, D]

        corr_entries    = corr[:, self._cov_rows, self._cov_cols]           # [B, cov_dim]
        log_var_entries = log_var[:, self._cov_rows]                        # [B, cov_dim] (valid where diag)
        return torch.where(self._cov_is_diag, log_var_entries, corr_entries)

    # ── Forward: one full timestep (predict + Daum-Huang-plus-residual update) ─
    def forward(self, particles_prev: torch.Tensor, current_meas: torch.Tensor,
                P_post_prev: torch.Tensor = None):
        """
        particles_prev : [B, N, state_dim] physical
        current_meas   : [B, obs_dim] physical
        P_post_prev    : [B, state_dim, state_dim] physical -- only used
                          (and only meaningful) when cov_conditioning=True;
                          see _kalman_cov_step.

        Returns (particles_pred, particles_prior) normally, both
        [B, N, state_dim] physical -- particles_prior is the pre-correction
        (predict-only) ensemble, kept for the same diagnostic/aux-loss
        purposes model.py's forward() returns it for. When
        cov_conditioning=True, additionally returns P_post as a third
        element (to be threaded into the next call's P_post_prev).

        The predict step (below) stays entirely in physical units -- F/Q
        are the real tdoa_data_gen.py generative-process matrices, so they
        need to act on the real physical state. The correction step
        (measurement_flow's λ-homotopy solve) instead runs entirely in
        NORMALISED units -- H/P/R/innovation are all rescaled into that
        space first, and the result is denormalised back to physical
        afterward. This keeps the learned residual's input scale sane
        (positions/velocities/innovations would otherwise be raw km-scale,
        badly mismatched with Kaiming-initialised layers expecting roughly
        unit-variance inputs -- this was the actual cause of a training-
        time NaN blowup at integration_steps=4, see session history) while
        keeping the predict step exact.

        The rescaling is provably lossless for the ANALYTICAL part of the
        flow (delta_theta=0): running the linear-Gaussian Daum-Huang
        update in a rescaled coordinate system and converting back gives
        the identical result to running it directly in physical space
        (verified numerically to 1e-6 on a synthetic case -- see session
        history). This is a standard property of affine reparameterisation
        of linear-Gaussian systems, the same principle behind why a Kalman
        filter can be run in any consistent linear coordinate system.
        Rescaling rules used below (x_norm = (x_phys - mean) / std):
            H_norm = H_phys * t_std[None,None,:] / m_std[None,:,None]
            R_norm = R_phys / outer(m_std, m_std)          (see set_norm_stats)
            innov_norm = innov_phys / m_std   (mean cancels in a difference)
            P computed directly from the normalised particle ensemble.
        """
        B, N = particles_prev.shape[0], particles_prev.shape[1]

        # STEP 1: PREDICT -- physical units throughout.
        x_det = self._predict(particles_prev)
        noise = torch.randn(B, N, self.state_dim, device=x_det.device, dtype=x_det.dtype)
        x_prior_phys = x_det + noise @ self.Q_chol.T

        # STEP 2: CORRECT -- normalised units throughout.
        x_prior = (x_prior_phys - self.t_mean) / self.t_std
        x_bar   = x_prior.mean(dim=1)                                     # [B, D] normalised
        x_bar_phys = x_bar * self.t_std + self.t_mean

        # innov_at_mean only needs the exact (known) h(x_bar) -- no
        # Jacobian required -- so this is computed the same way in both
        # modes.
        z_pred_at_mean_phys = self.meas_model(x_bar_phys)                 # [B, obs_dim] physical
        innov_at_mean  = (current_meas - z_pred_at_mean_phys) / self.m_std  # normalised
        innov_expanded = innov_at_mean.unsqueeze(1).expand(-1, N, -1)     # [B, N, obs_dim]

        if self.matrix_free:
            if self.cov_conditioning:
                # H is needed here purely to drive the side-channel Kalman
                # update (_kalman_cov_step) -- it never enters the ODE
                # net's own inputs directly, only cov_signal does. Still
                # skips the expensive per-lambda-step state_dim x state_dim
                # inverse the non-matrix-free branch pays for.
                H_phys = self.meas_model.jacobian(x_bar_phys)             # [B, obs_dim, D] physical
                P_prior_phys, P_post_phys = self._kalman_cov_step(P_post_prev, H_phys)
                P_prior_norm = P_prior_phys / torch.outer(self.t_std, self.t_std)
                cov_signal = self._build_cov_signal(P_prior_norm)         # [B, cov_dim]
                cov_signal_expanded = cov_signal.unsqueeze(1).expand(-1, N, -1)

                def wrapper_flow(lam, x_):
                    return self.measurement_flow(
                        lam, x_, innov_expanded, cov_signal=cov_signal_expanded)
            else:
                # H/P_inv are never consulted by the vector field in this
                # mode -- skip the Jacobian and the (expensive, per-step)
                # batched matrix inverse entirely. See
                # KnownDaumHuangODEFunc's docstring.
                def wrapper_flow(lam, x_):
                    return self.measurement_flow(lam, x_, innov_expanded)
        else:
            # Empirical prior covariance from the normalised particle
            # spread -- the same P_prior baseline_edh_tdoa.py computes,
            # just in normalised space; the +1e-6*I regularisation is far
            # more meaningful here than it would be against km^2-scale
            # physical variances.
            centered = x_prior - x_bar.unsqueeze(1)                       # [B, N, D] normalised
            P_prior  = torch.einsum('bnd,bne->bde', centered, centered) / (N - 1)
            P_prior  = P_prior + torch.eye(self.state_dim, device=P_prior.device) * 1e-6
            P_inv    = torch.linalg.inv(P_prior)

            H_phys = self.meas_model.jacobian(x_bar_phys)                 # [B, obs_dim, D] physical
            H = H_phys * self.t_std.view(1, 1, -1) / self.m_std.view(1, -1, 1)  # normalised

            def wrapper_flow(lam, x_):
                return self.measurement_flow(
                    lam, x_, innov_expanded, H, P_inv, self.R_inv_norm, innov_at_mean, x_bar)

        particles_pred_norm = self.measurement_solver(wrapper_flow, x_prior, self.lam_steps)
        particles_pred = particles_pred_norm * self.t_std + self.t_mean

        if self.cov_conditioning:
            return particles_pred, x_prior_phys, P_post_phys
        return particles_pred, x_prior_phys

    # ── Loss ─────────────────────────────────────────────────────────────────
    def calculate_loss(self, predictions, targets, mask=None):
        if self.loss_type == 'l1':
            elementwise = torch.abs(predictions - targets)
        elif self.loss_type in ('l2', 'mse'):
            elementwise = (predictions - targets) ** 2
        elif self.loss_type == 'huber':
            elementwise = torch.nn.functional.huber_loss(
                predictions, targets, reduction='none', delta=self.huber_delta)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        elementwise = elementwise.mean(dim=-1)
        if mask is not None:
            active = mask.float()
            return (elementwise * active).sum() / active.sum().clamp(min=1)
        return elementwise.mean()

    # ── Unroll ───────────────────────────────────────────────────────────────
    def _unroll(self, obs_window, true_states):
        """
        obs_window, true_states : [B, T, obs_dim]/[B, T, state_dim], already
        PHYSICAL (FileTrackingDataModule constructed with normalize=False --
        see train_known_node.py). No denormalisation needed here --
        forward() handles its own internal normalize/denormalize round trip
        around the correction step only (see its docstring).
        """
        seq_len    = true_states.shape[1]
        batch_size = true_states.shape[0]

        particles_curr = true_states[:, 0, :].unsqueeze(1).expand(-1, self.num_particles, -1)
        # P_post starts at 0 -- _unroll starts particles at the true state
        # exactly (see above), so there is no uncertainty yet to propagate;
        # step 1's P_prior then naturally bootstraps to Q. Only meaningful
        # when cov_conditioning=True (see _kalman_cov_step).
        P_post = (torch.zeros(batch_size, self.state_dim, self.state_dim,
                              device=true_states.device, dtype=true_states.dtype)
                  if self.cov_conditioning else None)

        predictions_list = []
        priors_list      = []
        for t in range(1, seq_len):
            current_meas = obs_window[:, t, :]
            if self.cov_conditioning:
                particles_pred, particles_prior, P_post = self(
                    particles_curr, current_meas, P_post)
            else:
                particles_pred, particles_prior = self(particles_curr, current_meas)

            predictions_list.append(particles_pred.mean(dim=1))
            priors_list.append(particles_prior.mean(dim=1))
            particles_curr = particles_pred

        return (torch.stack(predictions_list, dim=1),
                torch.stack(priors_list, dim=1),
                true_states)

    # ── Batch unpacking ──────────────────────────────────────────────────────
    @staticmethod
    def _unpack_batch(batch):
        obs_window, true_states, mask = batch
        return obs_window, true_states, mask

    # ── Training ─────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        obs_window, true_states, mask = self._unpack_batch(batch)
        active_mask = mask[:, 1:]

        if batch_idx == 0:
            print(f"[Epoch {self.current_epoch}] "
                  f"seq={true_states.shape[1]} batch={true_states.shape[0]} "
                  f"particles={self.num_particles}", flush=True)

        all_predictions, _, true_phys = self._unroll(obs_window, true_states)
        loss = self.calculate_loss(all_predictions, true_phys[:, 1:, :], mask=active_mask)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    # ── Validation ───────────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        obs_window, true_states, mask = self._unpack_batch(batch)
        all_predictions, _, true_phys = self._unroll(obs_window, true_states)
        active_mask = mask[:, 1:]
        loss = self.calculate_loss(all_predictions, true_phys[:, 1:, :], mask=active_mask)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    # ── Full-trajectory rollout RMSE ────────────────────────────────────────
    def on_validation_epoch_end(self):
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            return

        dataset = self.val_dataset
        meas_tensor  = dataset.measurements
        state_tensor = dataset.true_states
        n_traj = min(5, meas_tensor.shape[0])
        if not hasattr(self, '_val_indices'):
            self._val_indices = torch.randperm(meas_tensor.shape[0])[:n_traj]

        all_rmses, pos_rmses = [], []
        self.eval()
        with torch.no_grad():
            for idx in self._val_indices:
                # Already physical -- val_dataset was built with
                # normalize=False (see train_known_node.py).
                meas_seq  = meas_tensor[idx].to(self.device).unsqueeze(0).float()
                true_seq  = state_tensor[idx].to(self.device).unsqueeze(0).float()
                total_steps = meas_seq.shape[1]

                x_curr = true_seq[:, 0, :].unsqueeze(1).expand(-1, self.num_particles, -1)
                predictions = [true_seq[0, 0, :].cpu()]
                P_post = (torch.zeros(1, self.state_dim, self.state_dim,
                                      device=self.device, dtype=x_curr.dtype)
                          if self.cov_conditioning else None)

                for t in range(1, total_steps):
                    if self.cov_conditioning:
                        particles_pred, _, P_post = self(x_curr, meas_seq[:, t, :], P_post)
                    else:
                        particles_pred, _ = self(x_curr, meas_seq[:, t, :])
                    predictions.append(particles_pred.mean(dim=1).squeeze(0).cpu())
                    x_curr = particles_pred

                predictions = torch.stack(predictions, dim=0)
                true_phys   = true_seq.squeeze(0).cpu()
                all_rmses.append(
                    torch.sqrt(((predictions[1:] - true_phys[1:]) ** 2).mean()).item())
                pos_error = predictions[1:, :2] - true_phys[1:, :2]
                pos_rmses.append(torch.sqrt((pos_error ** 2).mean()).item())

        self.log('val_rmse_rollout', sum(all_rmses) / len(all_rmses), prog_bar=True)
        self.log('val_rmse_position', sum(pos_rmses) / len(pos_rmses), prog_bar=True)

    # ── Optimizer hooks ──────────────────────────────────────────────────────
    def on_before_optimizer_step(self, optimizer):
        self.log('lr', optimizer.param_groups[0]['lr'],
                 prog_bar=True, on_step=True, on_epoch=False)

    def on_after_backward(self):
        if self.trainer.global_step % 5 == 0:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.log(f'grad_norm/{name}', param.grad.norm().item(),
                             on_step=True, on_epoch=False, add_dataloader_idx=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
