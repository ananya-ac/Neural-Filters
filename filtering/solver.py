import torch


def _integrate_euler(wrapper_flow, x0, lam_steps):
    x = x0
    lam = torch.zeros(1, device=x.device, dtype=x.dtype)
    for delta in lam_steps:
        dt = delta.to(dtype=x.dtype)
        x = x + dt * wrapper_flow(lam, x)
        lam = lam + dt
    return x


def _integrate_rk4(wrapper_flow, x0, lam_steps):
    x = x0
    lam = torch.zeros(1, device=x.device, dtype=x.dtype)
    for delta in lam_steps:
        dt = delta.to(dtype=x.dtype)
        k1 = wrapper_flow(lam,        x)
        k2 = wrapper_flow(lam + dt/2, x + dt/2 * k1)
        k3 = wrapper_flow(lam + dt/2, x + dt/2 * k2)
        k4 = wrapper_flow(lam + dt,   x + dt   * k3)
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        lam = lam + dt
    return x


def _integrate_torchdiffeq(wrapper_flow, x0, use_adjoint, rtol, atol, adjoint_params=None):
    if use_adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    bsz, n_particles, dim = x0.shape

    def _ode(lam_, x_flat):
        x_ = x_flat.view(bsz, n_particles, dim)
        return wrapper_flow(lam_, x_).view(bsz * n_particles, dim)

    t_span = torch.tensor([0.0, 1.0], dtype=x0.dtype, device=x0.device)
    kwargs = dict(method='dopri5', rtol=rtol, atol=atol)
    if use_adjoint:
        # odeint_adjoint needs to know which leaf tensors require gradients
        # through the backward-time adjoint solve. _ode is a plain closure
        # (not an nn.Module), so torchdiffeq can't auto-discover this --
        # without adjoint_params it raises "func must be an instance of
        # nn.Module ... alternatively [params] can be specified explicitly".
        # The caller (e.g. ModularNeuralODEFilter.forward) passes
        # tuple(self.parameters()) -- safe even though wrapper_flow only
        # touches a subset: torchdiffeq just contributes zero/no gradient
        # for the params outside its computation graph.
        if adjoint_params is None:
            raise ValueError(
                "adjoint=True requires adjoint_params (e.g. "
                "tuple(self.parameters())) to be passed through at call "
                "time -- wrapper_flow is a plain closure, not an "
                "nn.Module torchdiffeq could auto-discover parameters from.")
        kwargs['adjoint_params'] = adjoint_params
    x_final = odeint(
        _ode,
        x0.view(bsz * n_particles, dim),
        t_span,
        **kwargs,
    )[-1]
    return x_final.view(bsz, n_particles, dim)


def build_measurement_ode_solver(
    integration_type='rk4',
    adaptive_steps=False,
    adjoint=False,
    rtol=1e-4,
    atol=1e-5,
):
    """Build a solver callable for measurement-flow ODE integration.

    Returned callable signature:
        solve(wrapper_flow, x0, lam_steps, adjoint_params=None)
    where wrapper_flow(lam, x) returns dx/dlam with x shaped [B, N, D].
    adjoint_params is only consulted when adjoint=True -- pass
    tuple(self.parameters()) from the calling model's forward() (must be
    supplied at call time, not solver-build time, since __init__ may not
    have finished registering every submodule yet when this function runs
    -- see _integrate_torchdiffeq's docstring comment). Ignored by every
    other path (rk4/euler/non-adjoint-adaptive), accepted everywhere for a
    uniform call signature.
    """
    if adaptive_steps:
        return lambda wrapper_flow, x0, lam_steps, adjoint_params=None: _integrate_torchdiffeq(
            wrapper_flow, x0, use_adjoint=adjoint, rtol=rtol, atol=atol,
            adjoint_params=adjoint_params,
        )

    if integration_type == 'rk4':
        return lambda wrapper_flow, x0, lam_steps, adjoint_params=None: _integrate_rk4(
            wrapper_flow, x0, lam_steps
        )

    if integration_type == 'euler':
        return lambda wrapper_flow, x0, lam_steps, adjoint_params=None: _integrate_euler(
            wrapper_flow, x0, lam_steps
        )

    raise ValueError(
        "integration_type must be 'rk4' or 'euler' when adaptive_steps=False"
    )


@torch.no_grad()
def sample_cfg_euler(model, condition, num_steps=10, w=2.0):
    """Generates next states given the condition (x_{t-1}, z_t) using CFG."""
    model.eval()
    batch_size, device = condition.shape[0], condition.device

    #state_dim = model.output_net.out_features
    state_dim = list(model.output_net.children())[-1].out_features

    # 1. Start from base distribution (pure noise) using the dynamic dimension
    x_t = torch.randn(batch_size, state_dim, device=device)

    # 2. Time steps for the ODE solver
    dt = 1.0 / num_steps
    t_vals = torch.linspace(0, 1.0 - dt, num_steps, device=device)

    null_condition = torch.zeros_like(condition)

    for t_val in t_vals:
        t_tensor = torch.full((batch_size, 1), t_val, device=device)

        # Predict unconditionally and conditionally
        v_uncond = model(x_t, t_tensor, null_condition)
        v_cond = model(x_t, t_tensor, condition)

        # Extrapolate using Classifier-Free Guidance scale 'w'
        v_cfg = v_uncond + w * (v_cond - v_uncond)

        # Euler Integration step
        x_t = x_t + v_cfg * dt

    return x_t


@torch.no_grad()
def sample_cfg_rk4(model, condition, num_steps=10, w=1.0):
    """Generates next states using a 4th-order Runge-Kutta (RK4) ODE solver with CFG."""
    model.eval()
    batch_size, device = condition.shape[0], condition.device

    # Dynamically get the state dimension from the model's output layer
    #state_dim = model.output_net.out_features
    state_dim = list(model.output_net.children())[-1].out_features

    # 1. Start from base distribution (pure noise)
    x_t = torch.randn(batch_size, state_dim, device=device)

    # 2. Time steps for the ODE solver
    dt = 1.0 / num_steps
    t_vals = torch.linspace(0, 1.0 - dt, num_steps, device=device)

    null_condition = torch.zeros_like(condition)

    # --- Helper function to calculate the CFG vector field ---
    def get_v_cfg(x, current_t):
        # Create the time tensor for the batch
        t_tensor = torch.full((batch_size, 1), current_t, device=device)

        # Predict unconditionally and conditionally
        v_uncond = model(x, t_tensor, null_condition)
        v_cond = model(x, t_tensor, condition)

        # Extrapolate using Classifier-Free Guidance scale 'w'
        return v_uncond + w * (v_cond - v_uncond)
    # ---------------------------------------------------------

    for t_val in t_vals:
        t_float = t_val.item()

        # k1: Slope at the beginning of the interval
        k1 = get_v_cfg(x_t, t_float)

        # k2: Slope at the midpoint (using k1)
        x_k2 = x_t + 0.5 * dt * k1
        t_k2 = t_float + 0.5 * dt
        k2 = get_v_cfg(x_k2, t_k2)

        # k3: Slope at the midpoint (using k2)
        x_k3 = x_t + 0.5 * dt * k2
        t_k3 = t_float + 0.5 * dt
        k3 = get_v_cfg(x_k3, t_k3)

        # k4: Slope at the end of the interval (using k3)
        x_k4 = x_t + dt * k3
        t_k4 = t_float + dt
        k4 = get_v_cfg(x_k4, t_k4)

        # RK4 Integration step: Weighted average of the slopes
        x_t = x_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_t
