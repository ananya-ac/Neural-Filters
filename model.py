
import torch.nn as nn
import math
import torch
import pytorch_lightning as pl
from torchdiffeq import odeint_adjoint as odeint

class PhysicsODEFunc(nn.Module):
    """Propagates particles through REAL time (the physics prior)."""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), # +1 for time 't'
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, t, x):
        # x shape: [Batch, NumParticles, StateDim]
        t_vec = torch.full((x.shape[0], x.shape[1], 1), t, device=x.device)
        h = torch.cat([x, t_vec], dim=-1)
        return self.net(h)

# class MeasurementFlowODEFunc(nn.Module):
#     """Propagates particles through PSEUDO-time (the sensor update)."""
#     def __init__(self, state_dim, context_dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             # +1 for pseudo-time 'lambda'
#             nn.Linear(state_dim + context_dim + 1, hidden_dim), 
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, state_dim)
#         )
        
#     # def forward(self, lam, x, context):
#     #     # Expand context so every particle gets a copy of the sensor reading
#     #     # context shape: [Batch, ContextDim] -> [Batch, NumParticles, ContextDim]
#     #     context_expanded = context.unsqueeze(1).expand(-1, x.shape[1], -1)
#     #     lam_vec = torch.full((x.shape[0], x.shape[1], 1), lam, device=x.device)
        
#     #     h = torch.cat([x, context_expanded, lam_vec], dim=-1)
#     #     return self.net(h)

#     def forward(self, lam, x, context):
#         # SAFET CHECK: Extract the float value if 'lam' is a tensor from torchdiffeq
#         lam_val = lam.item() if isinstance(lam, torch.Tensor) else lam
        
#         context_expanded = context.unsqueeze(1).expand(-1, x.shape[1], -1)
#         lam_vec = torch.full((x.shape[0], x.shape[1], 1), lam_val, device=x.device)
        
#         h = torch.cat([x, context_expanded, lam_vec], dim=-1)
#         return self.net(h)

class MeasurementFlowODEFunc(nn.Module):
    """Propagates particles through PSEUDO-time with Deep Context Injection."""
    def __init__(self, state_dim, context_dim, hidden_dim):
        super().__init__()
        
        # Layer 1: Takes [State + Context + Lambda]
        self.fc1 = nn.Linear(state_dim + context_dim + 1, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Layer 2: Takes [Hidden + Context + Lambda] -> DEEP INJECTION
        self.fc2 = nn.Linear(hidden_dim + context_dim + 1, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Output Layer: Takes [Hidden + Context + Lambda] -> DEEP INJECTION
        self.fc3 = nn.Linear(hidden_dim + context_dim + 1, state_dim)
        
        self.act = nn.SiLU()

    def forward(self, lam, x, context):
        # 1. Prepare the conditioning vectors (Context + Lambda)
        lam_val = lam.item() if isinstance(lam, torch.Tensor) else lam
        
        # Expand context so every particle gets a copy
        context_expanded = context.unsqueeze(1).expand(-1, x.shape[1], -1)
        lam_vec = torch.full((x.shape[0], x.shape[1], 1), lam_val, device=x.device)
        
        # Bundle the conditioning variables together
        cond = torch.cat([context_expanded, lam_vec], dim=-1)
        
        # 2. Layer 1 (Standard Injection)
        h1_input = torch.cat([x, cond], dim=-1)
        h1 = self.act(self.ln1(self.fc1(h1_input)))
        
        # 3. Layer 2 (Deep Injection)
        # We concatenate the raw sensor context AGAIN to the hidden state!
        h2_input = torch.cat([h1, cond], dim=-1)
        h2 = self.act(self.ln2(self.fc2(h2_input)))
        
        # 4. Output Layer (Deep Injection)
        # One last reminder of the sensors before outputting the vector field
        h3_input = torch.cat([h2, cond], dim=-1)
        out = self.fc3(h3_input)
        
        return out

class ObservationEncoder(nn.Module):
    """Extracts features STRICTLY from the current observation (no history)."""
    def __init__(self, obs_dim, context_dim):
        super().__init__()
        # A simple MLP to map the instantaneous observation to the latent context
        self.net = nn.Sequential(
            nn.Linear(obs_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
    def forward(self, current_obs):
        # Expected input shape: [Batch, ObsDim]
        return self.net(current_obs)


class DiscretePhysicsPredictor(nn.Module):
    """Propagates particles forward exactly one discrete time step."""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, x):
        # x shape: [Batch, NumParticles, StateDim]
        # Residual connection: we predict the change (velocity/momentum) and add it to the current position
        dx = self.net(x)
        return x + dx

class ModularNeuralODEFilter(pl.LightningModule):
    def __init__(self, state_dim=4, obs_dim=1, context_dim=32, hidden_dim=64, num_particles=50, lr=1e-3, process_noise_std=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.lr = lr
        self.process_noise_std = process_noise_std
        
        # 1. Sub-networks
        self.obs_encoder = ObservationEncoder(obs_dim, context_dim)
        #self.physics_ode = PhysicsODEFunc(state_dim, hidden_dim)
        self.physics_predictor = DiscretePhysicsPredictor(state_dim, hidden_dim)
        self.measurement_flow = MeasurementFlowODEFunc(state_dim, context_dim, hidden_dim)

    def forward(self, particles_prev, current_obs):
        # -----------------------------------------
        # STEP 1: PREDICT (Discrete Physics Step)
        # -----------------------------------------
        x_prior = self.physics_predictor(particles_prev)
        noise = torch.randn_like(x_prior) * self.process_noise_std
        x_prior = x_prior + noise
        
        # -----------------------------------------
        # STEP 2: UPDATE (Measurement Flow via torchdiffeq)
        # -----------------------------------------
        latent_obs = self.obs_encoder(current_obs)
        
        # Create a closure that matches torchdiffeq's expected signature f(t, y)
        def wrapper_flow(lam, x):
            return self.measurement_flow(lam, x, latent_obs)
            
        # We only need the start and end points of the pseudo-time interval
        lam_vals = torch.tensor([0.0, 1.0], device=self.device)
        num_steps = 5 #add to config
        step_size = (lam_vals[1] - lam_vals[0]) / num_steps

        # Integrate! 
        # integrated_states = odeint(
        #     wrapper_flow, 
        #     x_prior, 
        #     lam_vals, 
        #     method='rk4', 
        #     options={'step_size': step_size.item()} # Enforces the fixed steps
        # )
        integrated_states = odeint(
            wrapper_flow, 
            x_prior, 
            lam_vals, 
            method='rk4', 
            options={'step_size': step_size.item()},
            adjoint_params=tuple(self.measurement_flow.parameters()) # Enforces the fixed steps
        )
        
        # Integrate! 
        # You can change 'euler' to 'rk4' for significantly higher accuracy 
        # without changing your num_steps.
        # integrated_states = odeint(
        #     wrapper_flow, 
        #     x_prior, 
        #     lam_vals
        # )
        
        # odeint returns the state at all timepoints specified in lam_vals.
        # lam_vals has 2 elements (0.0 and 1.0), so we grab the last one [-1]
        x_posterior = integrated_states[-1]
            
        return x_posterior

    def training_step(self, batch, batch_idx):
        obs_window, true_states = batch 
        seq_len = true_states.shape[1]
        batch_size = true_states.shape[0]

        # 1. Initialize the Particle Cloud at t=0
        # Start with the true state, and expand it into N noisy particles
        x_start = true_states[:, 0, :].unsqueeze(1).expand(-1, self.num_particles, -1)
        initial_noise = torch.randn_like(x_start) * 0.1
        particles_curr = x_start + initial_noise
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
        teacher_forcing_ratio = max(0.0, 1.0 - (self.current_epoch / max_epochs))
        
        predictions_list = []

        # 2. Autoregressively unroll the predictions
        for t in range(1, seq_len):
            #current_obs = obs_window[:, :t+1, :]
            
            # Predict the particle cloud at time 't'
            #particles_pred = self(particles_prev=particles_curr, obs_seq=current_obs)

            current_obs = obs_window[:, t, :]
            
            # Predict the particle cloud at time 't'
            particles_pred = self(particles_prev=particles_curr, current_obs=current_obs)
            
            # Estimate the state by taking the mean of the particle cloud
            state_estimate = particles_pred.mean(dim=1)
            predictions_list.append(state_estimate)
            
            if torch.rand(1).item() < teacher_forcing_ratio:
                # Use the TRUE state to build the prior for the next step
                x_true = true_states[:, t, :]
                x_true_expanded = x_true.unsqueeze(1).expand(-1, self.num_particles, -1)
                
                # We still add process noise so the flow ODE learns to compress a cloud!
                particles_curr = x_true_expanded + (torch.randn_like(x_true_expanded) * self.process_noise_std)
            else:
                # Use our OWN prediction for the next step
                # (We detach it so gradients don't flow endlessly through the teacher-forced paths, saving memory)
                particles_curr = particles_pred.detach()
            
            # The posterior cloud becomes the prior cloud for the next step!
            particles_curr = particles_pred 

        # 3. Stack and Calculate Loss
        all_predictions = torch.stack(predictions_list, dim=1)
        loss = torch.nn.functional.mse_loss(all_predictions, true_states[:, 1:, :])

        
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        obs_window, true_states = batch 
        seq_len = true_states.shape[1]

        # 1. Initialize the Particle Cloud at t=0
        # Start with the true state, expand it into N noisy particles
        x_start = true_states[:, 0, :].unsqueeze(1).expand(-1, self.num_particles, -1)
        initial_noise = torch.randn_like(x_start) * self.process_noise_std
        particles_curr = x_start + initial_noise
        
        predictions_list = []
        # 2. Autoregressively unroll the predictions
        for t in range(1, seq_len):
            # Grab JUST the exact observation at time 't'
            current_obs = obs_window[:, t, :]
        
            # Predict the particle cloud at time 't'
            particles_pred = self(particles_prev=particles_curr, current_obs=current_obs)
            
            # Estimate the state by taking the mean of the particle cloud
            state_estimate = particles_pred.mean(dim=1)
            predictions_list.append(state_estimate)
            
            # The posterior cloud becomes the prior cloud for the next step!
            particles_curr = particles_pred 

        # 3. Stack and Calculate Loss
        all_predictions = torch.stack(predictions_list, dim=1)
        loss = torch.nn.functional.mse_loss(all_predictions, true_states[:, 1:, :])

        
        # Compare our predictions against the true states from t=1 onwards
        
        # Log the validation loss (prog_bar=True makes it visible in your terminal!)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)