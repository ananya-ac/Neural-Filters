import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import FileTrackingDataModule
from model import ModularNeuralODEFilter
import torch
import pytorch_lightning as pl
import argparse

if __name__ == "__main__":
    # 1. Initialize Data and Model
     
    # dm = TrackingDataModule(num_trajectories=1000, seq_len=50, batch_size=256)
    # model = ModularNeuralODEFilter()
    parser = argparse.ArgumentParser(description="Train NPFF with varying sequence lengths.")
    parser.add_argument('--seq_len', type=int, default=10, help='Length of the BPTT horizon.')
    args = parser.parse_args()
    seq_len = args.seq_len
    print(f"--- Starting run with seq_len: {seq_len} ---")

    # dm = FileTrackingDataModule(
    # data_path='acoustic_tracking_data.pt', 
    # seq_len=seq_len, 
    # batch_size=256
    # )
    dm = FileTrackingDataModule(
    data_path='acoustic_tracking_data_less_proc_noise.pt', 
    seq_len=seq_len, 
    batch_size=256
    )
    dm.setup() # Call this manually so dm.state_dim and dm.obs_dim are populated

    # 2. Feed the dynamic dimensions directly into the Neural ODE
    model = ModularNeuralODEFilter(
        state_dim=dm.state_dim,  
        obs_dim=dm.obs_dim,      
        context_dim=256, 
        hidden_dim=256,
        num_particles=25,        
        lr=1e-4,
        process_noise_std=0.01     
    )

    # 2. Callbacks for automatic saving and stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best-ode-filter')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # 3. Train the Model
    trainer = pl.Trainer(
        max_epochs=30,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0, # Keeps ODE solver stable
        accelerator='auto',    # Automatically uses GPU if available
        devices=1
    )
    
    print("Starting training...")
    trainer.fit(model, datamodule=dm)

    
    # 4. Evaluation and Plotting Phase
    # print("Evaluating best model on a validation trajectory...")
    
    # # FIX 1: Load using the correct class name!
    # best_model = ModularNeuralODEFilter.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # best_model = best_model.to(device)
    # best_model.eval()

    # traj_idx = 0
    # seq_len = dm.seq_len
    # total_steps = dm.val_meas_norm.shape[2]
    
    # meas_seq = dm.val_meas_norm[traj_idx].transpose(0, 1)  
    # true_seq_norm = dm.val_true_norm[traj_idx].transpose(0, 1)

    # predictions = torch.zeros((total_steps, 6))
    # predictions[:seq_len] = true_seq_norm[:seq_len] 

    # # FIX 2: We need a starting physics state to kick off the ODE!
    # # Grab the true state at the end of the warm-up period.
    # x_curr = true_seq_norm[seq_len - 1].unsqueeze(0).float().to(device)

    # with torch.no_grad():
    #     for t in range(seq_len, total_steps):
    #         window = meas_seq[t - seq_len + 1 : t + 1].unsqueeze(0).float().to(device)
            
    #         # FIX 3: Pass both the previous state AND the sensor window
    #         pred_state = best_model(x_prev=x_curr, obs_seq=window)
            
    #         # Save the prediction
    #         predictions[t] = pred_state.squeeze(0).cpu()
            
    #         # AUTO-REGRESSIVE UPDATE: 
    #         # The model's current prediction becomes x_curr for the next timestep!
    #         x_curr = pred_state

    # # Un-normalize for plotting
    # t_mean = dm.t_mean.squeeze()
    # t_std = dm.t_std.squeeze()

    # predictions_physical = (predictions * t_std) + t_mean
    # true_physical = (true_seq_norm * t_std) + t_mean

    # # Plot spatial trajectory
    # plt.figure(figsize=(10, 8))
    # plt.plot(true_physical[:, 0].numpy(), true_physical[:, 1].numpy(), label='True Trajectory', color='blue', linewidth=2)
    # plt.plot(predictions_physical[seq_len:, 0].numpy(), predictions_physical[seq_len:, 1].numpy(), 
    #          label='Modular Neural ODE Prediction', color='red', linestyle='--', linewidth=2)
    
    # plt.scatter([10, 30], [0, 0], marker='^', color='green', s=150, label='Sensors', zorder=5)
    # plt.title('Modular Latent ODE Filter: Trajectory')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    # plt.savefig('eval_traj.png')

    # print("Evaluating best model on a validation trajectory...")
    
    # # FIX: You must load the weights from the Lightning checkpoint!
    # best_model = ModularNeuralODEFilter.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    # best_model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # best_model.to(device)

    
    # state_dim = dm.state_dim
    
    # # Grab the un-normalization stats once outside the loop
    # t_mean = dm.t_mean.squeeze().cpu()
    # t_std = dm.t_std.squeeze().cpu()
    
    # # Initialize the figure outside the loop
    # plt.figure(figsize=(10, 8))
    
    # # Select 5 random trajectory indices from the validation set
    # num_val_trajs = dm.val_dataset.measurements.shape[0]
    # random_indices = torch.randperm(num_val_trajs)[:5]
    
    # for i, traj_idx in enumerate(random_indices):
        
    #     # Access the pre-transposed validation dataset: shape [Time, Features]
    #     meas_seq = dm.val_dataset.measurements[traj_idx].to(device)  
    #     true_seq_norm = dm.val_dataset.true_states[traj_idx].to(device)
        
    #     total_steps = meas_seq.shape[0]
    #     warmup_steps = dm.seq_len 
        
    #     # Only print the warning once
    #     if i == 0 and warmup_steps >= total_steps:
    #         print(f"⚠️ Warning: Your seq_len ({warmup_steps}) is equal to or larger than "
    #               f"the trajectory length ({total_steps}). The model will not make any "
    #               f"autoregressive predictions. Lower seq_len or generate longer data!")

    #     # Initialize prediction storage dynamically based on state_dim
    #     predictions = torch.zeros((total_steps, state_dim))
    #     predictions[:warmup_steps] = true_seq_norm[:warmup_steps].cpu() 

    #     # Grab the true state at the end of the warm-up period to kick off the ODE
    #     x_curr = true_seq_norm[warmup_steps - 1].unsqueeze(0).float()

    #     with torch.no_grad():
    #         for t in range(warmup_steps, total_steps):
                
    #             # Since t >= seq_len, this cleanly slices exactly 'seq_len' steps
    #             start_idx = t + 1 - dm.seq_len
    #             window = meas_seq[start_idx : t + 1].unsqueeze(0).float()
                
    #             # Predict the next step
    #             pred_state = best_model(x_prev=x_curr, obs_seq=window)
                
    #             # Save prediction
    #             predictions[t] = pred_state.squeeze(0).cpu()
                
    #             # AUTO-REGRESSIVE UPDATE:
    #             x_curr = pred_state

    #     # Un-normalize
    #     predictions_physical = (predictions * t_std) + t_mean
    #     true_physical = (true_seq_norm.cpu() * t_std) + t_mean
        
    #     # --- Plotting the Trajectories ---
    #     # We only add a label on the very first iteration (i==0) to avoid legend clutter
        
    #     # 1. True Trajectory (added slight transparency (alpha) to easily see overlaps)
    #     plt.plot(true_physical[:, 0].numpy(), true_physical[:, 1].numpy(), 
    #              label='True Trajectory' if i == 0 else "", color='blue', linewidth=2, alpha=0.5)
                 
    #     # 2. Warmup Phase (Truth fed into the model)
    #     plt.plot(predictions_physical[:warmup_steps, 0].numpy(), predictions_physical[:warmup_steps, 1].numpy(), 
    #              label='Warmup Phase' if i == 0 else "", color='gray', linestyle=':', linewidth=4, zorder=3)

    #     # 3. Auto-Regressive Prediction 
    #     if warmup_steps < total_steps:
    #         plt.plot(predictions_physical[warmup_steps-1:, 0].numpy(), predictions_physical[warmup_steps-1:, 1].numpy(), 
    #                  label='Auto-Regressive Prediction' if i == 0 else "", color='red', linestyle='--', linewidth=2, zorder=4)
    
    # # Add Sensors outside the loop so they only plot once
    # sensors_x = [10, 10, 70, 70]
    # sensors_y = [10, 70, 10, 70]
    # plt.scatter(sensors_x, sensors_y, marker='^', color='green', s=150, label='Sensors', zorder=5)
    
    # plt.title(f'Modular Latent ODE Filter (Dim: {state_dim}) - 5 Random Trajectories')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    # plt.savefig('eval_traj.png')
    # print("Saved evaluation plot to 'eval_traj.png'!")



    print("Evaluating best model on a validation trajectory...")
    best_model = ModularNeuralODEFilter.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    state_dim = dm.state_dim
    num_particles = 50 #best_model.num_particles
    
    # Grab the un-normalization stats
    t_mean = dm.t_mean.squeeze().cpu()
    t_std = dm.t_std.squeeze().cpu()
    
    plt.figure(figsize=(10, 8))
    
    # Select exactly 1 random trajectory index
    num_val_trajs = dm.val_dataset.measurements.shape[0]
    traj_idx = torch.randint(0, num_val_trajs, (1,)).item()
    
    meas_seq = dm.val_dataset.measurements[traj_idx].to(device)  
    true_seq_norm = dm.val_dataset.true_states[traj_idx].to(device)
    
    total_steps = meas_seq.shape[0]
    warmup_steps = dm.seq_len 
    
    if warmup_steps >= total_steps:
        print(f"⚠️ Warning: Your seq_len ({warmup_steps}) is equal to or larger than "
              f"the trajectory length ({total_steps}). The model will not make any "
              f"autoregressive predictions. Lower seq_len or generate longer data!")

    # Storage for the mean state estimate and the full particle cloud
    predictions = torch.zeros((total_steps, state_dim))
    predictions[:warmup_steps] = true_seq_norm[:warmup_steps].cpu() 
    
    particle_history = torch.zeros((total_steps, num_particles, state_dim))

    # Initialize the particle cloud at the end of the warmup period
    # Shape expands to: [Batch=1, NumParticles, StateDim]
    x_start = true_seq_norm[warmup_steps - 1].unsqueeze(0).unsqueeze(1).expand(-1, num_particles, -1).float()
    
    # Add a little noise to generate the initial cloud diversity
    initial_noise = torch.randn_like(x_start) * best_model.process_noise_std
    x_curr = x_start + initial_noise

    with torch.no_grad():
        for t in range(warmup_steps, total_steps):
            
            start_idx = t + 1 - dm.seq_len
            #window = meas_seq[start_idx : t + 1].unsqueeze(0).float()
            current_obs = meas_seq[t].unsqueeze(0).float()
                
                # Predict the new particle cloud
            particles_pred = best_model(particles_prev=x_curr, current_obs=current_obs)
            
            # Predict the new particle cloud
            #particles_pred = best_model(particles_prev=x_curr, obs_seq=window)
            
            # Estimate state as the mean of the cloud
            state_estimate = particles_pred.mean(dim=1).squeeze(0)
            
            # Save mean prediction and the raw particles
            predictions[t] = state_estimate.cpu()
            particle_history[t] = particles_pred.squeeze(0).cpu()
            
            # Update: the posterior cloud becomes the prior for the next step!
            x_curr = particles_pred

    # Un-normalize everything back to the physical scale
    predictions_physical = (predictions * t_std) + t_mean
    true_physical = (true_seq_norm.cpu() * t_std) + t_mean
    particle_history_physical = (particle_history * t_std) + t_mean
    
    # --- Plotting ---
    
    # 1. True Trajectory
    plt.plot(true_physical[:, 0].numpy(), true_physical[:, 1].numpy(), 
             label='True Trajectory', color='blue', linewidth=2, zorder=3)
             
    # 2. Warmup Phase (Truth fed into the model)
    plt.plot(predictions_physical[:warmup_steps, 0].numpy(), predictions_physical[:warmup_steps, 1].numpy(), 
             label='Warmup Phase', color='gray', linestyle=':', linewidth=4, zorder=4)

    # 3. Autoregressive Phase (Particles + Mean)
    if warmup_steps < total_steps:
        # Flatten the particle history for scatter plotting
        p_x = particle_history_physical[warmup_steps:, :, 0].numpy().flatten()
        p_y = particle_history_physical[warmup_steps:, :, 1].numpy().flatten()
        
        # Plot the particle cloud with high transparency (alpha=0.05). 
        # Overlapping particles will naturally form a darker, solid path where the model is confident!
        plt.scatter(p_x, p_y, color='red', alpha=0.05, s=15, label='Particle Cloud', zorder=1)
        
        # Plot Mean Prediction over the cloud
        plt.plot(predictions_physical[warmup_steps-1:, 0].numpy(), predictions_physical[warmup_steps-1:, 1].numpy(), 
                 label='Mean Prediction', color='darkred', linestyle='--', linewidth=2, zorder=5)
    
    # Sensors
    
    # sensors_x = [10, 10, 70, 70]
    # sensors_y = [10, 70, 10, 70]
    sensors_x = [5, 5, 95, 95]
    sensors_y = [5, 95, 5, 95]
    
    plt.scatter(sensors_x, sensors_y, marker='^', color='green', s=150, label='Sensors', zorder=6)
    
    plt.title(f'Neural Particle Flow Filter (Dim: {state_dim}, Particles: {num_particles})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f'eval_traj_{seq_len}.png')
    print(f"Saved evaluation plot to 'eval_traj_{seq_len}.png'!")