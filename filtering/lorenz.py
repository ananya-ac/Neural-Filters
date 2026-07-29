import numpy as np
import torch
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt

def generate_lorenz_dataset(save_path='lorenz_tracking_data.pt', num_trajectories=10000, time_steps=100, dt=0.12):
    print(f"Generating {num_trajectories} Lorenz '63 trajectories...")
    
    # ---------------------------------------------------------
    # 1. System Parameters (From Section IV, Eq 49)
    # ---------------------------------------------------------
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    sensor_pos = np.array([6 * math.sqrt(2), 6 * math.sqrt(2), 27.0])
    R_cov = np.diag([0.1**2, 0.01**2, 0.01**2])
    
    # ---------------------------------------------------------
    # 2. Physics & Measurement Functions
    # ---------------------------------------------------------
    def lorenz_ode(t, x):
        return [
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ]
        
    def measure(x):
        r_vec = x - sensor_pos
        r = np.linalg.norm(r_vec)
        r = max(r, 1e-8) 
        
        alpha = np.arctan2(r_vec[1], r_vec[0])
        epsilon = np.arcsin(np.clip(r_vec[2] / r, -1.0, 1.0))
        return np.array([r, alpha, epsilon])

    # ---------------------------------------------------------
    # 3. Simulation Loop with Dense Output
    # ---------------------------------------------------------
    t_span = (0, time_steps * dt)
    
    # Coarse time steps for the actual Neural Net dataset (dt = 0.12)
    t_dataset = np.arange(0, time_steps * dt, dt)
    
    # Fine time steps purely for making the plot look smooth (dt = 0.01)
    t_plot = np.arange(0, time_steps * dt, 0.01)
    
    true_trajectories = np.zeros((num_trajectories, 3, time_steps))
    measurements = np.zeros((num_trajectories, 3, time_steps))
    
    # We will save the high-resolution versions of the first two trajectories just for plotting
    high_res_plots = []
    
    for i in range(num_trajectories):
        x0 = np.random.multivariate_normal([0.0, 1.0, 0.0], np.eye(3))
        
        # dense_output=True allows us to evaluate the physics at ANY arbitrary time
        sol = solve_ivp(lorenz_ode, t_span, x0, dense_output=True, method='RK45')
        
        # 1. Extract the coarse states strictly for the dataset (at t = 0.0, 0.12, 0.24...)
        states_coarse = sol.sol(t_dataset)
        true_trajectories[i] = states_coarse
        
        # 2. Save the smooth, high-res version if it's one of the first two trajectories
        if i < 2:
            high_res_plots.append(sol.sol(t_plot))
        
        # Generate noisy measurements at the coarse discrete time steps
        for t in range(time_steps):
            clean_y = measure(states_coarse[:, t])
            noisy_y = clean_y + np.random.multivariate_normal(np.zeros(3), R_cov)
            measurements[i, :, t] = noisy_y
            
        if (i + 1) % 50 == 0:
            print(f"  -> Processed {i + 1}/{num_trajectories} trajectories")

    # ---------------------------------------------------------
    # 4. Format & Save Dataset
    # ---------------------------------------------------------
    print("Calculating dataset statistics...")
    true_trajectories_pt = torch.tensor(true_trajectories, dtype=torch.float32)
    measurements_pt = torch.tensor(measurements, dtype=torch.float32)
    
    x_mean = true_trajectories_pt.mean(dim=(0, 2), keepdim=True)
    x_std = true_trajectories_pt.std(dim=(0, 2), keepdim=True)
    z_mean = measurements_pt.mean(dim=(0, 2), keepdim=True)
    z_std = measurements_pt.std(dim=(0, 2), keepdim=True)
    
    data = {
        'true_trajectories': true_trajectories_pt,
        'measurements': measurements_pt,
        'x_mean': x_mean,
        'x_std': x_std,
        'z_mean': z_mean,
        'z_std': z_std
    }
    
    torch.save(data, save_path)
    print(f"Done! Dataset saved to '{save_path}'")

    # ---------------------------------------------------------
    # 5. Plot the High-Resolution Smooth Trajectories
    # ---------------------------------------------------------
    print("Plotting 2 smooth sample trajectories...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # We plot the high_res_plots, NOT the true_trajectories dataset arrays
    ax.plot(high_res_plots[0][0, :], high_res_plots[0][1, :], high_res_plots[0][2, :], 
            label='Trajectory 1', color='blue', alpha=0.8, linewidth=1.2)
            
    ax.plot(high_res_plots[1][0, :], high_res_plots[1][1, :], high_res_plots[1][2, :], 
            label='Trajectory 2', color='orange', alpha=0.8, linewidth=1.2)
            
    ax.scatter(sensor_pos[0], sensor_pos[1], sensor_pos[2], 
               color='red', marker='^', s=150, label='Sensor Location')
    
    ax.set_title("Lorenz '63 Continuous Dynamics (High Resolution)")
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_zlabel('$X_3$')
    ax.legend()
    ax.grid(True)
    
    plot_filename = 'lorenz_samples_3d_smooth.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Done! Saved smooth trajectory plot to '{plot_filename}'")

if __name__ == "__main__":
    generate_lorenz_dataset()