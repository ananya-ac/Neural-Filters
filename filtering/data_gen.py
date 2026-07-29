import torch
import math
from config import NPFFConfig


def place_sensors(num_sensors, sim_Area_size, device='cpu'):
    """
    Procedurally places sensors in concentric squares.
    Places up to 8 sensors per square (4 corners, 4 midpoints),
    shrinking the area by 0.75 for each subsequent inner square.
    """
    sensors = []
    
    # The center of the simulation area
    center = sim_Area_size / 2.0
    
    # Initial area is the full simulation square
    current_area = float(sim_Area_size ** 2)
    
    while len(sensors) < num_sensors:
        # Calculate the side length of the current square
        L = math.sqrt(current_area)
        half_L = L / 2.0
        
        # Calculate min and max bounds for this square
        min_c = center - half_L
        max_c = center + half_L
        
        # 1) Corners (Ends of the square)
        corners = [
            [min_c, min_c],  # Bottom-Left
            [min_c, max_c],  # Top-Left
            [max_c, min_c],  # Bottom-Right
            [max_c, max_c]   # Top-Right
        ]
        
        # 2) Midpoints of each line
        midpoints = [
            [center, min_c], # Bottom edge midpoint
            [center, max_c], # Top edge midpoint
            [min_c, center], # Left edge midpoint
            [max_c, center]  # Right edge midpoint
        ]
        
        # Combine the placement candidates for this square level
        candidates = corners + midpoints
        
        # Add them to our main list until we hit the requested number
        for pos in candidates:
            if len(sensors) < num_sensors:
                sensors.append(pos)
            else:
                break
                
        # 3) Shrink the area for the next concentric square
        current_area *= 0.75
        
    return torch.tensor(sensors, dtype=torch.float32, device=device)

def setup_parameters():
    """Initializes the parameter structure using PyTorch tensors."""
    cfg = NPFFConfig().data
    
    ps = {
        'device': cfg.device,
        'T': cfg.T,
        'nTarget': cfg.nTarget,
        'nSensor': cfg.nSensor,
        'simAreaSize': cfg.simAreaSize,
        'dimState_all': cfg.nTarget * 4,
        'Amp': cfg.Amp,
        'invPow': cfg.invPow,
        'd0': cfg.d0,
        'measvar_real': cfg.measvar_real,
        'glitch':cfg.glitch
    }
    device = cfg.device

    ps['sensorsPos'] = place_sensors(
        num_sensors=cfg.nSensor, 
        sim_Area_size=cfg.simAreaSize, 
        device=cfg.device
    )

    # Propagation parameters (Nearly Constant Velocity model)
    dt = cfg.dt
   

    phi_single = torch.tensor([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ], dtype=torch.float32, device=device)


    # Full block diagonal transition matrix
    ps['Phi'] = torch.kron(torch.eye(ps['nTarget'], device=device), phi_single)

    # Process noise covariance (Q)
    q_single = torch.tensor([
        [dt**3/3, 0,       dt**2/2, 0      ],
        [0,       dt**3/3, 0,       dt**2/2],
        [dt**2/2, 0,       dt,      0      ],
        [0,       dt**2/2, 0,       dt     ]
    ], dtype=torch.float32, device=device) * cfg.process_noise_scale

    ps['Q'] = torch.kron(torch.eye(ps['nTarget'], device=device), q_single)

    # Precompute Cholesky decomposition of Q for fast batched noise generation
    ps['Q_chol'] = torch.linalg.cholesky(ps['Q'])
    ps['noise_distribution'] = cfg.noise_distribution #'laplace'

    return ps

def acoustic_propagate_batched(x_prev, Phi, Q_chol, noise_distribution):
    """Batched propagation with support for non-Gaussian heavy-tailed noise."""
    # 1. Deterministic Physics Step
    mean = x_prev @ Phi.T

    # # 2. Sample the Base Noise
    # if noise_distribution == 'laplace':
    #     # Laplace creates heavy tails: lots of small noise, but occasional massive spikes (gusts/currents)
    #     # We scale by 1/sqrt(2) so the overall variance roughly matches the Gaussian baseline
    #     laplace_dist = torch.distributions.laplace.Laplace(0.0, 1.0 / math.sqrt(2.0))
    #     base_noise = laplace_dist.sample(mean.shape).to(mean.device)
        
    # else:
    #     # Standard Normal Gaussian
    base_noise = torch.randn_like(mean)

    # 3. Shape the noise using the physics kinematics (Q_chol)
    noise = base_noise @ Q_chol.T

    return mean + noise

def generate_tracks_batched(ps, num_tracks=1000):
    device, dim, T = ps['device'], ps['dimState_all'], ps['T']

    valid_tracks = []
    needed = num_tracks

    lower_bound = 0.01 * ps['simAreaSize'] 
    upper_bound = 0.99 * ps['simAreaSize']
    noise_distribution = ps['noise_distribution']

    # We generate in chunks to avoid running out of memory while rejecting
    chunk_size = 5000 

    while needed > 0:
        current_batch = min(needed * 2, chunk_size) # Over-generate to account for rejections
        x_traj = torch.zeros((current_batch, dim, T), device=device)

        # Initialize starting positions with high spatial diversity and real momentum!
        for i in range(ps['nTarget']):
            # 1. Spawn anywhere in a safe inner boundary of the 100x100 map
            x_traj[:, i*4,   0].uniform_(10.0, 80.0) 
            x_traj[:, i*4+1, 0].uniform_(10.0, 80.0) 
            
            # 2. Give the target a definitive initial push
            x_traj[:, i*4+2, 0].uniform_(-1.0, 1.0)  
            x_traj[:, i*4+3, 0].uniform_(-1.0, 1.0)
        x_traj[:, :, 0] = acoustic_propagate_batched(x_traj[:, :, 0], ps['Phi'], ps['Q_chol'], noise_distribution=noise_distribution)

        # Sequentially propagate the batch through time
        for t in range(1, T):
            x_traj[:, :, t] = acoustic_propagate_batched(x_traj[:, :, t-1], ps['Phi'], ps['Q_chol'], noise_distribution=noise_distribution)

        # Check boundaries for the whole batch
        xx = x_traj[:, 0::4, :] # shape: (batch_size, nTarget, T)
        yy = x_traj[:, 1::4, :] 

        # Boolean masks of out-of-bounds states
        out_x = (xx < lower_bound) | (xx > upper_bound)
        out_y = (yy < lower_bound) | (yy > upper_bound)

        # If any target at any time is out of bounds, the trajectory is invalid
        bad_mask = (out_x | out_y).any(dim=1).any(dim=1) 
        good_mask = ~bad_mask
        # Keep only the valid ones
        good_tracks = x_traj[good_mask]
        if len(good_tracks) > 0:
            valid_tracks.append(good_tracks)
            needed -= len(good_tracks)
            print(f"Accepted {len(good_tracks)} tracks. Still need {max(0, needed)}...")

    # Concatenate all valid batches and return exactly 'num_tracks'
    return torch.cat(valid_tracks, dim=0)[:num_tracks]


def generate_measurements_batched(x_traj, ps):
    """Generates sensor measurements for an entire batch of trajectories."""
    device = ps['device']
    batch_size = x_traj.shape[0]
    T = ps['T']
    nSensor = ps['nSensor']

    # Extract positions: shape (batch_size, nTarget, T)
    xx = x_traj[:, 0::4, :]
    yy = x_traj[:, 1::4, :]

    z = torch.zeros((batch_size, nSensor, T), device=device)

    # Calculate measurements for each sensor across all targets and batches
    for s in range(nSensor):
        sx = ps['sensorsPos'][s, 0]
        sy = ps['sensorsPos'][s, 1]

        # Distance squared: shape (batch_size, nTarget, T)
        dist_sq = (xx - sx)**2 + (yy - sy)**2
        dist = torch.sqrt(dist_sq)

        # Sum acoustic signal across targets: shape (batch_size, T)
        signal_strength = torch.sum(ps['Amp'] / (dist**ps['invPow'] + ps['d0']), dim=1)

        # Add Gaussian noise
        noise_std = torch.sqrt(torch.tensor(ps['measvar_real'], device=device))
        noise = noise_std * torch.randn((batch_size, T), device=device)
        glitch_probability = ps['glitch']
        glitch_mask = torch.rand_like(signal_strength) < glitch_probability
        glitch_probability = ps['glitch']   # e.g. 0.01-0.05
        glitch_mask = torch.rand_like(signal_strength) < glitch_probability

        # Type of glitch — 50% dropout, 50% interference spike
        
        # Dropout — sensor returns near-zero
        signal_strength[glitch_mask] = torch.zeros_like(
            signal_strength[glitch_mask]) + noise_std * torch.randn_like(
            signal_strength[glitch_mask]).abs()

        
        z[:, s, :] = signal_strength + noise

    return z



def generate_bearing_data(num_trajectories=1000, total_iter=1000, dt=0.01, save_path='bearing_tracking_data.pt'):
    dtype = torch.float32
    dim = 6
    
    # Dynamics Matrices
    A = torch.zeros((dim, dim), dtype=dtype)
    A[0, 2] = 1.0; A[1, 3] = 1.0
    A[2, 4] = 1.0; A[3, 5] = 1.0

    B = torch.zeros((dim, 2), dtype=dtype)
    B[4, 0] = 1.0; B[5, 1] = 1.0

    R = torch.diag(torch.tensor([0.001**2, 0.001**2], dtype=dtype))
    W = torch.diag(torch.tensor([0.001**2, 0.001**2], dtype=dtype))

    # Sensor Locations
    s1 = torch.tensor([[10.0], [0.0]], dtype=dtype)
    s2 = torch.tensor([[30.0], [0.0]], dtype=dtype)

    base_x0 = torch.tensor([[5.0], [20.0], [0.0], [0.0], [2.0], [0.0]], dtype=dtype)

    # Pre-compute discrete transitions
    A1 = torch.eye(dim, dtype=dtype) + (dt * A) + (0.5 * dt**2 * (A @ A))
    noise_scale = math.sqrt(dt) * B @ torch.sqrt(W)

    all_xtrue = torch.zeros((num_trajectories, dim, total_iter), dtype=dtype)
    all_meas = torch.zeros((num_trajectories, 2, total_iter), dtype=dtype)

    print(f"Generating {num_trajectories} trajectories...")

    for traj_idx in range(num_trajectories):
        # Add random perturbation for dataset diversity
        perturbation = torch.tensor([[2.0], [2.0], [0.5], [0.5], [0.1], [0.1]], dtype=dtype) * torch.randn((dim, 1), dtype=dtype)
        
        xtrue = torch.zeros((dim, total_iter), dtype=dtype)
        xtrue[:, 0:1] = base_x0 + perturbation
        
        # 1. Propagate states
        for ii in range(1, total_iter):
            process_noise = torch.randn((2, 1), dtype=dtype)
            xtrue[:, ii:ii+1] = (A1 @ xtrue[:, ii-1:ii]) + (noise_scale @ process_noise)
            
        # 2. Vectorized Measurements
        ps1 = xtrue[0:2, :] - s1
        ps2 = xtrue[0:2, :] - s2
        
        meas_clean = torch.stack([
            torch.atan2(ps1[1, :], ps1[0, :]),
            torch.atan2(ps2[1, :], ps2[0, :])
        ], dim=0)
        
        sensor_noise = torch.sqrt(R) @ torch.randn((2, total_iter), dtype=dtype)
        meas = meas_clean + sensor_noise
        
        all_xtrue[traj_idx] = xtrue
        all_meas[traj_idx] = meas

    print("Calculating statistics...")
    x_mean = all_xtrue.mean(dim=(0, 2), keepdim=True)
    x_std = all_xtrue.std(dim=(0, 2), keepdim=True) + 1e-8
    
    z_mean = all_meas.mean(dim=(0, 2), keepdim=True)
    z_std = all_meas.std(dim=(0, 2), keepdim=True) + 1e-8

    # Package into dictionary
    data_to_save = {
        'true_trajectories': all_xtrue,
        'measurements': all_meas,
        'x_mean': x_mean,
        'x_std': x_std,
        'z_mean': z_mean,
        'z_std': z_std
    }

    torch.save(data_to_save, save_path)
    print(f"Data and statistics successfully saved to '{save_path}'!")
    print(f" -> States Shape: {all_xtrue.shape}")
    print(f" -> Obs Shape: {all_meas.shape}")

    

def main():
    sensor_type = 'acoustic'
    if sensor_type == 'acoustic':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # 1. Setup
        ps = setup_parameters()

        # 2. Generate Data 
        num_samples = 10000
        print(f"Generating {num_samples} batched trajectories...")
        true_trajectories = generate_tracks_batched(ps, num_tracks=num_samples)

        print(f"Generating batched measurements...")
        measurements = generate_measurements_batched(true_trajectories, ps)

        print(f"Generated states shape: {true_trajectories.shape}") 
        print(f"Generated meas shape:   {measurements.shape}")      

        # --- 3. Normalization Step ---
        print("Calculating statistics and normalizing data...")
        x_mean = true_trajectories.mean(dim=(0, 2), keepdim=True)
        x_std = true_trajectories.std(dim=(0, 2), keepdim=True) + 1e-6

        z_mean = measurements.mean(dim=(0, 2), keepdim=True)
        z_std = measurements.std(dim=(0, 2), keepdim=True) + 1e-6

        # Package all the necessary data and statistics into a dictionary
        data_to_save = {
            'true_trajectories': true_trajectories,
            'measurements': measurements,
            'x_mean': x_mean,
            'x_std': x_std,
            'z_mean': z_mean,
            'z_std': z_std
        }
        
        # Changed the output file so we don't overwrite the original Gaussian data
        data_path = NPFFConfig().train.data_path 
        
        # Save to disk
        torch.save(data_to_save, data_path)
        print(f"Data and normalization statistics successfully saved to {data_path}")
    
    else:
        generate_bearing_data()


if __name__ == '__main__':
    main()