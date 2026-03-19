import torch
import math


def setup_parameters(device='cuda'):
    """Initializes the parameter structure using PyTorch tensors."""
    ps = {}
    ps['device'] = device

    # Setup params
    ps['T'] = 500           # Number of time steps
    ps['nTarget'] = 1      # Number of targets
    ps['nSensor'] = 4      # Number of sensors
    ps['simAreaSize'] = 100 # Size of the surveillance area

    ps['dimState_all'] = ps['nTarget'] * 4 

    # Acoustic Likelihood parameters
    ps['Amp'] = 200.0 #100
    ps['invPow'] = 1.0 #2
    ps['d0'] = 0.1
    ps['measvar_real'] = 0.01

    # Mock sensor positions
    # ps['sensorsPos'] = torch.tensor([
    #     [10.0, 10.0],
    #     [10.0, 70.0],
    #     [70.0, 10.0],
    #     [70.0, 70.0]
    # ], dtype=torch.float32, device=device)

    ps['sensorsPos'] = torch.tensor([
    [5.0, 5.0],
    [5.0, 95.0],
    [95.0, 5.0],
    [95.0, 95.0]
])

    # Propagation parameters (Nearly Constant Velocity model)
    dt = 0.1
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
    ], dtype=torch.float32, device=device) * 0.005 #0.05

    ps['Q'] = torch.kron(torch.eye(ps['nTarget'], device=device), q_single)

    # Precompute Cholesky decomposition of Q for fast batched noise generation
    # choles?
    ps['Q_chol'] = torch.linalg.cholesky(ps['Q'])

    return ps

def acoustic_propagate_batched(x_prev, Phi, Q_chol):
    """
    Batched propagation: 
    x_prev is shape (batch_size, dimState_all)
    """
    # Batched matrix multiplication: (B, 16) @ (16, 16).T -> (B, 16)
    mean = x_prev @ Phi.T

    # Generate batched multivariate noise using precomputed Cholesky factor
    noise =  torch.randn_like(mean) @ Q_chol.T

    return mean + noise

def generate_tracks_batched(ps, num_tracks=1000):
    device, dim, T = ps['device'], ps['dimState_all'], ps['T']

    valid_tracks = []
    needed = num_tracks

    lower_bound = 0.05 * ps['simAreaSize']
    upper_bound = 0.95 * ps['simAreaSize']

    # We generate in chunks to avoid running out of memory while rejecting
    chunk_size = 5000 

    while needed > 0:
        current_batch = min(needed * 2, chunk_size) # Over-generate to account for rejections
        x_traj = torch.zeros((current_batch, dim, T), device=device)

        # Initialize starting positions clustered near the center (simAreaSize is 40, center is 20)
        # Initialize starting positions with high spatial diversity and real momentum!
        for i in range(ps['nTarget']):
            # 1. Spawn anywhere in a safe inner boundary of the 100x100 map
            x_traj[:, i*4,   0].uniform_(40.0, 50.0) 
            x_traj[:, i*4+1, 0].uniform_(40.0, 50.0) 
            
            # 2. Give the target a definitive initial push
            x_traj[:, i*4+2, 0].uniform_(-1.0, 1.0)  
            x_traj[:, i*4+3, 0].uniform_(-1.0, 1.0)
        x_traj[:, :, 0] = acoustic_propagate_batched(x_traj[:, :, 0], ps['Phi'], ps['Q_chol'])

        # Sequentially propagate the batch through time
        for t in range(1, T):
            x_traj[:, :, t] = acoustic_propagate_batched(x_traj[:, :, t-1], ps['Phi'], ps['Q_chol'])

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
        ps = setup_parameters(device=device)

        # 2. Generate Data 
        num_samples = 100
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

        # We keep the original tensors for plotting, and create new _norm tensors for training
        true_traj_norm = (true_trajectories - x_mean) / x_std
        meas_norm = (measurements - z_mean) / z_std


        # Package all the necessary data and statistics into a dictionary
        data_to_save = {
            'true_trajectories': true_trajectories,
            'measurements': measurements,
            'x_mean': x_mean,
            'x_std': x_std,
            'z_mean': z_mean,
            'z_std': z_std
        }

        # Save to disk
        torch.save(data_to_save, 'acoustic_tracking_data_less_proc_noise.pt')
        print("Data and normalization statistics successfully saved to 'acoustic_tracking_data.pt'!")
    
    else:
        generate_bearing_data()


if __name__ == '__main__':
    main()
