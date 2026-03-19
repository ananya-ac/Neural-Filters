from torch.utils.data import Dataset, DataLoader
from data_gen import generate_measurements_batched, generate_tracks_batched
import os
import torch
import math
import pytorch_lightning as pl

dtype = torch.float32 # Switched to float32 for Neural Network compatibility

# class MultiTrajectoryDataset(Dataset):
#     def __init__(self, measurements, true_states, seq_len=50):
#         self.measurements = measurements.transpose(1, 2) 
#         self.true_states = true_states.transpose(1, 2)
#         self.seq_len = seq_len
#         self.windows_per_traj = self.measurements.shape[1] - self.seq_len
#         self.total_samples = self.measurements.shape[0] * self.windows_per_traj

#     def __len__(self):
#         return self.total_samples

#     def __getitem__(self, idx):
#         traj_idx = idx // self.windows_per_traj
#         time_idx = idx % self.windows_per_traj
#         x = self.measurements[traj_idx, time_idx : time_idx + self.seq_len, :]
#         y = self.true_states[traj_idx, time_idx : time_idx + self.seq_len, :]
#         return x, y

# class TrackingDataModule(pl.LightningDataModule):
#     def __init__(self, num_trajectories=1000, seq_len=50, batch_size=256):
#         super().__init__()
#         self.num_traj = num_trajectories
#         self.seq_len = seq_len
#         self.batch_size = batch_size
        
#         # Will store normalization stats for un-normalizing later
#         self.t_mean = None
#         self.t_std = None

#     def setup(self, stage=None):
#         dim, dt, total_iter = 6, 0.01, 1000
        
#         A = torch.zeros((dim, dim), dtype=dtype)
#         A[0, 2] = A[1, 3] = A[2, 4] = A[3, 5] = 1.0
#         B = torch.zeros((dim, 2), dtype=dtype)
#         B[4, 0] = B[5, 1] = 1.0

#         R = torch.diag(torch.tensor([0.001**2, 0.001**2], dtype=dtype))
#         W = torch.diag(torch.tensor([0.001**2, 0.001**2], dtype=dtype))
#         s1 = torch.tensor([[10.0], [0.0]], dtype=dtype)
#         s2 = torch.tensor([[30.0], [0.0]], dtype=dtype)
#         base_x0 = torch.tensor([[5.0], [20.0], [0.0], [0.0], [2.0], [0.0]], dtype=dtype)

#         A1 = torch.eye(dim, dtype=dtype) + (dt * A) + (0.5 * dt**2 * (A @ A))
#         noise_scale = math.sqrt(dt) * B @ torch.sqrt(W)

#         all_xtrue = torch.zeros((self.num_traj, dim, total_iter), dtype=dtype)
#         all_meas = torch.zeros((self.num_traj, 2, total_iter), dtype=dtype)

#         # Generate Data
#         for traj_idx in range(self.num_traj):
#             perturbation = torch.tensor([[2.0], [2.0], [0.5], [0.5], [0.1], [0.1]], dtype=dtype) * torch.randn((dim, 1), dtype=dtype)
#             xtrue = torch.zeros((dim, total_iter), dtype=dtype)
#             xtrue[:, 0:1] = base_x0 + perturbation
            
#             for ii in range(1, total_iter):
#                 xtrue[:, ii:ii+1] = (A1 @ xtrue[:, ii-1:ii]) + (noise_scale @ torch.randn((2, 1), dtype=dtype))
                
#             ps1 = xtrue[0:2, :] - s1
#             ps2 = xtrue[0:2, :] - s2
            
#             meas_clean = torch.stack([torch.atan2(ps1[1, :], ps1[0, :]), torch.atan2(ps2[1, :], ps2[0, :])], dim=0)
#             all_meas[traj_idx] = meas_clean + (torch.sqrt(R) @ torch.randn((2, total_iter), dtype=dtype))
#             all_xtrue[traj_idx] = xtrue

#         # Split Data (80% Train, 20% Val)
#         split_idx = int(self.num_traj * 0.8)
#         train_meas, val_meas = all_meas[:split_idx], all_meas[split_idx:]
#         train_true, val_true = all_xtrue[:split_idx], all_xtrue[split_idx:]

#         # Calculate Statistics (Only on Train)
#         m_mean = train_meas.mean(dim=(0, 2), keepdim=True)
#         m_std = train_meas.std(dim=(0, 2), keepdim=True) + 1e-8
#         self.t_mean = train_true.mean(dim=(0, 2), keepdim=True)
#         self.t_std = train_true.std(dim=(0, 2), keepdim=True) + 1e-8

#         # Normalize Data
#         self.train_meas_norm = (train_meas - m_mean) / m_std
#         self.train_true_norm = (train_true - self.t_mean) / self.t_std
#         self.val_meas_norm = (val_meas - m_mean) / m_std
#         self.val_true_norm = (val_true - self.t_mean) / self.t_std

#         # Datasets
#         self.train_dataset = MultiTrajectoryDataset(self.train_meas_norm, self.train_true_norm, self.seq_len)
#         self.val_dataset = MultiTrajectoryDataset(self.val_meas_norm, self.val_true_norm, self.seq_len)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)



# class UnifiedTrajectoryDataset(Dataset):
#     """Dynamically handles any trajectory length and yields sliding windows."""
#     def __init__(self, measurements, true_states, seq_len=50):
#         # Transpose from [Batch, Features, Time] to [Batch, Time, Features]
#         self.measurements = measurements.transpose(1, 2) 
#         self.true_states = true_states.transpose(1, 2)
#         self.seq_len = seq_len
        
#         self.num_traj = self.measurements.shape[0]
#         self.time_steps = self.measurements.shape[1]
        
#         # Calculate how many valid windows can fit in the trajectory
#         # Using max(1, ...) ensures that if time_steps < seq_len, it won't crash 
#         # (though you should avoid that scenario!)
#         self.windows_per_traj = max(1, (self.time_steps - self.seq_len) + 1)
#         self.total_samples = self.num_traj * self.windows_per_traj

#     def __len__(self):
#         return self.total_samples

#     def __getitem__(self, idx):
#         # Map the flat index to a specific trajectory and time window
#         traj_idx = idx // self.windows_per_traj
#         time_idx = idx % self.windows_per_traj
        
#         # Slice the sequence
#         x = self.measurements[traj_idx, time_idx : time_idx + self.seq_len, :]
#         y = self.true_states[traj_idx, time_idx : time_idx + self.seq_len, :]
        
#         return x.float(), y.float()



# class UnifiedTrackingDataModule(pl.LightningDataModule):
#     def __init__(self, data_path='acoustic_tracking_data.pt', seq_len=50, batch_size=256):
#         super().__init__()
#         self.data_path = data_path
#         self.seq_len = seq_len
#         self.batch_size = batch_size
        
#         # These will be populated dynamically in setup()
#         self.state_dim = None
#         self.obs_dim = None
        
#     def setup(self, stage=None):
#         if not os.path.exists(self.data_path):
#             raise FileNotFoundError(f"Cannot find {self.data_path}. Generate data first.")
            
#         print(f"Loading unified data from {self.data_path}...")
#         loaded_data = torch.load(self.data_path, weights_only=False)
        
#         # Extract variables using the keys defined in your notebook
#         true_traj = loaded_data['true_trajectories']
#         meas = loaded_data['measurements']
        
#         self.t_mean = loaded_data['x_mean']
#         self.t_std = loaded_data['x_std']
#         m_mean = loaded_data['z_mean']
#         m_std = loaded_data['z_std']
        
#         # Extract dimensions dynamically! 
#         # This ignores your notebook's ps['T'] config and trusts the actual tensor.
#         self.num_samples = true_traj.shape[0]
#         self.state_dim = true_traj.shape[1] 
#         self.obs_dim = meas.shape[1]        
#         self.raw_time_steps = true_traj.shape[2]   
        
#         # Normalize the data using the saved statistics
#         true_traj_norm = (true_traj - self.t_mean) / self.t_std
#         meas_norm = (meas - m_mean) / m_std
        
#         # Split 80/20 for Train/Validation
#         split_idx = int(self.num_samples * 0.8)
        
#         self.train_dataset = UnifiedTrajectoryDataset(
#             meas_norm[:split_idx], true_traj_norm[:split_idx], self.seq_len
#         )
#         self.val_dataset = UnifiedTrajectoryDataset(
#             meas_norm[split_idx:], true_traj_norm[split_idx:], self.seq_len
#         )
        
#         print(f"Framework Ready!")
#         print(f"  -> State Dim: {self.state_dim} | Obs Dim: {self.obs_dim}")
#         print(f"  -> Raw Trajectory Length: {self.raw_time_steps}")
#         print(f"  -> Unrolled Sequence Length: {self.seq_len}")
#         print(f"  -> Total Train Windows: {len(self.train_dataset)}")

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset, batch_size=self.batch_size, 
#             shuffle=True, num_workers=4, persistent_workers=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset, batch_size=self.batch_size, 
#             shuffle=False, num_workers=4, persistent_workers=True
#         )


class UnifiedTrajectoryDataset(Dataset):
    """Dynamically handles any trajectory length and yields sliding windows."""
    def __init__(self, measurements, true_states, seq_len=50):
        # Transpose from [Batch, Features, Time] to [Batch, Time, Features]
        self.measurements = measurements.transpose(1, 2) 
        self.true_states = true_states.transpose(1, 2)
        self.seq_len = seq_len
        
        self.num_traj = self.measurements.shape[0]
        self.time_steps = self.measurements.shape[1]
        
        # Calculate how many valid windows can fit in the trajectory
        self.windows_per_traj = max(1, (self.time_steps - self.seq_len) + 1)
        self.total_samples = self.num_traj * self.windows_per_traj

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        traj_idx = idx // self.windows_per_traj
        time_idx = idx % self.windows_per_traj
        
        x = self.measurements[traj_idx, time_idx : time_idx + self.seq_len, :]
        y = self.true_states[traj_idx, time_idx : time_idx + self.seq_len, :]
        
        return x.float(), y.float()


class FileTrackingDataModule(pl.LightningDataModule):
    def __init__(self, data_path='bearing_tracking_data.pt', seq_len=50, batch_size=256):
        super().__init__()
        self.data_path = data_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        # Will be populated dynamically
        self.state_dim = None
        self.obs_dim = None
        
    def setup(self, stage=None):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Cannot find {self.data_path}. Run data generation script first.")
            
        print(f"Loading pre-generated data from {self.data_path}...")
        loaded_data = torch.load(self.data_path, weights_only=False, map_location='cpu')

        true_traj = loaded_data['true_trajectories']
        meas = loaded_data['measurements']
        
        self.t_mean = loaded_data['x_mean']
        self.t_std = loaded_data['x_std']
        m_mean = loaded_data['z_mean']
        m_std = loaded_data['z_std']
        
        # Extract dimensions dynamically! 
        self.num_samples = true_traj.shape[0]
        self.state_dim = true_traj.shape[1] 
        self.obs_dim = meas.shape[1]        
        
        # Normalize the data using the saved statistics
        true_traj_norm = (true_traj - self.t_mean) / self.t_std
        meas_norm = (meas - m_mean) / m_std
        
        # Split 80/20 for Train/Validation
        split_idx = int(self.num_samples * 0.8)
        
        self.train_dataset = UnifiedTrajectoryDataset(
            meas_norm[:split_idx], true_traj_norm[:split_idx], self.seq_len
        )
        self.val_dataset = UnifiedTrajectoryDataset(
            meas_norm[split_idx:], true_traj_norm[split_idx:], self.seq_len
        )
        
        print(f"Data Loaded Successfully!")
        print(f"  -> State Dim: {self.state_dim} | Obs Dim: {self.obs_dim}")
        print(f"  -> Total Train Windows: {len(self.train_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=4, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=4, persistent_workers=True
        )