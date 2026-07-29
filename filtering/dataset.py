from torch.utils.data import Dataset, DataLoader
import os
import torch
import pytorch_lightning as pl


# ─────────────────────────────────────────────────────────────────────────────
#  Uncontrolled dataset  (acoustic / NCV / any obs-only tracking task)
# ─────────────────────────────────────────────────────────────────────────────
class UncontrolledTrajectoryDataset(Dataset):
    """
    Yields fixed-length sliding-window BPTT batches from pre-normalised
    trajectories.  The mask is always all-True (no padding).

    stride=1          → dense overlap   (maximum windows, original behaviour)
    stride=seq_len//2 → 50 % overlap
    stride=seq_len    → non-overlapping windows

    Returns: (obs [T, obs_dim], states [T, state_dim], mask [T])
    """

    def __init__(self, measurements, true_states, seq_len=50, stride=1):
        # [N, Features, T] → [N, T, Features]
        self.measurements = measurements.transpose(1, 2)
        self.true_states  = true_states.transpose(1, 2)
        self.seq_len      = seq_len
        self.stride       = stride

        self.num_traj   = self.measurements.shape[0]
        self.time_steps = self.measurements.shape[1]

        self.windows_per_traj = max(
            1, (self.time_steps - self.seq_len) // self.stride + 1
        )
        self.total_samples = self.num_traj * self.windows_per_traj

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        traj_idx = idx // self.windows_per_traj
        time_idx = (idx % self.windows_per_traj) * self.stride

        x    = self.measurements[traj_idx, time_idx : time_idx + self.seq_len, :]
        y    = self.true_states [traj_idx, time_idx : time_idx + self.seq_len, :]
        mask = torch.ones(self.seq_len, dtype=torch.bool)
        return x.float(), y.float(), mask


# ─────────────────────────────────────────────────────────────────────────────
#  Controlled dataset  (Ackermann / any state+control tracking task)
# ─────────────────────────────────────────────────────────────────────────────
class ControlledTrajectoryDataset(Dataset):
    """
    Yields random BPTT windows from a pre-split dataset dict that includes
    control inputs alongside observations and states.

    The split dict must contain:
        states   : [N, T, state_dim]
        controls : [N, T, control_dim]
        obs      : [N, T, obs_dim]

    Returns: (obs [T, obs_dim], states [T, state_dim],
              controls [T, control_dim], mask [T])
    """

    def __init__(self, split: dict, seq_len: int):
        self.states   = split['states']    # [N, T, state_dim]
        self.controls = split['controls']  # [N, T, control_dim]
        self.obs      = split['obs']       # [N, T, obs_dim]
        self.T        = self.states.shape[1]
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        max_start = self.T - self.seq_len
        start = torch.randint(0, max_start + 1, (1,)).item()
        end   = start + self.seq_len

        obs      = self.obs[idx,      start:end, :]
        states   = self.states[idx,   start:end, :]
        controls = self.controls[idx, start:end, :]
        mask     = torch.ones(self.seq_len, dtype=torch.bool)
        return obs.float(), states.float(), controls.float(), mask


# ─────────────────────────────────────────────────────────────────────────────
#  Unified DataModule
# ─────────────────────────────────────────────────────────────────────────────
class FileTrackingDataModule(pl.LightningDataModule):
    """
    Single DataModule for both uncontrolled (acoustic) and controlled
    (Ackermann) tracking tasks.

    use_control=False
        Expects a .pt file with keys:
            true_trajectories : [N, state_dim, T]
            measurements      : [N, obs_dim,   T]
        Builds UncontrolledTrajectoryDataset with sliding windows.
        Normalisation stats are computed on the training split only.
        DataLoader yields (obs, states, mask).

    use_control=True
        Expects a .pt file with keys:
            train / val / test : dicts of {states, controls, obs}
            norm_stats         : pre-computed statistics (stored as metadata)
            config             : dict with state_dim / obs_dim / control_dim
        Builds ControlledTrajectoryDataset with random BPTT windows.
        DataLoader yields (obs, states, controls, mask).

    """

    def __init__(self,
                 data_path: str,
                 seq_len: int    = 100,
                 stride: int     = 1,
                 batch_size: int = 64,
                 dataset_type: str    = 'acoustic',
                 use_control: bool    = False,
                 use_minmax_norm: bool = False,
                 normalize: bool = True):
        """
        normalize=True (default) — uncontrolled path only (use_control=True
            already applies no normalisation regardless, see
            _setup_controlled's docstring): train_dataset/val_dataset yield
            z-scored (obs, states, mask), the original behaviour.
        normalize=False — train_dataset/val_dataset yield raw, physical-
            unit (obs, states, mask) instead. t_mean/t_std/m_mean/m_std are
            still computed and stored exactly as before either way (they're
            needed downstream regardless, e.g. a model that wants to work in
            physical units for part of its computation and normalised units
            for another part -- see model_known_node.py).
        """
        super().__init__()
        self.data_path         = data_path
        self.seq_len           = seq_len
        self.stride            = stride
        self.batch_size        = batch_size
        self.dataset_type      = dataset_type
        self.use_control       = use_control
        self.use_minmax_norm   = use_minmax_norm
        self.normalize         = normalize

        self.state_dim   = None
        self.obs_dim     = None
        self.control_dim = 0       # overwritten in setup() for controlled mode
        self.correction_scale = None  # set in setup(); [state_dim], Eq. 13/14 "yscale" analog

    # ── Setup ─────────────────────────────────────────────────────────────────
    def setup(self, stage=None):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Cannot find '{self.data_path}'. "
                "Run the data-generation script first."
            )

        if self.use_control:
            self._setup_controlled()
        else:
            self._setup_uncontrolled()

    def _setup_controlled(self):
        raw = torch.load(self.data_path, weights_only=False)

        self.norm_stats  = raw['norm_stats']
        dataset_cfg      = raw['config']

        self.state_dim   = dataset_cfg['state_dim']
        self.obs_dim     = dataset_cfg['obs_dim']
        self.control_dim = dataset_cfg['control_dim']

        train_split = raw['train']
        val_split   = raw['val']
        test_split  = raw.get('test', None)

        self.train_dataset = ControlledTrajectoryDataset(train_split, self.seq_len)
        self.val_dataset   = ControlledTrajectoryDataset(val_split,   self.seq_len)
        if test_split is not None:
            self.test_dataset = ControlledTrajectoryDataset(test_split, self.seq_len)

        # Eq. 13/14 "yscale" analog: typical magnitude of consecutive-timestep
        # state change, computed on the (already-normalised) training states.
        train_states = train_split['states']  # [N, T, state_dim]
        deltas    = train_states[:, 1:, :] - train_states[:, :-1, :]
        delta_min = deltas.amin(dim=(0, 1))
        delta_max = deltas.amax(dim=(0, 1))
        scale = delta_max - delta_min
        self.correction_scale = torch.where(
            scale == 0, torch.ones_like(scale), scale)

        print(f"  Loaded (controlled) from: {self.data_path}")
        print(f"    Train: {len(self.train_dataset)} trajectories")
        print(f"    Val:   {len(self.val_dataset)} trajectories")
        print(f"    state_dim={self.state_dim}  "
              f"obs_dim={self.obs_dim}  "
              f"control_dim={self.control_dim}  "
              f"seq_len={self.seq_len}")

    def _compute_norm_stats(self, data: torch.Tensor):
        """
        Compute per-channel (shift, scale) over dims (0, 2) of data [N, C, T],
        such that normalised = (data - shift) / scale.

        use_minmax_norm=False (default) — z-score: shift=mean, scale=std.
        use_minmax_norm=True  — min-max:  shift=min,  scale=max-min, matching
            yscale = ymax - ymin (Eq. 13, arXiv:2103.15341), consistent with
            the equation-scaling technique.
        """
        if self.use_minmax_norm:
            shift = data.amin(dim=(0, 2), keepdim=True)
            scale = data.amax(dim=(0, 2), keepdim=True) - shift
        else:
            shift = data.mean(dim=(0, 2), keepdim=True)
            scale = data.std(dim=(0, 2), keepdim=True)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return shift, scale

    def _setup_uncontrolled(self):
        print(f"Loading '{self.dataset_type}' data from {self.data_path} ...")
        loaded = torch.load(self.data_path, weights_only=False, map_location='cpu')

        true_traj = loaded['true_trajectories']   # [N, state_dim, T]
        raw_meas  = loaded['measurements']         # [N, obs_dim,   T]

        self.num_samples = true_traj.shape[0]
        self.state_dim   = true_traj.shape[1]
        self.obs_dim     = raw_meas.shape[1]

        # Train / val split before computing any statistics (no leakage)
        split_idx = int(self.num_samples * 0.8)

        # State normalisation on training split only
        train_states = true_traj[:split_idx]
        t_mean, t_std = self._compute_norm_stats(train_states)
        self.t_mean = t_mean
        self.t_std  = t_std
        true_traj_norm = (true_traj - t_mean) / t_std

        # Eq. 13/14 "yscale" analog for the ODE flow: typical magnitude of
        # consecutive-timestep state change, in the model's normalised space.
        # Pure data statistic (training split only, no leakage, no model
        # dependency) — the delta-based counterpart of the paper's ymax-ymin.
        train_norm = true_traj_norm[:split_idx]
        deltas     = train_norm[..., 1:] - train_norm[..., :-1]
        delta_min  = deltas.amin(dim=(0, 2))
        delta_max  = deltas.amax(dim=(0, 2))
        scale = delta_max - delta_min
        self.correction_scale = torch.where(
            scale == 0, torch.ones_like(scale), scale)

        train_meas = raw_meas[:split_idx]
        m_mean, m_std = self._compute_norm_stats(train_meas)
        self.m_mean = m_mean
        self.m_std  = m_std
        meas_norm = (raw_meas - m_mean) / m_std

        # Store any non-tensor metadata from the file (e.g. acoustic simulation
        # params: nSensor, simAreaSize, Amp, invPow, d0) under norm_stats so
        # known_measurement=True configs can access them via dm.norm_stats,
        # keeping the same interface as the controlled path.
        self.norm_stats = {
            k: v for k, v in loaded.items()
            if k not in ('true_trajectories', 'measurements')
            and not isinstance(v, torch.Tensor)
        }

        train_meas_ds  = meas_norm[:split_idx]      if self.normalize else raw_meas[:split_idx]
        val_meas_ds    = meas_norm[split_idx:]      if self.normalize else raw_meas[split_idx:]
        train_state_ds = true_traj_norm[:split_idx] if self.normalize else true_traj[:split_idx]
        val_state_ds   = true_traj_norm[split_idx:] if self.normalize else true_traj[split_idx:]

        self.train_dataset = UncontrolledTrajectoryDataset(
            train_meas_ds, train_state_ds, self.seq_len, self.stride
        )
        self.val_dataset = UncontrolledTrajectoryDataset(
            val_meas_ds, val_state_ds, self.seq_len, self.stride
        )

        print(f"  Data loaded successfully.")
        print(f"    Dataset type: {self.dataset_type}")
        print(f"    seq_len: {self.seq_len}  | stride: {self.stride}")
        print(f"    state_dim: {self.state_dim}  | obs_dim: {self.obs_dim}")
        print(f"    normalize: {self.normalize}")
        print(f"    Train windows: {len(self.train_dataset)}  "
              f"| Val windows: {len(self.val_dataset)}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=4, persistent_workers=False, pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=4, persistent_workers=False, pin_memory=False
        )

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            return None
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=4, persistent_workers=False, pin_memory=False
        )