import argparse
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataclasses import dataclass, field

from model_ackermann import ModularNeuralODEFilter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    device: str      = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path: str   = 'ackermann_lidar_dataset.pt'
    state_dim: int   = 4    # [x, y, theta, v]
    obs_dim: int     = 4    # LiDAR ranges to 4 landmarks
    control_dim: int = 2    # [delta, a]

@dataclass
class ModelConfig:
    num_particles:     int   = 100
    hidden_dim:        int   = 64
    noise_std:         float = 0.05
    integration_steps: int   = 4
    state_gru_dim:     int   = 16

@dataclass
class TrainConfig:
    lr:                float = 1e-4
    batch_size:        int   = 128
    num_epochs:        int   = 100
    bptt_seq_len:      int   = 40
    start_tbptt_k:     int   = 10
    curriculum_epochs: int   = 50

@dataclass
class AckermannConfig:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AckermannTrackingDataset(Dataset):
    """
    Wraps one split of the Ackermann LiDAR dataset.

    Each __getitem__ returns a random BPTT-length window:
        obs      : [bptt_seq_len, obs_dim]      LiDAR ranges (normalised)
        states   : [bptt_seq_len, state_dim]    (x, y, theta, v) (normalised)
        controls : [bptt_seq_len, control_dim]  (delta, a) (normalised)
        mask     : [bptt_seq_len]               all True
    """
    def __init__(self, split: dict, bptt_seq_len: int):
        self.states       = split['states']    # [N, T, 4]
        self.controls     = split['controls']  # [N, T, 2]
        self.obs          = split['obs']       # [N, T, 4]
        self.T            = self.states.shape[1]
        self.bptt_seq_len = bptt_seq_len

        
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        max_start = self.T - self.bptt_seq_len
        start = torch.randint(0, max_start + 1, (1,)).item()
        end   = start + self.bptt_seq_len

        obs      = self.obs[idx,      start:end, :]
        states   = self.states[idx,   start:end, :]
        controls = self.controls[idx, start:end, :]
        mask     = torch.ones(self.bptt_seq_len, dtype=torch.bool)

        return obs, states, controls, mask

# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class AckermannDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, bptt_seq_len: int, batch_size: int):
        super().__init__()
        self.data_path    = data_path
        self.bptt_seq_len = bptt_seq_len
        self.batch_size   = batch_size

        self.state_dim   = 4
        self.obs_dim     = 4
        self.control_dim = 2
        self.norm_stats  = None

    def setup(self, stage=None):
        raw = torch.load(self.data_path, weights_only=False)

        self.norm_stats  = raw['norm_stats']
        dataset_cfg      = raw['config']

        self.state_dim   = dataset_cfg['state_dim']
        self.obs_dim     = dataset_cfg['obs_dim']
        self.control_dim = dataset_cfg['control_dim']

        self.train_dataset = AckermannTrackingDataset(raw['train'], self.bptt_seq_len)
        self.val_dataset   = AckermannTrackingDataset(raw['val'],   self.bptt_seq_len)
        self.test_dataset  = AckermannTrackingDataset(raw['test'],  self.bptt_seq_len)

        print(f"  Dataset loaded from: {self.data_path}")
        print(f"    Train: {len(self.train_dataset)} trajectories")
        print(f"    Val:   {len(self.val_dataset)} trajectories")
        print(f"    Test:  {len(self.test_dataset)} trajectories")
        print(f"    state_dim={self.state_dim}  "
              f"obs_dim={self.obs_dim}  "
              f"control_dim={self.control_dim}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,   batch_size=self.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  batch_size=self.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)


# ---------------------------------------------------------------------------
# TBPTT Curriculum
# ---------------------------------------------------------------------------

class TBPTTCurriculum(pl.Callback):
    """
    Progressively increases the TBPTT truncation window from
    start_tbptt_k to max_tbptt_k over curriculum_epochs.
    bptt_seq_len stays fixed throughout.
    """
    def __init__(self, start_tbptt_k: int, max_tbptt_k: int,
                 curriculum_epochs: int):
        self.start_tbptt_k     = start_tbptt_k
        self.max_tbptt_k       = max_tbptt_k
        self.curriculum_epochs = curriculum_epochs
        self.step_size = max(1, (max_tbptt_k - start_tbptt_k) // curriculum_epochs)
        self._current_tbptt_k  = start_tbptt_k

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch >= self.curriculum_epochs:
            self._current_tbptt_k = self.max_tbptt_k
        else:
            self._current_tbptt_k = min(
                self.start_tbptt_k + epoch * self.step_size,
                self.max_tbptt_k
            )
        pl_module.tbptt_k = self._current_tbptt_k
        print(f"[TBPTT epoch {epoch}] tbptt_k={self._current_tbptt_k}", flush=True)
        pl_module.log('curriculum_tbptt_k', float(self._current_tbptt_k))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def config_to_dict(obj):
    if isinstance(obj, dict):
        return {k: config_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, '__dataclass_fields__'):
        import dataclasses
        return dataclasses.asdict(obj)
    elif hasattr(obj, '__dict__'):
        return {k: config_to_dict(v) for k, v in vars(obj).items()
                if not callable(v) and not k.startswith('__')}
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = AckermannConfig()

    parser = argparse.ArgumentParser(
        description="Train NODiF on the Ackermann LiDAR dataset.")

    # Data
    parser.add_argument('--data_path',         type=str,   default=cfg.data.data_path)

    # Model
    parser.add_argument('--num_particles',     type=int,   default=cfg.model.num_particles)
    parser.add_argument('--hidden_dim',        type=int,   default=cfg.model.hidden_dim)
    parser.add_argument('--noise_std',         type=float, default=cfg.model.noise_std)
    parser.add_argument('--integration_steps', type=int,   default=cfg.model.integration_steps)
    parser.add_argument('--state_gru_dim',     type=int,   default=cfg.model.state_gru_dim)
    parser.add_argument('--loss_type',         type=str,   default='l2',
                        choices=['l1', 'l2', 'mse', 'huber'])
    parser.add_argument('--integration_type',  type=str,   default='rk4',
                        choices=['rk4', 'euler'])
    parser.add_argument('--lam_schedule',      type=str,   default='exponential',
                        choices=['exponential', 'uniform'])

    # Training
    parser.add_argument('--lr',                type=float, default=cfg.train.lr)
    parser.add_argument('--batch_size',        type=int,   default=64)
    parser.add_argument('--num_epochs',        type=int,   default=cfg.train.num_epochs)
    parser.add_argument('--bptt_seq_len',      type=int,   default=cfg.train.bptt_seq_len)
    parser.add_argument('--start_tbptt_k',     type=int,   default=cfg.train.start_tbptt_k)
    parser.add_argument('--curriculum_epochs', type=int,   default=cfg.train.curriculum_epochs)

    # Logging
    parser.add_argument('--save_dir',          type=str,   default='',
                        help='Sub-folder inside lightning_logs/ to group runs.')

    args = parser.parse_args()

    final_save_dir = args.save_dir

    print("=" * 50)
    print("  NODiF — Ackermann LiDAR Training")
    print("=" * 50)
    for k, v in vars(args).items():
        print(f"  {k:<22} = {v}")
    print("=" * 50)

    # ── Data ─────────────────────────────────────────────────────────────
    dm = AckermannDataModule(
        data_path    = args.data_path,
        bptt_seq_len = 100,
        batch_size   = args.batch_size,
    )
    dm.setup()

    # ── Model ─────────────────────────────────────────────────────────────
    model = ModularNeuralODEFilter(
        state_dim         = dm.state_dim,        # 4
        obs_dim           = dm.obs_dim,           # 4 — lidar only
        control_dim       = dm.control_dim,       # 2 — passed separately to physics predictor
        hidden_dim        = args.hidden_dim,
        state_gru_dim     = args.state_gru_dim,
        num_particles     = args.num_particles,
        lr                = args.lr,
        noise_std         = args.noise_std,
        integration_steps = args.integration_steps,
        loss_type         = args.loss_type,
        integration_type  = args.integration_type,
        lam_schedule      = args.lam_schedule,
    )
    model.tbptt_k = args.start_tbptt_k

    # ── Logger ────────────────────────────────────────────────────────────
    experiment_name = f"{args.loss_type}_{args.integration_type}"
    logger = TensorBoardLogger(
        save_dir = final_save_dir,
        name     = experiment_name,
        version  = f"lam_{args.lam_schedule}",
    )

    full_hparams = config_to_dict(cfg)
    full_hparams['cli_args']             = vars(args)
    full_hparams['dataset_state_dim']    = dm.state_dim
    full_hparams['dataset_obs_dim']      = dm.obs_dim
    full_hparams['dataset_control_dim']  = dm.control_dim
    logger.log_hyperparams(full_hparams)

    # ── Callbacks ─────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(logger.log_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        monitor    = 'val_loss',
        mode       = 'min',
        save_top_k = 5,
        filename   = 'best_model_{val_loss:.4f}',
        dirpath    = ckpt_dir,
    )
    early_stop_callback = EarlyStopping(
        monitor  = 'val_loss',
        patience = 15,
        mode     = 'min',
    )
    tbptt_curriculum_cb = TBPTTCurriculum(
        start_tbptt_k     = args.start_tbptt_k,
        max_tbptt_k       = args.bptt_seq_len,
        curriculum_epochs = args.curriculum_epochs,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs        = args.num_epochs,
        callbacks         = [checkpoint_callback, early_stop_callback,
                             tbptt_curriculum_cb],
        logger            = logger,
        gradient_clip_val = 5.0,
        accelerator       = 'auto',
        devices           = 1,
    )
    trainer.fit(model, datamodule=dm)

    # ── Rename checkpoints by rank ─────────────────────────────────────────
    if ckpt_dir and os.path.exists(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]

        def extract_loss(fname):
            try:
                return float(fname.replace('best_model_', '').replace('.ckpt', ''))
            except ValueError:
                return float('inf')

        for rank, fname in enumerate(sorted(ckpt_files, key=extract_loss), 1):
            os.rename(
                os.path.join(ckpt_dir, fname),
                os.path.join(ckpt_dir, f'best_model_{rank}.ckpt')
            )
            print(f"  Rank {rank}: {fname} -> best_model_{rank}.ckpt")

    print(f"\nFinished. Best model: "
          f"{os.path.join(ckpt_dir, 'best_model_1.ckpt')}")


if __name__ == "__main__":
    main()