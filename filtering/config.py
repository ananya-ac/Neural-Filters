import torch
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    T: int = 100                  # was 1000 — keeps total_time = 100 × 1 = 100s
    nTarget: int = 2
    nSensor: int = 4
    simAreaSize: int = 100
    Amp: float = 100.0
    invPow: float = 1.0
    d0: float = 1
    measvar_real: float = 0.01
    process_noise_scale: float = 0.005 #0.1, 0.05, 0.005
    dt: float = 1.0               
    noise_distribution: str = 'gaussian'
    glitch: float = 0.00

@dataclass
class TrainConfig:
    lr: float = 3e-3
    batch_size: int = 512
    num_epochs: int = 100
    bptt_seq_len: int = 30        # ~30% of 100-step trajectory
    data_path: str = 'train_data/acoustic/no_glitch/acoustic_tracking_2_objects.pt'

@dataclass
class ModelConfig:
    num_particles: int = 10
    obs_seq_len: int = 1  
    obs_context_dim: int = 64
    hidden_dim: int = 128
    noise_std: float = 0.005#0.01
    integration_steps: int = 6
 
@dataclass
class NPFFConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)