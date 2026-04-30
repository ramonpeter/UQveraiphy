from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 12345
    # data
    n_obs: int = 500
    resolution: float = 0.03
    noise: float = 0.35
    # grid boundaries
    x_lim: tuple[float, float] = (-2, 3)
    y_lim: tuple[float, float] = (-2, 2.3)
    # MLP hyperparams
    width: int = 32
    depth: int = 2
    n_member: int = 5
    wd: float = 1 / 500
    # training
    n_steps: int = 5_000
    lr: float = 1e-3
    # HMC
    num_samples: int = 500
    num_warmup: int = 500
    # plotting
    data_alpha: float = 0.9
