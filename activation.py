from typing import Callable

import torch
import torch.nn as nn


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def switchable_activation(activation: str = "gelu") -> Callable:
    match activation:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "swish":
            return nn.SiLU()
        case "mish":
            return nn.Mish()
        case "leakyrelu":
            return nn.LeakyReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "elu":
            return nn.ELU()
        case "sin":
            return Sin()
        case _:
            raise ValueError(f"Activation function {activation} not supported")
