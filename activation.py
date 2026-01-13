from typing import Callable

import torch.nn as nn


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
        case _:
            raise ValueError(f"Activation function {activation} not supported")
