import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _RegressionLoss(nn.Module):
    """Base class to define a loss function for regression"""

    @abstractmethod
    def loss_func(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Needs to be implemented in subclass"""
        raise NotImplementedError()

    def forward(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.loss_func(pred, target)


class HetLoss(_RegressionLoss):
    """Class to define a normal heteroscedastic loss"""

    def __init__(self, lbound: int = -30, ubound: int = 11):
        """
        Args:
            lbound: lower bound of clamping. Defaults to 30.
        """
        super().__init__()
        self.lbound = lbound  # lower bound for clamping
        self.ubound = ubound  # upper bound for clamping

    def get_mu_and_sigma2(self, pred: Tensor) -> tuple[Tensor, Tensor]:
        mu, logsigma2 = pred[..., 0], pred[..., 1]
        logsigma2 = logsigma2.clamp(self.lbound, self.ubound)
        sigma2 = logsigma2.exp()
        return mu, sigma2

    def loss_func(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logsigma2 = pred[..., 0], pred[..., 1]
        logsigma2 = logsigma2.clamp(self.lbound, self.ubound)
        sigma2 = logsigma2.exp()
        reco = (target - mu) ** 2
        het_loss = 0.5 * (reco / sigma2 + logsigma2)
        return het_loss, reco, logsigma2


class NaturalHetLoss(_RegressionLoss):
    """Class to define a heteroscedastic loss in natural
    parametrization with eta1 and eta2 as introduced in:

    Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/a901d5540789a086ee0881a82211b63d-Paper-Conference.pdf
    Code: https://github.com/aleximmer/heteroscedastic-nn
    """

    def __init__(self, lbound: int = -30, ubound: int = 11):
        """
        Args:
            lbound: lower bound of clamping. Defaults to -30.
        """
        super().__init__()
        self.lbound = lbound  # lower bound for clamping
        self.ubound = ubound  # upper bound for clamping

    def get_mu_and_sigma2(self, pred: Tensor) -> tuple[Tensor, Tensor]:
        eta1, logneg2eta = pred[..., 0], pred[..., 1]
        logneg2eta = logneg2eta.clamp(
            -self.ubound, -self.lbound
        )  # minus clamping bound of HetLoss since logneg2eta = -log(sigma2)
        eta2 = -0.5 * logneg2eta.exp()
        mu = -eta1 / (2 * eta2)
        sigma2 = -1 / (2 * eta2)

        return mu, sigma2

    def loss_func(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eta1, logneg2eta = pred[..., 0], pred[..., 1]
        logneg2eta = logneg2eta.clamp(
            -self.ubound, -self.lbound
        )  # minus clamping bound of HetLoss since logneg2eta = -log(sigma2)
        eta2 = -0.5 * logneg2eta.exp()
        mu = -eta1 / (2 * eta2)
        reco = (target - mu) ** 2
        het_loss = -eta2 * reco - 0.5 * torch.log(-2 * eta2)
        return het_loss, reco, -torch.log(-2 * eta2)


class MSELoss(_RegressionLoss):
    """Standard MSE loss but with a prediction method"""

    def get_mu_and_sigma2(self, pred: Tensor) -> tuple[Tensor, Tensor]:
        mu = pred[..., 0]
        sigma2 = torch.zeros_like(mu)
        return mu, sigma2

    def loss_func(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu = pred[..., 0]
        mse = (target - mu) ** 2
        return mse, mse, torch.zeros_like(mu)


def kernel(x, y):
    """
    RBF kernel with median estimator
    """
    member = len(x)
    dnorm2 = (x.reshape(member, 1, -1) - y.reshape(1, member, -1)).square().sum(dim=2)
    sigma = torch.quantile(dnorm2.detach(), 0.5) / (2 * math.log(member + 1))
    return torch.exp(-dnorm2 / (2 * sigma))
