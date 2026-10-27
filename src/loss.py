"""
Physics-Guided Loss Function Formulations for Catalytic Deep Learning.
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class PhysicsInformedLoss(nn.Module):
    """
    Multi-objective loss function enforcing chemical kinetics & thermodynamics:
    1. Data fidelity (MSE on empirical observations)
    2. Thermodynamic ceiling penalty: max(0, X_pred - X_eq)^2
    3. Stoichiometric consistency penalty: ||Yield - (Conversion * Selectivity / 100)||^2
    4. Non-negativity constraint: max(0, -pred)^2
    """
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_eq: float = 0.5,
        lambda_yield: float = 0.4,
        lambda_pos: float = 0.1
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_eq = lambda_eq
        self.lambda_yield = lambda_yield
        self.lambda_pos = lambda_pos
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x_eq_batch: torch.Tensor,
        yield_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        pred: (batch, 4) -> [Conversion, Selectivity, Yield, Coke]
        target: (batch, 4)
        x_eq_batch: (batch,) or (batch, 1) -> Thermodynamic ceiling in normalized space
        yield_batch: (batch,) or (batch, 1) -> Yield target in normalized space
        """
        x_pred = pred[:, 0]
        s_pred = pred[:, 1]
        y_pred = pred[:, 2]

        # 1. Data Loss
        loss_data = self.mse(pred, target)

        # 2. Thermodynamic Equilibrium Constraint: X_pred <= X_eq
        loss_eq = torch.mean(torch.clamp(x_pred - x_eq_batch.squeeze(), min=0.0) ** 2)

        # 3. Stoichiometric Yield Law: Yield ~ Conversion * Selectivity
        # In normalized or raw coordinates, checks coupling consistency
        y_calc = x_pred * s_pred
        loss_yield = self.mse(y_calc, yield_batch.squeeze())

        # 4. Physical Non-negativity
        loss_pos = torch.mean(torch.clamp(-pred, min=0.0) ** 2)

        # Total Weighted Loss
        total_loss = (
            self.lambda_data * loss_data +
            self.lambda_eq * loss_eq +
            self.lambda_yield * loss_yield +
            self.lambda_pos * loss_pos
        )

        loss_breakdown = {
            'loss_data': loss_data.item(),
            'loss_eq': loss_eq.item(),
            'loss_yield': loss_yield.item(),
            'loss_pos': loss_pos.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_breakdown
