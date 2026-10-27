"""
Unit Tests for Deep Learning Architectures and Physics Loss Constraints.
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import torch
import pytest
from src.models import PhysicsANN
from src.loss import PhysicsInformedLoss


def test_physics_ann_forward_pass():
    """Validates shape consistency of PhysicsANN network."""
    batch_size = 16
    in_features = 25
    out_features = 4

    model = PhysicsANN(in_features, out_features)
    x = torch.randn(batch_size, in_features)
    out = model(x)

    assert out.shape == (batch_size, out_features)


def test_physics_loss_computation():
    """Validates physics loss components and non-negativity."""
    criterion = PhysicsInformedLoss()
    pred = torch.tensor([[0.8, 0.9, 0.72, 0.1], [0.5, 0.8, 0.40, 0.05]], dtype=torch.float32)
    target = torch.tensor([[0.78, 0.88, 0.70, 0.09], [0.52, 0.79, 0.41, 0.06]], dtype=torch.float32)
    x_eq = torch.tensor([0.85, 0.60], dtype=torch.float32)
    yield_t = torch.tensor([0.70, 0.41], dtype=torch.float32)

    total_loss, breakdown = criterion(pred, target, x_eq, yield_t)

    assert total_loss > 0.0
    assert 'loss_data' in breakdown
    assert 'loss_eq' in breakdown
    assert 'loss_yield' in breakdown
    assert 'loss_pos' in breakdown
