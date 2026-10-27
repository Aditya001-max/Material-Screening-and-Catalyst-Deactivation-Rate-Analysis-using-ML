"""
Machine Learning & Deep Learning Architectures for Catalyst Performance.
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import numpy as np
from typing import Tuple, List, Dict, Any

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# Graceful optional imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from src.loss import PhysicsInformedLoss
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False


if HAS_TORCH:
    class PhysicsANN(nn.Module):
        """
        Physics-Informed Deep Neural Network architecture:
        Dense(in_dim -> 128, ReLU) -> Dense(128 -> 64, ReLU) -> Dense(64 -> 32, ReLU) -> Dense(32 -> 4, Linear)
        """
        def __init__(self, in_features: int, out_features: int = 4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, out_features)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    def train_physics_informed_ann(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eq_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_eq_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu"
    ) -> Tuple[PhysicsANN, List[float], List[float]]:
        """
        Trains PhysicsANN with PhysicsInformedLoss and StepLR scheduler.
        """
        in_dim = x_train.shape[1]
        out_dim = y_train.shape[1]
        model = PhysicsANN(in_dim, out_dim).to(device)

        criterion = PhysicsInformedLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

        # Tensor preparation
        xt_t = torch.tensor(x_train, dtype=torch.float32)
        yt_t = torch.tensor(y_train, dtype=torch.float32)
        xeq_t = torch.tensor(x_eq_train, dtype=torch.float32)
        yld_t = torch.tensor(y_train[:, 2], dtype=torch.float32)

        xv_t = torch.tensor(x_val, dtype=torch.float32).to(device)
        yv_t = torch.tensor(y_val, dtype=torch.float32).to(device)
        xeq_v = torch.tensor(x_eq_val, dtype=torch.float32).to(device)
        yld_v = torch.tensor(y_val[:, 2], dtype=torch.float32).to(device)

        dataset = TensorDataset(xt_t, yt_t, xeq_t, yld_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_history = []
        val_history = []
        best_val_loss = float('inf')
        best_state_dict = None

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for xb, yb, xeq_b, yld_b in loader:
                xb, yb = xb.to(device), yb.to(device)
                xeq_b, yld_b = xeq_b.to(device), yld_b.to(device)

                pred = model(xb)
                loss, _ = criterion(pred, yb, xeq_b, yld_b)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()

            scheduler.step()
            epoch_train_loss = running_loss / len(loader)

            model.eval()
            with torch.no_grad():
                v_pred = model(xv_t)
                v_loss, _ = criterion(v_pred, yv_t, xeq_v, yld_v)
                epoch_val_loss = v_loss.item()

            train_history.append(epoch_train_loss)
            val_history.append(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state_dict = model.state_dict().copy()

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        return model, train_history, val_history
else:
    class PhysicsANN:
        pass


def build_random_forest(n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    """Builds Random Forest Regressor for multi-target prediction."""
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)


def build_xgboost(random_state: int = 42) -> Any:
    """Builds multi-target XGBoost regressor (or fallback GradientBoosting)."""
    if HAS_XGB:
        return xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=6, random_state=random_state)
    print("  [Warning] xgboost not installed in environment; using GradientBoostingRegressor fallback.")
    return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=random_state))


def build_catboost(random_state: int = 42) -> Any:
    """Builds multi-target CatBoost regressor (or fallback GradientBoosting)."""
    if HAS_CB:
        return cb.CatBoostRegressor(iterations=300, learning_rate=0.08, loss_function='MultiRMSE', random_seed=random_state, verbose=0)
    print("  [Warning] catboost not installed in environment; using GradientBoostingRegressor fallback.")
    return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=random_state))
