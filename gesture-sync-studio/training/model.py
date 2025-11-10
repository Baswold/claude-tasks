"""
Neural network models for audio-to-gesture generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioToGestureModel(nn.Module):
    """
    Sequence-to-sequence model: Audio features -> Gesture sequence
    Using LSTM/GRU architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 84,  # 12 bones * 7 DOF
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = False,
        use_attention: bool = False
    ):
        """
        Initialize model.

        Args:
            input_dim: Audio feature dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension (bone parameters)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Encoder: Process audio features
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Attention (optional)
        if use_attention:
            attn_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )

        # Output projection
        proj_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(proj_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        logger.info(f"Initialized AudioToGestureModel: {input_dim} -> {output_dim}")

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input audio features (batch, time_steps, input_dim)
            hidden: Initial hidden state (optional)

        Returns:
            Output gestures (batch, time_steps, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Encode audio features
        encoded, (h_n, c_n) = self.encoder(x, hidden)

        # Apply attention (optional)
        if self.use_attention:
            encoded, _ = self.attention(encoded, encoded, encoded)

        # Project to output
        output = self.fc_layers(encoded)

        return output


class SimpleGestureModel(nn.Module):
    """
    Simple frame-by-frame mapping model.
    Faster to train, less temporal coherence than LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 84,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_temporal_conv: bool = True
    ):
        """
        Initialize simple model.

        Args:
            input_dim: Audio feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            use_temporal_conv: Use 1D conv for temporal context
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_temporal_conv = use_temporal_conv

        # Optional temporal convolution
        if use_temporal_conv:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU()
            )
            mlp_input_dim = hidden_dim
        else:
            mlp_input_dim = input_dim

        # MLP layers
        layers = []
        current_dim = mlp_input_dim

        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim

            layers.append(nn.Linear(current_dim, next_dim))

            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            current_dim = next_dim

        self.mlp = nn.Sequential(*layers)

        logger.info(f"Initialized SimpleGestureModel: {input_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, time_steps, input_dim)

        Returns:
            Output gestures (batch, time_steps, output_dim)
        """
        if self.use_temporal_conv:
            # Conv1d expects (batch, channels, time)
            x_conv = x.transpose(1, 2)
            x_conv = self.temporal_conv(x_conv)
            x = x_conv.transpose(1, 2)

        # Apply MLP
        output = self.mlp(x)

        return output


class TransformerGestureModel(nn.Module):
    """
    Transformer-based model for audio-to-gesture generation.
    Better for long-range dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 84,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        """
        Initialize transformer model.

        Args:
            input_dim: Audio feature dimension
            hidden_dim: Model dimension
            output_dim: Output dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        logger.info(f"Initialized TransformerGestureModel: {input_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, time_steps, input_dim)

        Returns:
            Output gestures (batch, time_steps, output_dim)
        """
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transform
        x = self.transformer_encoder(x)

        # Project to output
        output = self.output_projection(x)

        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


def create_model(model_type: str = 'lstm', **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('lstm', 'simple', 'transformer')
        **kwargs: Model parameters

    Returns:
        Model instance
    """
    if model_type == 'lstm':
        return AudioToGestureModel(**kwargs)
    elif model_type == 'simple':
        return SimpleGestureModel(**kwargs)
    elif model_type == 'transformer':
        return TransformerGestureModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class TemporalSmoothnessLoss(nn.Module):
    """
    Loss to encourage temporal smoothness in generated gestures.
    Penalizes large frame-to-frame differences.
    """

    def __init__(self, weight: float = 0.1):
        """
        Initialize loss.

        Args:
            weight: Weight for smoothness term
        """
        super().__init__()
        self.weight = weight

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss.

        Args:
            predictions: Predicted gestures (batch, time_steps, output_dim)

        Returns:
            Smoothness loss value
        """
        # Compute frame-to-frame differences
        diff = predictions[:, 1:, :] - predictions[:, :-1, :]

        # L2 norm of differences
        smoothness_loss = torch.mean(diff ** 2)

        return self.weight * smoothness_loss


class VelocityLoss(nn.Module):
    """
    Loss based on velocity (first derivative) differences.
    Encourages similar motion dynamics between prediction and target.
    """

    def __init__(self, weight: float = 0.05):
        """
        Initialize loss.

        Args:
            weight: Weight for velocity term
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity loss.

        Args:
            predictions: Predicted gestures (batch, time_steps, output_dim)
            targets: Target gestures (batch, time_steps, output_dim)

        Returns:
            Velocity loss value
        """
        # Compute velocities
        pred_velocity = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_velocity = targets[:, 1:, :] - targets[:, :-1, :]

        # MSE on velocities
        velocity_loss = F.mse_loss(pred_velocity, target_velocity)

        return self.weight * velocity_loss


class GestureLoss(nn.Module):
    """
    Combined loss function for gesture generation.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        velocity_weight: float = 0.05
    ):
        """
        Initialize combined loss.

        Args:
            mse_weight: Weight for MSE loss
            smoothness_weight: Weight for smoothness loss
            velocity_weight: Weight for velocity loss
        """
        super().__init__()

        self.mse_weight = mse_weight
        self.smoothness_loss = TemporalSmoothnessLoss(weight=smoothness_weight)
        self.velocity_loss = VelocityLoss(weight=velocity_weight)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            predictions: Predicted gestures
            targets: Target gestures

        Returns:
            Total loss and dictionary of individual losses
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets) * self.mse_weight

        # Smoothness loss
        smoothness = self.smoothness_loss(predictions)

        # Velocity loss
        velocity = self.velocity_loss(predictions, targets)

        # Total loss
        total_loss = mse_loss + smoothness + velocity

        # Return losses
        losses = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'smoothness': smoothness.item(),
            'velocity': velocity.item()
        }

        return total_loss, losses
