import torch
import torch.nn as nn

class EmbeddingAndEncoding(nn.Module):
    """Module for embedding input and adding positional and time embeddings."""
    def __init__(self, input_dim, d_model, max_time_steps, seq_len):
        super().__init__()
        self.embedding = nn.Linear(input_dim, seq_len * d_model)
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.seq_len = seq_len
        self.max_time_steps = max_time_steps

    def forward(self, x_in_pressures_flat, x_time_steps):
        batch_size = x_in_pressures_flat.size(0)
        x_time_steps = x_time_steps.long()
        
        # Validate and clamp time step indices to prevent out-of-bounds errors
        if torch.any(x_time_steps < 0) or torch.any(x_time_steps >= self.max_time_steps):
            import warnings
            warnings.warn(
                f"Time step indices out of range [0, {self.max_time_steps-1}]. "
                f"Found min: {x_time_steps.min().item()}, max: {x_time_steps.max().item()}. "
                "Clamping to valid range.",
                UserWarning
            )
        x_time_steps = torch.clamp(x_time_steps, 0, self.max_time_steps - 1)

        x_embedded = self.embedding(x_in_pressures_flat)
        x_embedded = x_embedded.view(batch_size, self.seq_len, -1)

        x_embedded = (
            x_embedded
            + self.time_step_embedding(x_time_steps).unsqueeze(1)
            + self.positional_encoding
        )
        return x_embedded