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

    def forward(self, x_in_pressures_flat, x_time_steps):
        batch_size = x_in_pressures_flat.size(0)
        x_time_steps = x_time_steps.long()

        x_embedded = self.embedding(x_in_pressures_flat)
        x_embedded = x_embedded.view(batch_size, self.seq_len, -1)

        x_embedded = (
            x_embedded
            + self.time_step_embedding(x_time_steps).unsqueeze(1)
            + self.positional_encoding
        )
        return x_embedded