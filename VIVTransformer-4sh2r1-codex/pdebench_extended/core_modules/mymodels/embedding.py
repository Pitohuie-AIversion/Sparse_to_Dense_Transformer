import torch
import torch.nn as nn

class EmbeddingAndEncoding(nn.Module):
    """Module for embedding input and adding positional and time embeddings."""
    def __init__(self, input_dim, d_model, max_time_steps, seq_len):
        super().__init__()
        # 直接将输入映射到d_model维度，序列长度由输入数据决定
        self.embedding = nn.Linear(1, d_model)  # 每个输入元素映射到d_model
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))
        self.seq_len = seq_len
        self.input_dim = input_dim

    def forward(self, x_in_pressures_flat, x_time_steps):
        batch_size = x_in_pressures_flat.size(0)
        x_time_steps = x_time_steps.long()

        # 将输入reshape为(batch_size, seq_len, 1)，其中seq_len = input_dim
        x_reshaped = x_in_pressures_flat.view(batch_size, self.input_dim, 1)
        
        # 对每个输入元素进行embedding
        x_embedded = self.embedding(x_reshaped)  # (batch_size, input_dim, d_model)

        # 获取time embedding并正确广播到序列维度
        # 确保x_time_steps是正确的形状
        if x_time_steps.dim() > 1:
            x_time_steps = x_time_steps.squeeze()  # 移除多余的维度
        time_emb = self.time_step_embedding(x_time_steps)  # (batch_size, d_model)
        time_emb = time_emb.unsqueeze(1).expand(-1, self.input_dim, -1)  # (batch_size, input_dim, d_model)
        
        x_embedded = (
            x_embedded
            + time_emb
            + self.positional_encoding
        )
        return x_embedded