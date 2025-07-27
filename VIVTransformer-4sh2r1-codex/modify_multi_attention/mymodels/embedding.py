import torch
import torch.nn as nn

class EmbeddingAndEncoding(nn.Module):
    """Module for embedding input and adding positional and time embeddings."""
    def __init__(self, input_dim, d_model, max_time_steps, seq_len):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x_in_pressures_flat, x_time_steps):
        batch_size = x_in_pressures_flat.size(0)
        x_time_steps = x_time_steps.long()
        
        # x_in_pressures_flat shape: [batch_size, seq_len, input_dim]
        # 如果输入是2D，需要重塑为3D
        if x_in_pressures_flat.dim() == 2:
            # 计算实际的seq_len
            total_features = x_in_pressures_flat.size(1)
            # 假设input_dim是从embedding层的输入维度推断
            actual_seq_len = total_features // self.embedding.in_features
            
            if actual_seq_len * self.embedding.in_features == total_features:
                x_in_pressures_flat = x_in_pressures_flat.view(batch_size, actual_seq_len, self.embedding.in_features)
                actual_seq_len_to_use = actual_seq_len
            else:
                # 如果无法整除，使用配置的seq_len
                input_dim = total_features // self.seq_len
                x_in_pressures_flat = x_in_pressures_flat.view(batch_size, self.seq_len, input_dim)
                actual_seq_len_to_use = self.seq_len
        else:
            # 输入已经是3D，使用实际的seq_len
            actual_seq_len_to_use = x_in_pressures_flat.size(1)
        
        # 重塑为 [batch_size * seq_len, input_dim] 进行embedding
        x_reshaped = x_in_pressures_flat.view(-1, x_in_pressures_flat.size(-1))
        x_embedded = self.embedding(x_reshaped)  # [batch_size * seq_len, d_model]
        
        # 使用实际的seq_len进行重塑
        x_embedded = x_embedded.view(batch_size, actual_seq_len_to_use, self.d_model)

        # 添加时间步嵌入和位置编码
        # x_time_steps shape: [batch_size, seq_len]
        # 需要为每个时间步生成嵌入
        time_emb = self.time_step_embedding(x_time_steps)  # [batch_size, seq_len, d_model]
        x_embedded = x_embedded + time_emb + self.positional_encoding
        
        return x_embedded