import torch
import torch.nn as nn
import math

class EmbeddingAndEncoding2D(nn.Module):
    """改进的嵌入模块，添加2D空间坐标嵌入以保留空间邻域信息。"""
    
    def __init__(self, input_dim, d_model, max_time_steps, seq_len, grid_height=7, grid_width=7):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        
        # 1D位置编码（原有的序列位置编码）
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # 2D空间坐标嵌入
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.seq_len = seq_len
        self.max_time_steps = max_time_steps
        
        # 验证网格尺寸与序列长度的一致性
        assert grid_height * grid_width == seq_len, f"网格尺寸 {grid_height}x{grid_width} 与序列长度 {seq_len} 不匹配"
        
        # 2D位置嵌入：分别为x和y坐标创建嵌入
        self.x_position_embedding = nn.Embedding(grid_width, d_model // 2)
        self.y_position_embedding = nn.Embedding(grid_height, d_model // 2)
        
        # 如果d_model是奇数，需要调整
        self.coord_projection = nn.Linear(d_model, d_model) if d_model % 2 != 0 else None
        
        # 初始化2D坐标网格
        self.register_buffer('coord_grid', self._create_coordinate_grid())
        
    def _create_coordinate_grid(self):
        """创建2D坐标网格，将2D坐标映射到展平的序列索引。"""
        # 创建网格坐标 (H, W, 2)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.grid_height),
            torch.arange(self.grid_width),
            indexing='ij'
        )
        
        # 展平为 (seq_len, 2)，其中每行是 [y, x] 坐标
        coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1)
        return coords
    
    def _get_2d_position_embedding(self, batch_size):
        """获取2D位置嵌入。"""
        # 获取x和y坐标
        y_coords = self.coord_grid[:, 0]  # (seq_len,)
        x_coords = self.coord_grid[:, 1]  # (seq_len,)
        
        # 获取x和y的嵌入
        x_embed = self.x_position_embedding(x_coords)  # (seq_len, d_model//2)
        y_embed = self.y_position_embedding(y_coords)  # (seq_len, d_model//2)
        
        # 拼接x和y嵌入
        coord_embed = torch.cat([x_embed, y_embed], dim=-1)  # (seq_len, d_model)
        
        # 如果d_model是奇数，通过线性层调整维度
        if self.coord_projection is not None:
            coord_embed = self.coord_projection(coord_embed)
        
        # 扩展到batch维度
        coord_embed = coord_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        return coord_embed
    
    def forward(self, x_in_pressures_flat, x_time_steps):
        """前向传播，添加2D空间位置信息。
        
        Args:
            x_in_pressures_flat: 展平的压力数据 (batch_size, input_dim)
            x_time_steps: 时间步 (batch_size,)
            
        Returns:
            嵌入后的数据，包含空间位置信息 (batch_size, seq_len, d_model)
        """
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
        
        # 基础嵌入：将展平数据转换为每个位置的特征
        # 假设输入是展平的网格数据，需要重塑为 (batch_size, seq_len, feature_per_point)
        if x_in_pressures_flat.size(1) == self.seq_len:
            # 如果输入已经是 (batch_size, seq_len) 格式
            x_reshaped = x_in_pressures_flat.unsqueeze(-1)  # (batch_size, seq_len, 1)
        else:
            # 如果输入是完全展平的，需要重塑
            feature_per_point = x_in_pressures_flat.size(1) // self.seq_len
            x_reshaped = x_in_pressures_flat.view(batch_size, self.seq_len, feature_per_point)
        
        # 应用线性嵌入
        x_embedded = self.embedding(x_reshaped)  # (batch_size, seq_len, d_model)
        
        # 添加时间嵌入
        time_embed = self.time_step_embedding(x_time_steps).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # 获取2D位置嵌入
        spatial_embed = self._get_2d_position_embedding(batch_size)  # (batch_size, seq_len, d_model)
        
        # 组合所有嵌入
        x_embedded = (
            x_embedded +
            time_embed +
            self.positional_encoding +
            spatial_embed
        )
        
        return x_embedded
    
    def get_spatial_coordinates(self):
        """返回空间坐标信息，用于可视化或调试。"""
        return self.coord_grid
    
    def visualize_coordinate_mapping(self):
        """可视化坐标映射关系。"""
        print(f"网格尺寸: {self.grid_height} x {self.grid_width}")
        print(f"序列长度: {self.seq_len}")
        print("坐标映射 (序列索引 -> [y, x]):")
        for i in range(min(10, self.seq_len)):  # 只显示前10个
            y, x = self.coord_grid[i]
            print(f"  索引 {i:2d} -> [{y:2d}, {x:2d}]")
        if self.seq_len > 10:
            print(f"  ... (共 {self.seq_len} 个位置)")


class EmbeddingAndEncodingOriginal(nn.Module):
    """原始的嵌入模块，保持向后兼容性。"""
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


# 为了向后兼容，保持原有的类名
EmbeddingAndEncoding = EmbeddingAndEncodingOriginal