import torch
import torch.nn as nn


def get_svd_modes_batched(tensor, topk=10):
    """
    批量化 SVD 模态计算，提升效率
    
    Args:
        tensor: [B, H, W] 输入张量（已经 reshape）
        topk: 前 K 个模态
    
    Returns:
        modes: List of [B, H, W] 张量，长度为 topk
    """
    B, H, W = tensor.shape
    
    # 批量 SVD 计算，避免 Python 循环
    u, s, vh = torch.linalg.svd(tensor, full_matrices=False)  # [B, H, min(H,W)], [B, min(H,W)], [B, min(H,W), W]
    
    # 防止 topk 超出奇异值数量
    effective_topk = min(topk, s.shape[-1])
    
    modes = []
    for k in range(effective_topk):
        # 批量计算第 k 个模态
        s_k = s[:, k]  # [B]
        u_k = u[:, :, k]  # [B, H]
        vh_k = vh[:, k, :]  # [B, W]
        
        # 批量外积：s_k * u_k.outer(vh_k) for each batch
        mode_k = s_k.view(B, 1, 1) * torch.bmm(u_k.unsqueeze(-1), vh_k.unsqueeze(1))  # [B, H, W]
        modes.append(mode_k)
    
    # 如果 topk 超出实际模态数，用零张量填充
    while len(modes) < topk:
        zero_mode = torch.zeros_like(tensor)
        modes.append(zero_mode)
    
    return modes


def get_svd_modes_with_config_reshape(tensor, topk=10, grid_height=None, grid_width=None):
    """
    支持配置驱动 reshape 的 SVD 模态计算
    
    Args:
        tensor: [B, N] 或 [B, H, W] 输入张量
        topk: 前 K 个模态
        grid_height: 配置中的网格高度
        grid_width: 配置中的网格宽度
    
    Returns:
        modes: List of [B, H, W] 张量，长度为 topk
    """
    if tensor.dim() == 2:  # [B, N] -> [B, H, W]
        B, N = tensor.shape
        
        if grid_height is not None and grid_width is not None:
            # 使用配置参数进行 reshape
            if grid_height * grid_width != N:
                raise ValueError(f"grid_height({grid_height}) * grid_width({grid_width}) = {grid_height * grid_width} != seq_len({N})")
            H, W = grid_height, grid_width
        else:
            # 回退到完全平方数假设
            hw = int(N**0.5)
            if hw * hw != N:
                raise ValueError(f"seq_len({N}) is not a perfect square and no grid dimensions provided")
            H, W = hw, hw
        
        tensor = tensor.view(B, H, W)
    
    return get_svd_modes_batched(tensor, topk)


def svd_topk_losses_optimized(pred, target, topk=10, grid_height=None, grid_width=None):
    """优化版本的 SVD top-k 损失计算"""
    pred_modes = get_svd_modes_with_config_reshape(pred, topk, grid_height, grid_width)
    target_modes = get_svd_modes_with_config_reshape(target, topk, grid_height, grid_width)
    
    losses = []
    for k in range(topk):
        loss_k = ((pred_modes[k] - target_modes[k]) ** 2).mean()
        losses.append(loss_k)
    
    return losses  # [L_svd1, L_svd2, ..., L_svd_topk]


class TotalLossWithSVDOptimized(nn.Module):
    """
    优化版本的 SVD 损失函数
    
    关键优化:
    1. 支持配置驱动的 reshape（非完全平方数）
    2. 批量化 SVD 计算
    3. 条件性 SVD 计算（早期退出）
    4. 更好的数值稳定性
    """
    
    def __init__(self, base_weight=0.5, svd_weights=None, topk=10, 
                 grid_height=None, grid_width=None, svd_enabled=True):
        super().__init__()
        
        if svd_weights is None:
            svd_weights = [0.5 / topk] * topk  # 默认均分权重
        
        # 权重归一化
        all_weights = [base_weight] + svd_weights
        weight_sum = sum(all_weights)
        self.base_weight = base_weight / weight_sum
        self.svd_weights = [w / weight_sum for w in svd_weights]
        
        self.topk = topk
        self.base_loss = nn.MSELoss()
        
        # 配置参数
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.svd_enabled = svd_enabled
        
        # 早期退出判断
        self._svd_effectively_disabled = (
            not svd_enabled or 
            topk <= 0 or 
            not any(w > 1e-8 for w in self.svd_weights)
        )

    def forward(self, pred, target):
        """
        前向传播
        
        Args:
            pred: [B, N] 或 [B, H, W] 预测张量
            target: [B, N] 或 [B, H, W] 目标张量
        
        Returns:
            total_loss: 标量损失
        """
        # 基础 MSE 损失
        loss_base = self.base_loss(pred, target)
        
        # 早期退出：如果 SVD 实际上被禁用，跳过所有 SVD 计算
        if self._svd_effectively_disabled:
            return self.base_weight * loss_base
        
        # SVD 损失计算
        try:
            loss_svds = svd_topk_losses_optimized(
                pred, target, 
                topk=self.topk,
                grid_height=self.grid_height,
                grid_width=self.grid_width
            )
            
            # 组合损失
            total_loss = self.base_weight * loss_base
            for w, l in zip(self.svd_weights, loss_svds):
                total_loss += w * l
                
            return total_loss
            
        except Exception as e:
            # SVD 计算失败时回退到基础损失
            print(f"Warning: SVD computation failed ({e}), falling back to base loss")
            return self.base_weight * loss_base
    
    def update_config(self, grid_height=None, grid_width=None, svd_enabled=None):
        """动态更新配置参数"""
        if grid_height is not None:
            self.grid_height = grid_height
        if grid_width is not None:
            self.grid_width = grid_width
        if svd_enabled is not None:
            self.svd_enabled = svd_enabled
            
        # 重新计算早期退出条件
        self._svd_effectively_disabled = (
            not self.svd_enabled or 
            self.topk <= 0 or 
            not any(w > 1e-8 for w in self.svd_weights)
        )
    
    def get_svd_stats(self, pred, target):
        """获取 SVD 统计信息（用于调试和分析）"""
        if pred.dim() == 2:
            if self.grid_height and self.grid_width:
                pred = pred.view(-1, self.grid_height, self.grid_width)
                target = target.view(-1, self.grid_height, self.grid_width)
            else:
                N = pred.shape[1]
                hw = int(N**0.5)
                pred = pred.view(-1, hw, hw)
                target = target.view(-1, hw, hw)
        
        # 计算奇异值
        pred_svd = torch.linalg.svd(pred, full_matrices=False)
        target_svd = torch.linalg.svd(target, full_matrices=False)
        
        return {
            'pred_singular_values': pred_svd[1].detach().cpu(),
            'target_singular_values': target_svd[1].detach().cpu(),
            'effective_rank_pred': (pred_svd[1] > 1e-6).sum(dim=-1).float().mean().item(),
            'effective_rank_target': (target_svd[1] > 1e-6).sum(dim=-1).float().mean().item(),
        }


# 向后兼容的别名
TotalLossWithSVD = TotalLossWithSVDOptimized