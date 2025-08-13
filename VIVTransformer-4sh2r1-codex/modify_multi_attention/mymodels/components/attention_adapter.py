from enum import Enum
import torch
import torch.nn as nn
from fightingcv_attention.attention.A2Atttention import DoubleAttention
from fightingcv_attention.attention.AFT import AFT_FULL
from fightingcv_attention.attention.BAM import BAMBlock
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.CoAtNet import CoAtNet
from fightingcv_attention.attention.CoordAttention import CoordAtt
from fightingcv_attention.attention.CoTAttention import CoTAttention
from fightingcv_attention.attention.DANet import DAModule
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.ExternalAttention import ExternalAttention
from fightingcv_attention.attention.HaloAttention import HaloAttention
from fightingcv_attention.attention.OutlookAttention import OutlookAttention
from fightingcv_attention.attention.ParNetAttention import ParNetAttention
from fightingcv_attention.attention.PolarizedSelfAttention import (
    SequentialPolarizedSelfAttention,
)
from fightingcv_attention.attention.PSA import PSA
from fightingcv_attention.attention.ResidualAttention import ResidualAttention
from fightingcv_attention.attention.S2Attention import S2Attention
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.SGE import SpatialGroupEnhance
from fightingcv_attention.attention.ShuffleAttention import ShuffleAttention
from fightingcv_attention.attention.SKAttention import SKAttention
from fightingcv_attention.attention.TripletAttention import TripletAttention
from fightingcv_attention.attention.ViP import WeightedPermuteMLP


class AdapterType(Enum):
    QKV = "qkv"
    CNN = "cnn"
    SINGLE_INPUT = "single_input"


class AttentionAdapter(nn.Module):
    """Base adapter for attention mechanisms."""

    def __init__(self, attention_module):
        super().__init__()
        self.attention = attention_module

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class QKVAttentionAdapter(AttentionAdapter):
    """Adapter for attention mechanisms that expect Q, K, V inputs."""

    def forward(self, x, memory=None, return_attention=False):
        if memory is None:
            memory = x
        target_device = x.device
        
        def _safe_cpu_fallback(attention_module, x_cpu, memory_cpu, return_attention):
            """安全的CPU回退执行函数"""
            try:
                if return_attention:
                    try:
                        out = attention_module(x_cpu, memory_cpu, memory_cpu, return_attention=True)
                        if isinstance(out, tuple) and len(out) == 2:
                            y, att = out
                        else:
                            y, att = out, None
                    except TypeError:
                        # 模块不支持 return_attention 参数
                        y = attention_module(x_cpu, memory_cpu, memory_cpu)
                        att = None
                else:
                    y = attention_module(x_cpu, memory_cpu, memory_cpu)
                    att = None
                
                # 确保输出格式正确
                if isinstance(y, tuple):
                    y = y[0]
                
                # 移回目标设备
                y = y.to(target_device)
                if return_attention:
                    return y, att
                return y
                
            except Exception as e:
                print(f"[ERROR] CPU fallback also failed for {attention_module.__class__.__name__}: {e}")
                # 返回输入的副本作为fallback
                if return_attention:
                    return x.clone(), None
                return x.clone()
        
        # 若模块被标记为强制CPU执行，则直接走CPU路径
        if getattr(self.attention, "_force_cpu", False):
            cpu = torch.device("cpu")
            try:
                self.attention = self.attention.to(cpu)
            except Exception:
                pass
            return _safe_cpu_fallback(self.attention, x.to(cpu), memory.to(cpu), return_attention)
        
        try:
            # 首先尝试在GPU上运行
            return self.attention(x, memory, memory, return_attention=return_attention)
        except TypeError:
            # 处理不支持 return_attention 的模块
            if return_attention:
                return self.attention(x, memory, memory), None
            else:
                return self.attention(x, memory, memory)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # 统一处理OOM和运行时错误
            error_msg = str(e).lower()
            is_oom = "out of memory" in error_msg
            is_device_mismatch = (
                "input type (torch.cuda.floattensor) and weight type (torch.floattensor)" in error_msg
                or "expected all tensors to be on the same device" in error_msg
            )
            
            if is_oom or is_device_mismatch:
                att_cls = self.attention.__class__.__name__
                print(f"[WARN] {att_cls} GPU execution failed ({'OOM' if is_oom else 'Device Mismatch'}), falling back to CPU")
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 移动模块到CPU
                cpu = torch.device("cpu")
                try:
                    self.attention = self.attention.to(cpu)
                except Exception as move_error:
                    print(f"[WARN] Failed to move {att_cls} to CPU: {move_error}")
                
                # 在CPU上执行
                return _safe_cpu_fallback(self.attention, x.to(cpu), memory.to(cpu), return_attention)
            else:
                # 其他RuntimeError直接抛出
                raise


class CNNStyleAttentionAdapter(AttentionAdapter):
    """Adapter for CNN-style attention mechanisms that expect a 4D tensor."""

    def forward(self, x, memory=None, return_attention=False):
        batch_size, seq_len, d_model = x.shape
        spatial_dim = int(seq_len**0.5)
        if spatial_dim * spatial_dim != seq_len:
            raise ValueError(
                "Sequence length cannot form square spatial dimensions for CNN attention."
            )

        x_reshaped = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, d_model, spatial_dim, spatial_dim)
        )
        target_device = x_reshaped.device
        
        def _safe_cpu_fallback(attention_module, x_reshaped_cpu):
            """安全的CPU回退执行函数"""
            try:
                output = attention_module(x_reshaped_cpu)
                if isinstance(output, tuple):
                    output = output[0]
                if output is None:
                    output = torch.zeros_like(x_reshaped_cpu)
                return output.to(target_device)
            except Exception as e:
                print(f"[ERROR] CPU fallback also failed for {attention_module.__class__.__name__}: {e}")
                # 返回零张量作为fallback
                return torch.zeros_like(x_reshaped)
        
        # 强制CPU执行路径：若模块被标记，则直接用CPU运行
        if getattr(self.attention, "_force_cpu", False):
            cpu = torch.device("cpu")
            try:
                self.attention = self.attention.to(cpu)
            except Exception:
                pass
            output = _safe_cpu_fallback(self.attention, x_reshaped.to(cpu))
            output = output.view(batch_size, d_model, seq_len).transpose(1, 2)
            if return_attention:
                return output, None
            return output
        
        # 确保注意力模块在正确设备上
        try:
            if next(self.attention.parameters(), None) is not None:
                self.attention = self.attention.to(target_device)
        except Exception:
            pass
        
        try:
            output = self.attention(x_reshaped)
            if isinstance(output, tuple):
                output = output[0]
            if output is None:
                output = torch.zeros_like(x_reshaped)
                
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # 统一处理OOM和运行时错误
            error_msg = str(e).lower()
            is_oom = "out of memory" in error_msg
            is_device_mismatch = (
                "input type (torch.cuda.floattensor) and weight type (torch.floattensor)" in error_msg
                or "expected all tensors to be on the same device" in error_msg
            )
            
            if is_oom or is_device_mismatch:
                att_cls = self.attention.__class__.__name__
                print(f"[WARN] {att_cls} GPU execution failed ({'OOM' if is_oom else 'Device Mismatch'}), falling back to CPU")
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 移动到CPU并执行
                cpu = torch.device("cpu")
                try:
                    self.attention = self.attention.to(cpu)
                except Exception as move_error:
                    print(f"[WARN] Failed to move {att_cls} to CPU: {move_error}")
                
                output = _safe_cpu_fallback(self.attention, x_reshaped.to(cpu))
            else:
                # 其他RuntimeError直接抛出
                raise
        except Exception as e:
            # 其他异常的处理
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"ERROR: Exception in {self.attention.__class__.__name__}: {e}")
            raise RuntimeError(f"Attention module {self.attention.__class__.__name__} failed: {e}") from e

        output = output.view(batch_size, d_model, seq_len).transpose(1, 2)

        if return_attention:
            return output, None  # No attention weights to return
        else:
            return output


class SingleInputAttentionAdapter(AttentionAdapter):
    """Adapter for attention mechanisms that expect a single tensor input."""

    def forward(self, x, memory=None, return_attention=False):
        # Warn if memory is provided but will be ignored
        if memory is not None:
            import warnings
            warnings.warn(
                f"{self.attention.__class__.__name__} is a single-input attention mechanism "
                "that does not support cross-attention. The 'memory' parameter will be ignored.",
                UserWarning
            )
        
        # Special handling for attentions that expect 4D input (B, H, W, C)
        if isinstance(self.attention, (OutlookAttention, WeightedPermuteMLP)):
            B, N, C = x.shape
            orig_N = N  # 保存原始序列长度
            # 尝试找到接近正方形的 H, W
            H = int(N ** 0.5)
            W = H
            if H * W != N:
                # 优先寻找能整除 N 的 W 使得 H=W 或 H≈W
                found = False
                for w in range(H, H + 16):
                    if w == 0: continue
                    if N % w == 0:
                        h = N // w
                        if abs(h - w) <= 2:
                            H, W = h, w
                            found = True
                            break
                if not found:
                    # 回退：填充到最近的正方形
                    H = W = H + 1
                    pad = H * W - N
                    x = torch.cat([x, torch.zeros(B, pad, C, device=x.device, dtype=x.dtype)], dim=1)
                    N = H * W
            x_reshaped = x.view(B, H, W, C)
            
            # 强制CPU执行路径
            if getattr(self.attention, "_force_cpu", False):
                target_device = x_reshaped.device
                cpu = torch.device("cpu")
                try:
                    self.attention = self.attention.to(cpu)
                except Exception:
                    pass
                try:
                    output = self.attention(x_reshaped.to(cpu))
                    if isinstance(output, tuple):
                        output = output[0]
                except Exception as e:
                    print(f"[ERROR] CPU forced path failed for {self.attention.__class__.__name__}: {e}")
                    output = torch.zeros_like(x_reshaped.to(cpu))
                output = output.to(target_device)
                output = output.view(B, N, C)
                if N != orig_N:
                    output = output[:, :orig_N, :]
                return (output, None) if return_attention else output
            
            try:
                output = self.attention(x_reshaped)
                if isinstance(output, tuple):
                    output = output[0]
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # 如果GPU失败，回退到CPU
                error_msg = str(e).lower()
                is_oom = "out of memory" in error_msg
                is_device_mismatch = (
                    "input type (torch.cuda.floattensor) and weight type (torch.floattensor)" in error_msg
                    or "expected all tensors to be on the same device" in error_msg
                )
                if is_oom or is_device_mismatch:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cpu = torch.device("cpu")
                    try:
                        self.attention = self.attention.to(cpu)
                    except Exception:
                        pass
                    try:
                        output = self.attention(x_reshaped.to(cpu))
                        if isinstance(output, tuple):
                            output = output[0]
                    except Exception:
                        output = torch.zeros_like(x_reshaped.to(cpu))
                    output = output.to(x_reshaped.device)
                else:
                    raise
            
            output = output.view(B, N, C)
            if N != orig_N:
                output = output[:, :orig_N, :]
            return (output, None) if return_attention else output
        
        # 非4D特殊模块
        target_device = x.device
        if getattr(self.attention, "_force_cpu", False):
            cpu = torch.device("cpu")
            try:
                self.attention = self.attention.to(cpu)
            except Exception:
                pass
            try:
                y = self.attention(x.to(cpu))
                if isinstance(y, tuple):
                    y = y[0]
            except Exception:
                y = torch.zeros_like(x.to(cpu))
            y = y.to(target_device)
            return (y, None) if return_attention else y
        
        try:
            y = self.attention(x)
            if isinstance(y, tuple):
                y = y[0]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e).lower()
            is_oom = "out of memory" in error_msg
            is_device_mismatch = (
                "input type (torch.cuda.floattensor) and weight type (torch.floattensor)" in error_msg
                or "expected all tensors to be on the same device" in error_msg
            )
            if is_oom or is_device_mismatch:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cpu = torch.device("cpu")
                try:
                    self.attention = self.attention.to(cpu)
                except Exception:
                    pass
                try:
                    y = self.attention(x.to(cpu))
                    if isinstance(y, tuple):
                        y = y[0]
                except Exception:
                    y = torch.zeros_like(x.to(cpu))
                y = y.to(target_device)
            else:
                raise
        return (y, None) if return_attention else y


ADAPTER_CATALOG = {
    AdapterType.QKV: QKVAttentionAdapter,
    AdapterType.CNN: CNNStyleAttentionAdapter,
    AdapterType.SINGLE_INPUT: SingleInputAttentionAdapter,
}


def get_attention_adapter(attention_module, adapter_type: AdapterType):
    if adapter_type not in ADAPTER_CATALOG:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    adapter_class = ADAPTER_CATALOG[adapter_type]
    return adapter_class(attention_module)
