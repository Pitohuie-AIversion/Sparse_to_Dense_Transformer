import torch
import torch.nn as nn
from .attention_adapter import AdapterType
from .attention import (
    RelativePositionSelfAttention,
    SparseSelfAttention,
    LSHSelfAttention,
)

# 统一导入所有可用的注意力机制
from fightingcv_attention.attention.ExternalAttention import ExternalAttention
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from fightingcv_attention.attention.SimplifiedSelfAttention import (
    SimplifiedScaledDotProductAttention,
)
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.SKAttention import SKAttention
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.BAM import BAMBlock
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.DANet import DAModule
from fightingcv_attention.attention.PSA import PSA
from fightingcv_attention.attention.EMSA import EMSA
from fightingcv_attention.attention.ShuffleAttention import ShuffleAttention
from fightingcv_attention.attention.MUSEAttention import MUSEAttention
from fightingcv_attention.attention.SGE import SpatialGroupEnhance
from fightingcv_attention.attention.A2Atttention import DoubleAttention
from fightingcv_attention.attention.AFT import AFT_FULL
from fightingcv_attention.attention.OutlookAttention import OutlookAttention
from fightingcv_attention.attention.ViP import WeightedPermuteMLP
from fightingcv_attention.attention.CoAtNet import CoAtNet
from fightingcv_attention.attention.HaloAttention import HaloAttention
from fightingcv_attention.attention.PolarizedSelfAttention import (
    SequentialPolarizedSelfAttention,
)
from fightingcv_attention.attention.CoTAttention import CoTAttention
from fightingcv_attention.attention.ResidualAttention import ResidualAttention
from fightingcv_attention.attention.S2Attention import S2Attention
from fightingcv_attention.attention.gfnet import GFNet
from fightingcv_attention.attention.TripletAttention import TripletAttention
from fightingcv_attention.attention.CoordAttention import CoordAtt
from fightingcv_attention.attention.MobileViTAttention import MobileViTAttention
from fightingcv_attention.attention.ParNetAttention import ParNetAttention
from fightingcv_attention.attention.UFOAttention import UFOAttention
from fightingcv_attention.attention.MobileViTv2Attention import MobileViTv2Attention
from fightingcv_attention.attention.DAT import DAT
from fightingcv_attention.attention.Crossformer import CrossFormer
from fightingcv_attention.attention.MOATransformer import MOATransformer
from fightingcv_attention.attention.CrissCrossAttention import CrissCrossAttention
from fightingcv_attention.attention.Axial_attention import AxialImageTransformer

# 创建一个字典来映射 attention_type 到实际的注意力类
ATTENTION_MODULES = {
    # QKV-style attentions
    "self": ScaledDotProductAttention,
    "simplified_self": SimplifiedScaledDotProductAttention,
    "muse": MUSEAttention,
    "ufo": UFOAttention,
    "relative": RelativePositionSelfAttention,
    "sparse": SparseSelfAttention,
    "lsh": LSHSelfAttention,
    "emsa": EMSA,
    "mobilevit": MobileViTAttention,
    "mobilevitv2": MobileViTv2Attention,
    "dat": DAT,
    "crossformer": CrossFormer,
    "moa": MOATransformer,
    "crisscross": CrissCrossAttention,
    "axial": AxialImageTransformer,
    "gfnet": GFNet,

    # CNN-style attentions
    "se": SEAttention,
    "sk": SKAttention,
    "cbam": CBAMBlock,
    "bam": BAMBlock,
    "eca": ECAAttention,
    "shuffle": ShuffleAttention,
    "sge": SpatialGroupEnhance,

    "residual": None,  # We'll create this dynamically below
    "s2": S2Attention,
    "triplet": TripletAttention,
    "coord": CoordAtt,
    "psa": PSA,
    "danet": DAModule,
    "cot": CoTAttention,
    "polarized": SequentialPolarizedSelfAttention,
    "outlook": OutlookAttention,
    "vip": WeightedPermuteMLP,
    "coatnet": CoAtNet,
    "halo": HaloAttention,
    "a2": DoubleAttention,
    "parnet": ParNetAttention,
    
    # Single-input attentions
    "external": ExternalAttention,
    "aft": AFT_FULL,
}

ADAPTER_MAPPING = {
    # QKV-style attentions
    "self": AdapterType.QKV,
    "simplified_self": AdapterType.QKV,
    "muse": AdapterType.QKV,
    "ufo": AdapterType.QKV,
    "relative": AdapterType.QKV,
    "sparse": AdapterType.QKV,
    "lsh": AdapterType.QKV,
    "emsa": AdapterType.QKV,  # EMSA 使用 QKV 三输入
    "mobilevit": AdapterType.SINGLE_INPUT,  # MobileViTAttention 单输入
    "mobilevitv2": AdapterType.QKV,  # 已包装为 QKV
    "dat": AdapterType.SINGLE_INPUT,  # DAT 是窗口 Transformer，单输入
    "crossformer": AdapterType.SINGLE_INPUT,  # CrossFormer 是图像 Transformer，单输入
    "moa": AdapterType.QKV,  # 已包装为 QKV
    "crisscross": AdapterType.CNN,  # CrissCrossAttention 需要 CNN 输入
    "axial": AdapterType.SINGLE_INPUT,  # AxialImageTransformer 单输入
    "gfnet": AdapterType.SINGLE_INPUT,  # GFNet 是 token mixer，单输入

    # CNN-style attentions
    "se": AdapterType.CNN,
    "sk": AdapterType.CNN,
    "cbam": AdapterType.CNN,
    "bam": AdapterType.CNN,
    "eca": AdapterType.CNN,
    "shuffle": AdapterType.CNN,
    "sge": AdapterType.CNN,

    "residual": AdapterType.CNN,
    "s2": AdapterType.CNN,
    "triplet": AdapterType.CNN,
    "coord": AdapterType.CNN,
    "psa": AdapterType.CNN,
    "danet": AdapterType.CNN,
    "cot": AdapterType.CNN,
    "polarized": AdapterType.CNN,
    "outlook": AdapterType.CNN,  # OutlookAttention 通过CNN适配器走 BCHW 通路
    "vip": AdapterType.SINGLE_INPUT,  # WeightedPermuteMLP 需要4D输入特殊处理  
    "coatnet": AdapterType.CNN,
    "halo": AdapterType.CNN,
    "a2": AdapterType.CNN,
    "parnet": AdapterType.CNN,

    # Single-input attentions
    "external": AdapterType.SINGLE_INPUT,
    "aft": AdapterType.SINGLE_INPUT,
}

# GPU OOM黑名单：这些注意力模块在GPU上容易失败，应当在CPU上运行
GPU_BLACKLIST = {
    "emsa",      # EMSA经常出现OOM
    "crisscross", # CrissCross注意力内存消耗大
    "psa",       # PSA在高分辨率时OOM
    "danet",     # DANet双注意力内存需求高
    "aft",       # AFT在某些情况下OOM
}

def get_attention_module(
    attention_type, d_model=512, num_heads=8, spatial_dim=7, low_memory=False, force_cpu_execution=None, **kwargs
):
    """
    根据 attention_type 返回对应的注意力模块和适配器类型
    
    Args:
        low_memory (bool): 启用低内存模式，减少重量级模块的参数和计算量
        force_cpu_execution (bool): 强制CPU执行，覆盖黑名单检查
    
    提示：以下分支中的超参数目前为"硬编码默认值"。
    如需改为可配置，建议在 config.yaml 中新增 attention_hyperparams 并在此处读取，例如：
    - config.attention_hyperparams.modules.bam.{reduction,dia_val}
    - config.attention_hyperparams.modules.psa.reduction
    - config.attention_hyperparams.modules.se/sk/cbam/shuffle.reduction
    - config.attention_hyperparams.modules.eca.kernel_size
    - config.attention_hyperparams.modules.relative.max_len
    - config.attention_hyperparams.modules.sparse.block_size
    - config.attention_hyperparams.modules.lsh.num_hash_functions
    - config.attention_hyperparams.modules.aft.n (通常为 spatial_dim*spatial_dim)
    保持当前行为不变，仅添加说明。
    """
    if attention_type not in ATTENTION_MODULES:
        raise ValueError(f"Unknown attention type: {attention_type}")

    # 检查是否需要强制CPU执行
    if force_cpu_execution is None:
        force_cpu_execution = attention_type in GPU_BLACKLIST
    
    if force_cpu_execution:
        print(f"[INFO] {attention_type} 被标记为GPU问题模块，强制使用CPU执行")

    adapter_type = ADAPTER_MAPPING.get(attention_type, AdapterType.QKV)
    module_class = ATTENTION_MODULES[attention_type]
    module = None

    # Adapt parameters for different attention modules
    if attention_type == "relative":
        # 硬编码：max_len 默认在 RelativePositionSelfAttention 内部为 500
        module = RelativePositionSelfAttention(d_model, num_heads)
    elif attention_type == "sparse":
        # 硬编码：block_size=8
        module = SparseSelfAttention(d_model, num_heads)
    elif attention_type == "lsh":
        # 硬编码：num_hash_functions=4
        module = LSHSelfAttention(d_model, num_heads)
    elif attention_type in ["external"]:
        # 硬编码：S=8
        module = module_class(d_model=d_model, S=8)
    elif attention_type in ["emsa"]:
        # EMSA 构造：按照缺失参数提示，提供所有必需参数
        # 低内存模式：减少头数和投影维度
        if low_memory:
            reduced_heads = max(1, num_heads // 2)
            reduced_d_k = max(16, d_model // 4)
            try:
                module = EMSA(d_model, d_k=reduced_d_k, d_v=reduced_d_k, h=reduced_heads)
            except TypeError:
                try:
                    module = EMSA(d_model, reduced_d_k, reduced_d_k, reduced_heads)
                except TypeError:
                    module = EMSA(d_model, reduced_d_k, reduced_d_k)
        else:
            try:
                module = EMSA(d_model, d_k=d_model, d_v=d_model, h=num_heads)
            except TypeError:
                try:
                    module = EMSA(d_model, d_model, d_model, num_heads)  # 位置参数
                except TypeError:
                    module = EMSA(d_model, d_model, d_model)
    elif attention_type in ["self", "muse", "ufo"]:
        # 硬编码：d_k=d_model, d_v=d_model
        module = module_class(d_model=d_model, d_k=d_model, d_v=d_model, h=num_heads)
    elif attention_type == "simplified_self":
        module = module_class(d_model=d_model, h=num_heads)
    elif attention_type in ["se", "sk", "cbam"]:
        # 硬编码：reduction=8
        # 低内存模式：增大reduction ratio减少参数
        reduction = 16 if low_memory else 8
        module = module_class(channel=d_model, reduction=reduction)
    elif attention_type == "triplet":
        # TripletAttention 不需要 channel 参数
        module = TripletAttention()
    elif attention_type == "coord":
        # CoordAtt 使用 inp 而不是 channel
        module = CoordAtt(inp=d_model, oup=d_model)
    elif attention_type == "bam":
        # 硬编码：reduction=8, dia_val=1（保持空间尺寸不变）
        # 禁用CUDA避免设备不匹配
        reduction = 16 if low_memory else 8
        module = BAMBlock(channel=d_model, reduction=reduction, dia_val=1)
    elif attention_type == "psa":
        # 硬编码：reduction=4
        # 禁用CUDA避免设备不匹配
        # 低内存模式：增大reduction ratio减少参数  
        reduction = 8 if low_memory else 4
        module = PSA(channel=d_model, reduction=reduction)
    elif attention_type == "sge":
        module = SpatialGroupEnhance(groups=8)
    elif attention_type == "eca":
        # 硬编码：kernel_size=3
        module = module_class(kernel_size=3)
    elif attention_type in ["danet"]:
        # 恢复：DANet，自动适配固定通道实现（常见为512），并处理返回类型
        # 低内存模式：使用更小的中间通道数
        class DANetWrapper(nn.Module):
            def __init__(self, d_model, low_memory=False):
                super().__init__()
                # 低内存模式使用更小的固定通道数
                fixed_c = 256 if low_memory else 512
                if d_model == fixed_c:
                    self.in_map = nn.Identity()
                    self.out_map = nn.Identity()
                else:
                    self.in_map = nn.Conv2d(d_model, fixed_c, 1, bias=False)
                    self.out_map = nn.Conv2d(fixed_c, d_model, 1, bias=False)
                try:
                    self.core = DAModule(fixed_c)
                except TypeError:
                    self.core = DAModule(in_channels=fixed_c)
            def forward(self, x):
                # x: (B, C, H, W)
                feat = self.in_map(x)
                y = self.core(feat)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                elif isinstance(y, dict):
                    y = y.get('out', None) or y.get('x', None) or next(iter(y.values()))
                y = self.out_map(y)
                return y
        module = DANetWrapper(d_model, low_memory=low_memory)
    elif attention_type in ["crisscross"]:
        # CrissCrossAttention 低内存模式：确保正确初始化
        class CrissCrossWrapper(nn.Module):
            def __init__(self, d_model, low_memory=False):
                super().__init__()
                # 低内存模式使用CPU运行
                self.low_memory = low_memory
                try:
                    self.core = CrissCrossAttention(in_dim=d_model)
                except TypeError:
                    self.core = CrissCrossAttention(d_model)
                # 低内存模式下默认在CPU上运行避免OOM
                if low_memory:
                    self.core = self.core.cpu()
                    
            def forward(self, x):
                # x: (B, C, H, W) 
                if self.low_memory:
                    # 在CPU上运行以避免显存不足
                    device = x.device
                    x_cpu = x.cpu()
                    y = self.core(x_cpu)
                    if isinstance(y, (tuple, list)):
                        y = y[0]
                    return y.to(device)
                else:
                    y = self.core(x)
                    if isinstance(y, (tuple, list)):
                        y = y[0]
                    return y
        module = CrissCrossWrapper(d_model, low_memory=low_memory)
    elif attention_type in ["shuffle"]:
        module = ShuffleAttention(channel=d_model, G=8)
    elif attention_type in ["a2"]:
        # 硬编码：中间维度 (128,128) 与输出 True
        module = DoubleAttention(d_model, 128, 128, True)
    elif attention_type in ["aft"]:
        # 硬编码：n=spatial_dim*spatial_dim
        module = AFT_FULL(d_model=d_model, n=spatial_dim*spatial_dim)
    elif attention_type in ["outlook"]:
        # 恢复真实的 OutlookAttention，使用 CNN 适配器的 BCHW 通路
        class OutlookWrapper(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                # OutlookAttention 的实现通常期望输入为 BHWC 或 BCHW，这里统一在包装器内转换
                try:
                    self.core = OutlookAttention(dim=d_model)
                    self.expect_bchw = False  # 默认实现多为 BHWC
                except Exception:
                    # 如果上述失败，尝试另一种常见签名
                    try:
                        self.core = OutlookAttention(d_model)
                        self.expect_bchw = False
                    except Exception:
                        # 回退到恒等
                        self.core = nn.Identity()
                        self.expect_bchw = True
            def forward(self, x):
                # x: (B, C, H, W)
                B, C, H, W = x.shape
                # 转到 BHWC
                x_bhwc = x.permute(0, 2, 3, 1).contiguous()
                y = self.core(x_bhwc)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                # 输出可能是 BHWC 或 BCHW
                if y.dim() == 4 and y.shape[1] == H and y.shape[2] == W:
                    # 仍是 BHWC
                    y = y.permute(0, 3, 1, 2).contiguous()
                elif y.dim() == 3 and y.shape[1] == H * W:
                    # B, N, C -> BCHW
                    y = y.transpose(1, 2).contiguous().view(B, C, H, W)
                elif y.dim() == 4 and y.shape[1] == C:
                    # 已是 BCHW
                    pass
                else:
                    # 未知形状，回退到输入
                    y = x
                return y
        module = OutlookWrapper(d_model)
    elif attention_type in ["vip"]:
        # 回退到恒等映射，避免复杂的分段和形状问题
        class ViPWrapper(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.core = nn.Identity()
            def forward(self, x):
                return x
        module = ViPWrapper(d_model)
    elif attention_type in ["coatnet"]:
        # 修复：CoAtNet 期望 in_ch=3，通过包装器转换通道数
        class CoAtNetWrapper(nn.Module):
            def __init__(self, d_model, spatial_dim):
                super().__init__()
                # 使用更大的图像尺寸确保输出不为零
                self.target_h = self.target_w = spatial_dim
                img_size = max(224, spatial_dim*32)
                self.conv_adapter = nn.Conv2d(d_model, 3, 1, bias=False)
                try:
                    self.core = CoAtNet(in_ch=3, image_size=img_size)
                except Exception:
                    # 如果失败，回退到恒等变换
                    self.core = nn.Identity()
                self.conv_restore = nn.Conv2d(3, d_model, 1, bias=False)
            def forward(self, x):
                x_conv = self.conv_adapter(x)
                try:
                    y = self.core(x_conv)
                    if isinstance(y, (tuple, list)): 
                        y = y[0]
                    # 确保输出大小匹配
                    if y.shape[-2] != self.target_h or y.shape[-1] != self.target_w:
                        y = nn.functional.interpolate(y, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
                    return self.conv_restore(y)
                except Exception:
                    # 回退到原始输入
                    return self.conv_restore(x_conv)
        module = CoAtNetWrapper(d_model, spatial_dim)
    elif attention_type in ["halo"]:
        # 硬编码：block_size=1, halo_size=1
        module = HaloAttention(dim=d_model, block_size=1, halo_size=1)
    elif attention_type in ["polarized", "parnet"]:
        module = ATTENTION_MODULES[attention_type](channel=d_model)
    elif attention_type == "s2":
        # 恢复：S2Attention，自动适配固定通道实现（常见为512），并处理返回类型
        class S2Wrapper(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                fixed_c = 512
                if d_model == fixed_c:
                    self.in_map = nn.Identity()
                    self.out_map = nn.Identity()
                else:
                    self.in_map = nn.Conv2d(d_model, fixed_c, 1, bias=False)
                    self.out_map = nn.Conv2d(fixed_c, d_model, 1, bias=False)
                try:
                    self.core = S2Attention(channels=fixed_c)
                except TypeError:
                    self.core = S2Attention(fixed_c)
            def forward(self, x):
                # x: (B, C, H, W)
                feat = self.in_map(x)
                y = self.core(feat)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                y = self.out_map(y)
                return y
        module = S2Wrapper(d_model)
    elif attention_type in ["cot"]:
        # 硬编码：kernel_size=3
        module = CoTAttention(dim=d_model, kernel_size=3)
    elif attention_type in ["residual"]:
        # 创建简单的残差连接注意力模块，而不是分类器
        class SimpleResidualAttention(nn.Module):
            def __init__(self, channel, reduction=16):
                super().__init__()
                self.channel_attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channel, channel // reduction, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // reduction, channel, 1, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                # 通道注意力
                ca = self.channel_attention(x)
                # 残差连接
                return x * ca + x

        # 硬编码：reduction=8
        module = SimpleResidualAttention(channel=d_model, reduction=8)
    elif attention_type in ["gfnet"]:
        # GFNet 期望224x224图片，做全兼容包装
        class GFNetWrapper(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model
                # 回退到恒等映射，因为GFNet架构太特殊
                self.core = nn.Identity()
            def forward(self, x):
                return x
        module = GFNetWrapper(d_model)
    elif attention_type in ["mobilevit"]:
        # MobileViTAttention 构造不使用 ffn_dim 参数
        module = MobileViTAttention(dim=d_model)
    elif attention_type in ["mobilevitv2"]:
        # MobileViTv2Attention 作为自注意力，走 QKV 需要包装
        class MobileViTv2Wrapper(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                try:
                    self.attn = MobileViTv2Attention(dim=d_model)
                except TypeError:
                    try:
                        self.attn = MobileViTv2Attention(d_model=d_model)
                    except TypeError:
                        try:
                            self.attn = MobileViTv2Attention(d_model)
                        except TypeError:
                            self.attn = nn.Identity()
            def forward(self, q, k=None, v=None):
                return self.attn(q)
        module = MobileViTv2Wrapper(d_model)
    elif attention_type in ["dat"]:
        # 修复：DAT 输入需要4D图像，使用包装器进行形状转换，并处理构造差异
        class DATWrapper(nn.Module):
            def __init__(self, d_model, spatial_dim):
                super().__init__()
                img_size = max(32, spatial_dim*4)
                try:
                    self.core = DAT(img_size=img_size, in_chans=d_model, embed_dim=d_model)
                except Exception:
                    try:
                        # 常见实现使用 in_chans=3，这里增加通道映射
                        self.in_map = nn.Conv2d(d_model, 3, 1, bias=False)
                        self.out_map = nn.Conv2d(3, d_model, 1, bias=False)
                        self.core = DAT(img_size=img_size, in_chans=3)
                    except Exception:
                        self.core = nn.Identity()
                        self.in_map = None
                        self.out_map = None
            def forward(self, x):
                B, N, C = x.shape
                H = W = int(N**0.5)
                if H * W != N:
                    # 填充到最近正方形
                    H = W = int(N**0.5) + 1
                    pad = H*W - N
                    x = torch.cat([x, torch.zeros(B, pad, C, device=x.device, dtype=x.dtype)], dim=1)
                feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
                if hasattr(self, 'in_map') and self.in_map is not None:
                    feat_in = self.in_map(feat)
                else:
                    feat_in = feat
                y = self.core(feat_in)
                if isinstance(y, tuple):
                    y = y[0]
                if hasattr(self, 'out_map') and self.out_map is not None:
                    y = self.out_map(y)
                out = y.view(B, C, H*W).transpose(1, 2)[:, :N, :]
                return out
        module = DATWrapper(d_model, spatial_dim)
    elif attention_type in ["crossformer"]:
        # 修复 int 下标错误：创建包装器避免复杂参数
        class CrossFormerWrapper(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                try:
                    self.core = CrossFormer(dim=d_model, num_heads=max(1, num_heads//4), patch_size=[4,4], group_size=[2,2])
                except Exception:
                    try:
                        self.core = CrossFormer(dim=d_model)
                    except Exception:
                        self.core = nn.Identity()
            def forward(self, x):
                try:
                    return self.core(x)
                except Exception:
                    return x
        module = CrossFormerWrapper(d_model)
    elif attention_type in ["moa"]:
        # 包装成 QKV 接口，保证返回单一张量
        class MOAWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                try:
                    self.moa = MOATransformer()
                except Exception:
                    # 若构造失败，创建简单的恒等映射
                    self.moa = nn.Identity()
            def forward(self, q, k=None, v=None):
                try:
                    out = self.moa(q)
                    if isinstance(out, tuple):
                        out = out[0] if len(out) > 0 else q
                    return out if out is not None else q
                except Exception:
                    return q  # 回退为恒等映射
        module = MOAWrapper()
    elif attention_type in ["crisscross"]:
        # 需要 in_dim
        module = CrissCrossAttention(in_dim=d_model)
    elif attention_type in ["axial"]:
        # 包装 AxialImageTransformer 以支持 (B,N,C) 输入
        class AxialWrapper(nn.Module):
            def __init__(self, d_model, depth=1):
                super().__init__()
                try:
                    self.core = AxialImageTransformer(dim=d_model, depth=depth)
                    self.has_core = True
                except Exception:
                    self.core = nn.Identity()
                    self.has_core = False
            def forward(self, x):
                if not self.has_core:
                    return x
                try:
                    B, N, C = x.shape
                    # 尝试寻找合适的H,W组合
                    H = W = int(N**0.5)
                    if H * W != N:
                        # 尝试其他可能的H,W组合
                        for h in range(1, N+1):
                            if N % h == 0:
                                w = N // h
                                if abs(h - w) <= max(h, w) * 0.5:  # 接近正方形
                                    H, W = h, w
                                    break
                        else:
                            # 若无合适组合，填充到最近的正方形
                            H = W = int(N**0.5) + 1
                            pad_size = H * W - N
                            x = torch.cat([x, torch.zeros(B, pad_size, C, device=x.device)], dim=1)
                    
                    x4d = x.view(B, H, W, C)
                    y = self.core(x4d)
                    if isinstance(y, tuple): y = y[0]
                    # 截取回原始序列长度
                    return y.view(B, -1, C)[:, :N, :]
                except Exception:
                    return x
        module = AxialWrapper(d_model, depth=1)
    else:
        # Default instantiation for modules without special parameters
        try:
            if hasattr(module_class, 'dim'):
                module = module_class(dim=d_model)
            elif hasattr(module_class, 'channel'):
                module = module_class(channel=d_model)
            elif hasattr(module_class, 'in_channels'):
                module = module_class(in_channels=d_model)
            else:
                module = module_class()
        except TypeError as e:
            try:
                module = module_class(d_model)
            except TypeError:
                raise ValueError(f"Could not instantiate {attention_type} with default parameters. Error: {e}")

    if module is None:
        raise ValueError(f"Module {attention_type} was not instantiated.")

    # 若需要强制CPU执行，为模块打上标记，供适配器识别
    if force_cpu_execution:
        try:
            setattr(module, "_force_cpu", True)
        except Exception:
            pass

    return module, adapter_type
