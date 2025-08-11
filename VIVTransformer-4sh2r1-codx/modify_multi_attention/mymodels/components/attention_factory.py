import torch.nn as nn
from mymodels.components.attention_adapter import AdapterType
from mymodels.components.attention import (
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

# from fightingcv_attention.attention.ACmix import ACmix
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
    "gfnet": GFNet,  # Has its own forward but fits QKV pattern conceptually

    # CNN-style attentions
    "se": SEAttention,
    "sk": SKAttention,
    "cbam": CBAMBlock,
    "bam": BAMBlock,
    "eca": ECAAttention,
    "shuffle": ShuffleAttention,

    "residual": ResidualAttention,
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
    # QKV Compatible - tested and working
    "self": AdapterType.QKV,
    "simplified_self": AdapterType.QKV,
    "muse": AdapterType.QKV,
    "ufo": AdapterType.QKV,
    "relative": AdapterType.QKV,
    "sparse": AdapterType.QKV,
    "lsh": AdapterType.QKV,
    
    # CNN Compatible - tested and working
    "se": AdapterType.CNN,
    "sk": AdapterType.CNN,
    "cbam": AdapterType.CNN,
    "bam": AdapterType.CNN,
    "eca": AdapterType.CNN,
    "shuffle": AdapterType.CNN,
    "residual": AdapterType.CNN,
    "psa": AdapterType.CNN,
    "cot": AdapterType.CNN,
    "polarized": AdapterType.CNN,
    "halo": AdapterType.CNN,
    "a2": AdapterType.CNN,  # Actually works with CNN format
    "parnet": AdapterType.CNN,
    
    # Single Input Compatible - tested and working
    "external": AdapterType.SINGLE_INPUT,
    "aft": AdapterType.SINGLE_INPUT,
    
    # Incompatible or require special handling - commented out for now
    # "emsa": AdapterType.QKV,  # Missing required parameters
    # "danet": AdapterType.CNN,  # Channel mismatch issues
    # "outlook": AdapterType.SINGLE_INPUT,  # Matrix multiplication issues
    # "vip": AdapterType.SINGLE_INPUT,  # Matrix multiplication issues
    # "coatnet": AdapterType.CNN,  # Output size too small
    # "gfnet": AdapterType.SINGLE_INPUT,  # Forward method signature issues
    # "mobilevit": AdapterType.SINGLE_INPUT,  # Parameter issues
    # "mobilevitv2": AdapterType.SINGLE_INPUT,  # Forward method signature issues
    # "dat": AdapterType.SINGLE_INPUT,  # Groups parameter issues
    # "crossformer": AdapterType.QKV,  # Subscriptable issues
    # "moa": AdapterType.SINGLE_INPUT,  # Incompatible with sequence data
    # "crisscross": AdapterType.SINGLE_INPUT,  # Forward method signature issues
    # "axial": AdapterType.SINGLE_INPUT,  # Missing required parameters
    # "s2": AdapterType.CNN,  # Parameter name issues
    # "triplet": AdapterType.CNN,  # Parameter name issues
    # "coord": AdapterType.CNN,  # Parameter name issues
}


# Only include tested and compatible attention mechanisms
ATTENTION_MODULES = {
    # QKV Compatible - tested and working
    "self": ScaledDotProductAttention,
    "simplified_self": SimplifiedScaledDotProductAttention,
    "muse": MUSEAttention,
    "ufo": UFOAttention,
    "relative": RelativePositionSelfAttention,
    "sparse": SparseSelfAttention,
    "lsh": LSHSelfAttention,
    
    # CNN Compatible - tested and working
    "se": SEAttention,
    "sk": SKAttention,
    "cbam": CBAMBlock,
    "bam": BAMBlock,
    "eca": ECAAttention,
    "shuffle": ShuffleAttention,
    "residual": ResidualAttention,
    "psa": PSA,
    "cot": CoTAttention,
    "polarized": SequentialPolarizedSelfAttention,
    "halo": HaloAttention,
    "a2": DoubleAttention,
    "parnet": ParNetAttention,
    
    # Single Input Compatible - tested and working
    "external": ExternalAttention,
    "aft": AFT_FULL,
    
    # Incompatible modules - commented out to prevent errors
    # "emsa": EMSA,  # Missing required parameters
    # "danet": DAModule,  # Channel mismatch issues
    # "outlook": OutlookAttention,  # Matrix multiplication issues
    # "vip": WeightedPermuteMLP,  # Matrix multiplication issues
    # "coatnet": CoAtNet,  # Output size too small
    # "gfnet": GFNet,  # Forward method signature issues
    # "mobilevit": MobileViTAttention,  # Parameter issues
    # "mobilevitv2": MobileViTv2Attention,  # Forward method signature issues
    # "dat": DAT,  # Groups parameter issues
    # "crossformer": CrossFormer,  # Subscriptable issues
    # "moa": MOATransformer,  # Incompatible with sequence data
    # "crisscross": CrissCrossAttention,  # Forward method signature issues
    # "axial": AxialImageTransformer,  # Missing required parameters
    # "s2": S2Attention,  # Parameter name issues
    # "triplet": TripletAttention,  # Parameter name issues
    # "coord": CoordAtt,  # Parameter name issues
    # "acmix": ACmix,  # Not imported
}

def get_attention_module(
    attention_type, d_model=512, num_heads=8, spatial_dim=7, **kwargs
):
    """
    根据 attention_type 返回对应的注意力模块和适配器类型
    """
    """
    根据 attention_type 返回对应的注意力模块，适配不同注意力模块的参数
    """
    if attention_type not in ATTENTION_MODULES:
        raise ValueError(f"Unknown attention type: {attention_type}")

    adapter_type = ADAPTER_MAPPING.get(attention_type, "qkv")  # Default to QKV
    module_class = ATTENTION_MODULES[attention_type]
    module = None

    # Adapt parameters for different attention modules
    if attention_type == "relative":
        module = RelativePositionSelfAttention(d_model, num_heads)
    elif attention_type == "sparse":
        module = SparseSelfAttention(d_model, num_heads)
    elif attention_type == "lsh":
        module = LSHSelfAttention(d_model, num_heads)
    elif attention_type in ["external"]:
        module = module_class(d_model=d_model, S=8)
    elif attention_type in ["self", "muse", "ufo"]:
        module = module_class(d_model=d_model, d_k=d_model, d_v=d_model, h=num_heads)
    elif attention_type == "simplified_self":
        module = module_class(d_model=d_model, h=num_heads)
    elif attention_type in ["se", "sk", "cbam", "bam", "triplet", "coord", "psa"]:
        module = module_class(channel=d_model, reduction=8)
    elif attention_type == "sge":
        module = module_class(groups=8)
    elif attention_type == "eca":
        module = module_class(kernel_size=3)
    elif attention_type in ["danet"]:
        module = module_class(d_model=d_model, kernel_size=3, H=spatial_dim, W=spatial_dim)
    elif attention_type in ["shuffle"]:
        module = module_class(channel=d_model, G=8)
    elif attention_type in ["a2"]:
        module = module_class(d_model, 128, 128, True)
    elif attention_type in ["aft"]:
        module = module_class(d_model=d_model, n=spatial_dim*spatial_dim)
    elif attention_type in ["outlook"]:
        module = module_class(dim=d_model)
    elif attention_type in ["vip"]:
        module = module_class(d_model, seg_dim=8)
    elif attention_type in ["coatnet"]:
        module = module_class(in_ch=d_model, image_size=spatial_dim)
    elif attention_type in ["halo"]:
        module = module_class(dim=d_model, block_size=1, halo_size=1)
    elif attention_type in ["polarized", "parnet", "s2"]:
        module = module_class(channel=d_model)
    elif attention_type in ["cot"]:
        module = module_class(dim=d_model, kernel_size=3)
    elif attention_type in ["residual"]:
        module = module_class(channel=d_model, num_class=d_model, la=0.2)
    elif attention_type in ["gfnet"]:
        # Parameters for GFNet might need specific configuration
        module = module_class(embed_dim=d_model, img_size=spatial_dim)
    elif attention_type in ["mobilevit"]:
        module = module_class(dim=d_model, ffn_dim=d_model*2, spatial_dims=spatial_dim) # Provide default dims
    elif attention_type in ["mobilevitv2"]:
        module = module_class(d_model=d_model)
    elif attention_type in ["dat"]:
        # DAT has complex parameters, may need a more robust configuration scheme
        module = module_class(img_size=spatial_dim, in_chans=3, embed_dim=d_model, num_heads=num_heads) # Corrected in_chans
    elif attention_type in ['crossformer']:
        module = module_class(dim=d_model, num_heads=num_heads, patch_size=[spatial_dim, spatial_dim], group_size=[spatial_dim//2, spatial_dim//2], img_size=spatial_dim, in_chans=3)
    else:
        # Default instantiation for modules without special parameters
        try:
            # Attempt to instantiate with common parameter names
            if hasattr(module_class, 'dim'):
                module = module_class(dim=d_model)
            elif hasattr(module_class, 'channel'):
                module = module_class(channel=d_model)
            elif hasattr(module_class, 'in_channels'):
                module = module_class(in_channels=d_model)
            else:
                module = module_class()
        except TypeError as e:
            # Fallback for modules that require at least the dimension/channel
            try:
                module = module_class(d_model)
            except TypeError:
                raise ValueError(f"Could not instantiate {attention_type} with default parameters. Error: {e}")

    if module is None:
         raise ValueError(f"Module {attention_type} was not instantiated.")

    return module, adapter_type
