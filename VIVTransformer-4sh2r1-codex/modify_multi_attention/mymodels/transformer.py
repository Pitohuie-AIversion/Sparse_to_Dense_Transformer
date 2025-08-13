from typing import List

import torch
import torch.nn as nn
import copy
import math

from .components.attention_adapter import (
    get_attention_adapter,
)
from .components.attention_factory import (
    get_attention_module,
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

from ..utils.model_utils import clones
from .embedding import EmbeddingAndEncoding
from .embedding_2d import EmbeddingAndEncoding2D

def _create_attention_layer(attention_type, d_model, num_heads, seq_len=49, low_memory=False):
    """Helper function to create an attention layer."""
    spatial_dim = int(seq_len**0.5)
    module, adapter_type = get_attention_module(
        attention_type, d_model=d_model, num_heads=num_heads, spatial_dim=spatial_dim, low_memory=low_memory
    )
    return get_attention_adapter(module, adapter_type)


class CustomEncoderLayer(nn.Module):
    """A single layer of the custom Transformer encoder.

    This layer includes a self-attention mechanism and a feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        attention_type: str = "self",
        dropout: float = 0.1,
        seq_len: int = 49,
        low_memory: bool = False,
    ):
        super().__init__()
        self.self_attn = _create_attention_layer(attention_type, d_model, num_heads, seq_len=seq_len, low_memory=low_memory)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化和残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """Passes the input through the encoder layer.

        Args:
            src: The sequence to the encoder layer.

        Returns:
            The output from the encoder layer.
        """
        # Self-Attention
        attn_output = self.self_attn(src, return_attention=return_attention)
        if return_attention:
            if isinstance(attn_output, tuple) and len(attn_output) == 2:
                src2, attn_weights = attn_output
            else:
                src2 = attn_output
                attn_weights = None
        else:
            src2 = attn_output

        if isinstance(src2, tuple):
            src2 = src2[0]

        # 如果需要注意力权重但未提供，则构造一个替代注意力图 [B, 1, L, L]
        if return_attention and attn_weights is None:
            with torch.no_grad():
                B, L, D = src.shape
                scale = 1.0 / math.sqrt(max(D, 1))
                sim = torch.matmul(src, src.transpose(1, 2)) * scale  # [B, L, L]
                attn_weights = torch.softmax(sim, dim=-1).unsqueeze(1)  # [B, 1, L, L]

        # 残差连接 + LayerNorm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward 网络
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if return_attention:
            return src, attn_weights
        return src


class CustomDecoderLayer(nn.Module):
    """A single layer of the custom Transformer decoder.

    This layer includes self-attention, cross-attention, and a feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_type: str = "relative",
        seq_len: int = 49,
        low_memory: bool = False,
    ):
        """Initializes the CustomDecoderLayer.

        Args:
            d_model: The number of expected features in the input.
            num_heads: The number of heads in the multiheadattention models.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            attention_type: The type of attention mechanism to use.
            
            提示：dim_feedforward 与 dropout 当前在此处使用默认值（硬编码默认）。
            如果需要可配置化，建议在 config.yaml 中新增：
              model.dim_feedforward, model.dropout
            然后在模型构造时传入。
        """
        super().__init__()

        self.self_attn = _create_attention_layer(attention_type, d_model, num_heads, seq_len=seq_len, low_memory=low_memory)
        self.multihead_attn = _create_attention_layer(
            attention_type, d_model, num_heads, seq_len=seq_len, low_memory=low_memory
        )

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化和残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """Passes the inputs through the decoder layer.

        Args:
            tgt: The sequence to the decoder layer.
            memory: The sequence from the last layer of the encoder.

        Returns:
            The output from the decoder layer.
        """
        # Self-Attention
        self_attn_output = self.self_attn(tgt, return_attention=return_attention)
        if return_attention:
            if isinstance(self_attn_output, tuple) and len(self_attn_output) == 2:
                tgt2, self_attn_weights = self_attn_output
            else:
                tgt2 = self_attn_output
                self_attn_weights = None
        else:
            tgt2 = self_attn_output

        if isinstance(tgt2, tuple):
            tgt2 = tgt2[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-Attention (Encoder-Decoder Attention)
        cross_attn_output = self.multihead_attn(
            tgt, memory, return_attention=return_attention
        )
        if return_attention:
            if isinstance(cross_attn_output, tuple) and len(cross_attn_output) == 2:
                tgt2, cross_attn_weights = cross_attn_output
            else:
                tgt2 = cross_attn_output
                cross_attn_weights = None
        else:
            tgt2 = cross_attn_output

        if isinstance(tgt2, tuple):
            tgt2 = tgt2[0]

        # 对于缺失的注意力，构造替代注意力 [B, 1, L, L]
        if return_attention:
            with torch.no_grad():
                B, L, D = tgt.shape
                scale = 1.0 / math.sqrt(max(D, 1))
                if self_attn_weights is None:
                    sim_self = torch.matmul(tgt, tgt.transpose(1, 2)) * scale  # [B, L, L]
                    self_attn_weights = torch.softmax(sim_self, dim=-1).unsqueeze(1)
                if cross_attn_weights is None:
                    # memory 形状与 tgt 对齐
                    if memory.dim() == 3 and memory.size(0) == B and memory.size(2) == D:
                        sim_cross = torch.matmul(tgt, memory.transpose(1, 2)) * scale  # [B, L, L]
                        cross_attn_weights = torch.softmax(sim_cross, dim=-1).unsqueeze(1)
                    else:
                        # 回退到自注意力形式，至少提供一个稳定的可视化
                        sim_cross = torch.matmul(tgt, tgt.transpose(1, 2)) * scale
                        cross_attn_weights = torch.softmax(sim_cross, dim=-1).unsqueeze(1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if return_attention:
            return tgt, (self_attn_weights, cross_attn_weights)
        return tgt


class CustomEncoder(nn.Module):
    """A stack of N custom encoder layers."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        """Initializes the CustomEncoder.

        Args:
            encoder_layer: An instance of the CustomEncoderLayer.
            num_layers: The number of sub-encoder-layers in the encoder.
        """
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Passes the input through the stack of encoder layers.

        Args:
            src: The sequence to the encoder.

        Returns:
            The output from the encoder.
        """
        for layer in self.layers:
            src = layer(src)
        return src


class CustomDecoder(nn.Module):
    """A stack of N custom decoder layers."""

    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        """Initializes the CustomDecoder.

        Args:
            decoder_layer: An instance of the CustomDecoderLayer.
            num_layers: The number of sub-decoder-layers in the decoder.
        """
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """Passes the inputs through the stack of decoder layers.

        Args:
            tgt: The sequence to the decoder.
            memory: The sequence from the last layer of the encoder.

        Returns:
            The output from the decoder.
        """
        attention_weights_list = []
        for layer in self.layers:
            layer_output = layer(tgt, memory, return_attention=return_attention)
            if return_attention:
                tgt, attention_weights = layer_output
                attention_weights_list.append(attention_weights)
            else:
                tgt = layer_output

        if return_attention:
            return tgt, attention_weights_list
        return tgt


class TransformerFlowReconstructionModel(nn.Module):
    """The main Transformer model for flow reconstruction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        d_model: int = 512,
        max_time_steps: int = 100,
        attention_type: str = "relative",
        seq_len: int = 49,
        grid_height: int = 7,
        grid_width: int = 7,
        use_2d_embedding: bool = True,
        low_memory: bool = False,
        # 可选：以下两项目前在子层中默认（硬编码默认），如需开放配置，可透传
        # dim_feedforward: int = 2048,
        # dropout: float = 0.1,
    ):
        """Initializes the TransformerFlowReconstructionModel.

        Args:
            input_dim: The dimension of the input features.
            output_dim: The dimension of the output features.
            num_heads: The number of heads in the multiheadattention models.
            num_layers: The number of sub-encoder-layers and sub-decoder-layers.
            d_model: The number of expected features in the encoder/decoder inputs.
            max_time_steps: The maximum number of time steps for time embedding.
            attention_type: The type of attention mechanism to use.
            seq_len: The length of the input sequence.
            grid_height: The height of the 2D grid (for 2D spatial embedding).
            grid_width: The width of the 2D grid (for 2D spatial embedding).
            use_2d_embedding: Whether to use 2D spatial embedding or original embedding.
            low_memory: Enable low-memory mode for heavy attention modules.
            
            提示：如需让 dim_feedforward 与 dropout 也可由配置控制，
                 可将其加入 config.yaml 的 model 字段，并在下方 encoder_layer/decoder_layer 构造时传入。
        """
        super().__init__()

        self.attention_type = attention_type
        self.seq_len = seq_len
        self.use_2d_embedding = use_2d_embedding
        self.low_memory = low_memory
        
        # 选择嵌入模块
        if use_2d_embedding:
            # 使用2D空间嵌入，保留空间邻域信息
            self.embedding_encoding = EmbeddingAndEncoding2D(
                input_dim=1,  # 每个网格点一个压力值
                d_model=d_model,
                max_time_steps=max_time_steps,
                seq_len=seq_len,
                grid_height=grid_height,
                grid_width=grid_width
            )
        else:
            # 使用原始嵌入方法
            self.embedding_encoding = EmbeddingAndEncoding(
                input_dim=input_dim,
                d_model=d_model,
                max_time_steps=max_time_steps,
                seq_len=seq_len
            )

        encoder_layer = CustomEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=2048,  # 硬编码默认，如需可配置化请参考上方注释
            attention_type=self.attention_type,  # 使用实例变量
            seq_len=seq_len,
            low_memory=self.low_memory,
        )
        decoder_layer = CustomDecoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=2048,  # 硬编码默认，如需可配置化请参考上方注释
            attention_type=self.attention_type,  # 使用实例变量
            seq_len=seq_len,
            low_memory=self.low_memory,
        )

        self.encoder = CustomEncoder(encoder_layer, num_layers)
        self.decoder = CustomDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(
        self, x_in_pressures_flat: torch.Tensor, x_time_steps: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        """Defines the forward pass of the model.

        Args:
            x_in_pressures_flat: The flattened input pressure data.
            x_time_steps: The time steps corresponding to the input data.

        Returns:
            The predicted flattened output pressure data.
        """
        x_embedded = self.embedding_encoding(x_in_pressures_flat, x_time_steps)
        encoder_output = self.encoder(x_embedded)
        
        # Create a separate target sequence for decoder (learnable or zero-initialized)
        # This prevents cross-attention from degenerating into self-attention
        batch_size, seq_len, d_model = encoder_output.shape
        decoder_target = torch.zeros_like(encoder_output)  # Initialize with zeros
        # Alternative: use a learnable parameter
        # decoder_target = self.decoder_start_token.expand(batch_size, seq_len, -1)
        
        decoder_raw_output = self.decoder(decoder_target, encoder_output, return_attention=return_attention)

        attention_weights = None
        if return_attention:
            if isinstance(decoder_raw_output, tuple) and len(decoder_raw_output) == 2:
                decoder_output, attention_weights = decoder_raw_output
            else:
                decoder_output = decoder_raw_output
        else:
            decoder_output = decoder_raw_output

        decoder_output_mean = decoder_output.mean(dim=1)
        out_pressure_flat_pred = self.fc_out(decoder_output_mean)

        if return_attention:
            return out_pressure_flat_pred, attention_weights
        return out_pressure_flat_pred
