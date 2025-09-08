import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.atten import PointWiseFeedForward


class PointwiseAggregatedAttention(nn.Module):
    def __init__(self, d_model, 
                 num_heads,     
                 relative_attention_num_buckets,
                 relative_attention_bucket_dim,
                 relative_attention_max_distance):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # TODO: add relative attention bias based on time
        self.rab_p = RelativeAttentionBias(num_heads, 
                                           relative_attention_num_buckets,
                                           relative_attention_bucket_dim,
                                           relative_attention_max_distance)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5

        att_w_bias = F.silu(attention_scores)
        att_w_bias = att_w_bias * mask[:,None]
        av = (att_w_bias @ v)
        return av.transpose(1, 2).flatten(2)

class RelativeTimeIntervalBias(nn.Module):
    def __init__(self,
                 max_time_interval, 
                 time_interval_dim,
                 num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.time_bias = nn.Sequential(nn.Embedding(max_time_interval + 1, time_interval_dim),
                                       nn.Linear(time_interval_dim, num_heads))
        
        
    def forward(self, input_time_matrix):
        # B x seq_len 
        return self.time_bias(input_time_matrix).permute(0,3,1,2)

class RelativeAttentionBias(nn.Module):
    def __init__(self, 
                 num_heads, 
                 relative_attention_num_buckets, 
                 relative_attention_bucket_dim, 
                 relative_attention_max_distance):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = nn.Sequential(nn.Embedding(relative_attention_num_buckets + 1, relative_attention_bucket_dim),
                                                     nn.Linear(relative_attention_bucket_dim, num_heads))

    def forward(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    # https://github.com/huggingface/transformers/blob/6cdbd73e01a9719bfaec07d91fd108e8d932bbbb/src/transformers/models/t5/modeling_t5.py#L384
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets


class HSTUBlock(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_heads,
                 max_time_interval, 
                 time_interval_dim, 
                 relative_attention_num_buckets,
                 relative_attention_bucket_dim,                 
                 relative_attention_max_distance, 
                 ):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_model * 4)  # Transform and split
        self.pointwise_attn = PointwiseAggregatedAttention(d_model, 
                                                           num_heads, 
                                                           relative_attention_num_buckets, 
                                                           relative_attention_bucket_dim,
                                                           relative_attention_max_distance)
        self.f2 = nn.Linear(d_model, d_model)
        self.norm = nn.RMSNorm(d_model)

    def split(self, x):
        u, v, q, k = x.chunk(4, dim=-1)
        return u, v, q, k

    def forward(self, x, attn_mask):
        # Pointwise Projection
        x_proj = F.silu(self.f1(x))
        u, v, q, k = self.split(x_proj)

        # Spatial Aggregation
        av = self.pointwise_attn(v, k, q, attn_mask)

        # Pointwise Transformation
        y = self.f2(self.norm(av * u))

        return y

class HstuAttentionDecoder(nn.Module):
    def __init__(self, 
                 blocks, 
                 hidden_units, 
                 num_heads, 
                 dropout_rate, 
                 norm_first, 
                 relative_attention_num_buckets, 
                 relative_attention_bucket_dim, 
                 relative_attention_max_distance):
        super().__init__()
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.norm_first = norm_first
        for _ in range(blocks):
            new_attn_layernorm = torch.nn.RMSNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = HSTUBlock(
                hidden_units, 
                num_heads, 
                max_time_interval, 
                time_interval_dim, 
                relative_attention_num_buckets, 
                relative_attention_bucket_dim, 
                relative_attention_max_distance
            )  
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.RMSNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        
        self.last_layernorm = torch.nn.RMSNorm(hidden_units, eps=1e-8)
        
    def forward(self, seqs, attention_mask):
        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs = self.attention_layers[i](x, attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs = self.attention_layers[i](seqs, attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
        log_feats = self.last_layernorm(seqs)
        return log_feats

