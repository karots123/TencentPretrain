import torch.nn as nn
from tencentpretrain.layers.multi_headed_attn import MultiHeadedAttention
from tencentpretrain.layers import *

class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args, layer_number=None):
        super(TransformerLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        if hasattr(args, "local_kv_heads_num"):
            local_kv_heads_num = args.local_kv_heads_num
        else:
            local_kv_heads_num = args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        lora_params = None
        if hasattr(args, "lora_params"):
            lora_params = args.lora_params

        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, local_kv_heads_num, args.dropout, has_bias=has_bias,
            with_scale = with_scale, lora_params=lora_params, layer_number=layer_number
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        self.feed_forward = str2feedforward[args.feed_forward](
            args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        self.layer_norm_1 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)
        self.layer_norm_2 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)

    def forward(self, hidden, mask, position_bias=None, has_residual_attention=False,
                prev_attn=None, freqs_cis=None, alibi=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            inter, prev_attn_out = self.self_attn(hidden, hidden, hidden, mask, position_bias, has_residual_attention,
                                                  prev_attn, freqs_cis, alibi)
            inter = self.dropout_1(inter)
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            inter, prev_attn_out = self.self_attn(inter, inter, inter, mask, position_bias, has_residual_attention,
                                                  prev_attn, freqs_cis, alibi)
            inter = self.dropout_1(inter)
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output = self.dropout_2(self.feed_forward(output)) + hidden
        return output, prev_attn_out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        if hasattr(args, "local_kv_heads_num"):
            local_kv_heads_num = args.local_kv_heads_num
        else:
            local_kv_heads_num = args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        lora_params = None
        if hasattr(args, "lora_params"):
            lora_params = args.lora_params

        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, local_kv_heads_num, args.dropout, has_bias=has_bias,
            with_scale=with_scale, lora_params=lora_params
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Multi-headed context-attention.
        self.context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, local_kv_heads_num, args.dropout, has_bias=has_bias,
            with_scale=with_scale, lora_params=lora_params
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        # Feed forward layer.
        self.feed_forward = str2feedforward[args.feed_forward](
            args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
        )
        self.dropout_3 = nn.Dropout(args.dropout)

        # Layer Normalization
        self.layer_norm_1 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)
        self.layer_norm_2 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)
        self.layer_norm_3 = str2layernorm[args.layernorm](args.hidden_size, eps=args.layernorm_eps)

    def forward(self, hidden, encoder_hidden, mask_decoder, mask_encoder, self_position_bias=None, context_position_bias=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            encoder_hidden: [batch_size x seq_length x emb_size]
            mask_encoder: [batch_size x 1 x seq_length x seq_length]
            mask_decoder: [batch_size x 1 x seq_length x seq_length]
            self_position_bias: [1 x heads_num x seq_length x seq_length]
            context_position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            query, _ = self.self_attn(hidden, hidden, hidden, mask_decoder, self_position_bias)
            query = self.dropout_1(query)
            query_norm = self.layer_norm_1(query + hidden)
            mid, _ = self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias)
            mid = self.dropout_2(mid)
            mid_norm = self.layer_norm_2(mid + query_norm)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = self.layer_norm_3(output + mid_norm)
        else:
            hidden_norm = self.layer_norm_1(hidden)
            query, _ = self.self_attn(hidden_norm, hidden_norm, hidden_norm, mask_decoder, self_position_bias)
            query = self.dropout_1(query)
            query = query + hidden
            query_norm = self.layer_norm_2(query)
            mid, _ = self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias)
            mid = self.dropout_2(mid)
            mid = mid + query
            mid_norm = self.layer_norm_3(mid)
            output = self.dropout_3(self.feed_forward(mid_norm)) + mid
        return output
