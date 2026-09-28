import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from moment_detr.deformable_transformer import DeformableTransformerEncoderLayer


class TextEncoder(nn.Module):
    def __init__(self, hidden_size, drop, input_drop, nheads, max_position_embeddings):
        super().__init__()
        # Replace BertAttention with Deformable Attention
        self.transformer_encoder = DeformableTransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nheads,
            dim_feedforward=hidden_size * 4,
            dropout=drop,
            activation="relu",
            normalize_before=True
        )
        
        self.pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            dropout=input_drop,
        )
        
        # Multi-Head Attention Pooling for better query representation
        self.attention_pool = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nheads, dropout=drop)
        
    def forward(self, feat, mask):
        """
        Args:
            feat: (N, L, D=hidden_size)
            mask: (N, L) with 1 indicating valid tokens

        Returns:
            (N, D)
        """
        feat = self.pos_embed(feat)  # (N, L, D)
        feat = self.transformer_encoder(feat, src_key_padding_mask=mask)
        
        # Multi-Head Attention Pooling to refine text embeddings
        feat = feat.permute(1, 0, 2)  # (L, N, D) for multi-head attention
        attn_output, _ = self.attention_pool(feat, feat, feat, key_padding_mask=mask)
        feat = attn_output.permute(1, 0, 2)  # (N, L, D)
        
        # Aggregate query representation
        pooled_feat = feat.mean(dim=1)  # (N, D)
        return pooled_feat


def mask_logits(target, mask, temperature=0.1):
    """Improved mask logits function for better numerical stability."""
    return target * mask + (1 - mask) * (-1e4 / temperature)


def build_text_encoder(args):
    return TextEncoder(
        hidden_size=args.hidden_dim,
        drop=args.dropout,
        input_drop=args.input_dropout,
        nheads=args.nheads,
        max_position_embeddings=args.max_q_l
    )
