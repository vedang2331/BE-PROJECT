import torch
import os
import sys 
from moment_detr.transformer import build_transformer
from moment_detr.position_encoding import build_position_encoding
from moment_detr.model import MomentDETR 

def build_inference_model(ckpt_path, **kwargs):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
    args = ckpt["opt"]
    if len(kwargs) > 0:
        args.update(kwargs)
    
    # Retrieve exact input_vid_proj weight size from checkpoint if available
    if "model" in ckpt and "input_vid_proj.0.net.1.weight" in ckpt["model"]:
        expected_vid_dim = ckpt["model"]["input_vid_proj.0.net.1.weight"].shape[1]
        args.v_feat_dim = expected_vid_dim
    elif getattr(args, "ctx_mode", None) == "video_tef":
        args.v_feat_dim = args.v_feat_dim + 2
        
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=getattr(args, "contrastive_align_loss", False),
        contrastive_hdim=getattr(args, "contrastive_hdim", 64),
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
    )
    
    model.load_state_dict(ckpt["model"], strict=True)
    return model