import torch
import os
import sys
from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array, load_jsonl
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from pathlib import Path

class MomentDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_len = 2  # seconds
        self.device = device
        print(f"Using device: {self.device}")
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained Moment-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)

        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        
        expected_vid_dim = self.model.input_vid_proj[0].net[1].in_features

        if video_feats.shape[-1] != expected_vid_dim:
            if video_feats.shape[-1] + 2 == expected_vid_dim:
                video_feats = torch.cat([video_feats, tef], dim=-1)
            else:
                pad_size = expected_vid_dim - video_feats.shape[-1]
                padding = torch.zeros((video_feats.shape[0], pad_size), device=self.device)
                video_feats = torch.cat([video_feats, padding], dim=-1)
    
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )

        outputs = self.model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1)
        scores = prob[..., 0]
        pred_spans = outputs["pred_spans"]
        _saliency_scores = outputs["saliency_scores"].half()
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            res = list(map(lambda x: x[2], cur_ranked_preds))
            print("Confidence scores:", res)
            a = len(res)
            cur_ranked_high = []
            for i in range(a):
                if (res[i] > 0.9000):
                    print(cur_ranked_preds[i])
                    cur_ranked_high.append(cur_ranked_preds[i])
                    
            if not cur_ranked_high and cur_ranked_preds:
                cur_ranked_high.append(cur_ranked_preds[0])

            cur_query_pred = dict(
                query=query_list[idx],
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx]
            )
            predictions.append(cur_query_pred)
        
        return predictions, cur_ranked_high


from utils.video_utils import validate_video_file


def run_example():
    if len(sys.argv) < 2:
        print("Usage: python run_text_query_2.py <video_path>")
        return

    video_path = sys.argv[1]
    query_path = "queries.jsonl"
    
    print(f"Video Path: {video_path}")
    print(f"Query Path: {query_path}")

    is_valid, err_msg, _ = validate_video_file(video_path, max_duration=150.0)
    if not is_valid:
        print(err_msg, file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(query_path):
        print(f"Error: Query file '{query_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        queries = load_jsonl(query_path)
        query_text_list = [e["query"] for e in queries if isinstance(e, dict) and "query" in e and e["query"].strip()]
        if not query_text_list:
            print(f"Error: Query file '{query_path}' contains no valid text queries.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error reading query file '{query_path}': {e}", file=sys.stderr)
        sys.exit(1)
    
    ckpt_path = "results_1/model_best.ckpt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file '{ckpt_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moment_detr_predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device=device
    )
    print("Run prediction...")
    predictions, cur_ranked_high = moment_detr_predictor.localize_moment(
        video_path=video_path, query_list=query_text_list)

    for idx, query_data in enumerate(queries):
        print("-" * 30 + f" idx{idx}")
        print(f">> query: {query_data['query']}")
        print(f">> video_path: {video_path}")
        print(f">> Predicted moments ([start_in_seconds, end_in_seconds, score]): "
              f"{predictions[idx]['pred_relevant_windows']}")
        print(f">> Predicted saliency scores (for all 2-sec clip): "
              f"{predictions[idx]['pred_saliency_scores']}")
              
    import shutil
    # Sort high-confidence predictions by start timestamp while preserving intact (start, end) pairs
    cur_ranked_high = sorted(cur_ranked_high, key=lambda x: x[0])
    a = len(cur_ranked_high)

    start_pt = [x[0] for x in cur_ranked_high]
    end_pt = [x[1] for x in cur_ranked_high]

    print("Start points:", start_pt)
    print("End points:", end_pt)
    input_file = video_path

    os.makedirs("Results", exist_ok=True)

    for i in range(a):
        n = i
        s_s = max(0.0, cur_ranked_high[i][0])
        t_o = max(s_s + 0.1, cur_ranked_high[i][1])
        cmd = f'ffmpeg -y -i "{input_file}" -ss {s_s} -to {t_o} -c:v libx264 -crf 30 "Results/output_{n}.mp4"'
        os.system(cmd)

    with open('Results/video_clips.txt', 'w') as file_handler:
        for i in range(a):
            file_handler.write("file 'output_{}.mp4'\n".format(i))

    if a == 1:
        shutil.copyfile("Results/output_0.mp4", "Results/output.mp4")
    elif a > 1:
        cmd = f'ffmpeg -y -f concat -safe 0 -i Results/video_clips.txt -c:v libx264 -crf 30 Results/output.mp4'
        os.system(cmd)

if __name__ == "__main__":
    run_example()
