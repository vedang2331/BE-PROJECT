import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array, load_jsonl


def print_diagnostics():
    print("="*60)
    print("Moment-DETR System & Environment Diagnostics")
    print("="*60)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available : {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device     : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Runtime    : {torch.version.cuda}")
    else:
        print("WARNING: CUDA is not available. Running on CPU.")
    print("="*60)


class MomentDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.clip_len = 2  # seconds
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained Moment-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        # add tef
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, "The positional embedding of this pretrained MomentDETR only support video up " \
                               "to 150 secs (i.e., 75 2-sec clips) in length"
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

        # decode outputs
        outputs = self.model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries) foreground label is 0
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
                pred_saliency_scores=saliency_scores[idx]  # List(float), len==n_frames
            )
            predictions.append(cur_query_pred)

        return predictions


def run_example():
    print_diagnostics()
    
    # Define relative paths based on project root
    video_path = str(PROJECT_ROOT / "run_on_video" / "example" / "video_2.mp4")
    query_path = str(PROJECT_ROOT / "run_on_video" / "example" / "queries_v_2.jsonl")
    ckpt_path = str(PROJECT_ROOT / "run_on_video" / "moment_detr_ckpt" / "model_best.ckpt")

    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    assert os.path.exists(query_path), f"Query file not found: {query_path}"
    assert os.path.exists(ckpt_path), f"Checkpoint file not found: {ckpt_path}"

    queries = load_jsonl(query_path)
    query_text_list = [e["query"] for e in queries]

    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moment_detr_predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device=device
    )
    print("Run prediction...")
    predictions = moment_detr_predictor.localize_moment(
        video_path=video_path, query_list=query_text_list)

    # print data
    for idx, query_data in enumerate(queries):
        print("-" * 30 + f" idx{idx}")
        print(f">> query: {query_data['query']}")
        print(f">> video_path: {video_path}")
        print(f">> GT Keyshots: {query_data['relevant_windows']}")
        print(f">> Predicted keyshots ([start_in_seconds, end_in_seconds, score]): "
              f"{predictions[idx]['pred_relevant_windows']}")
        print(f">> GT importance scores: {query_data['saliency_scores']}")
        print(f">> Predicted importance scores (for all 2-sec clip): "
              f"{predictions[idx]['pred_saliency_scores']}")


if __name__ == "__main__":
    run_example()
