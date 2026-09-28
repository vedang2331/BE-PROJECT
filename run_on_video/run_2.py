import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array, load_jsonl


class MomentDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2  # seconds
        self.device = device
        print("Extracting features of input video..")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained QDA-DETR model..")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)  # Shape: (75, 512)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
    
        # Add temporal encoding features
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)
    
        # Retrieve the expected feature size from the model
        expected_vid_dim = self.model.input_vid_proj[0].net[1].in_features
        #print(f"✅ Model expects vid_dim: {expected_vid_dim}")
    
        if video_feats.shape[-1] != expected_vid_dim:
            pad_size = expected_vid_dim - video_feats.shape[-1]
            padding = torch.zeros((video_feats.shape[0], pad_size), device=self.device)
            video_feats = torch.cat([video_feats, padding], dim=-1)
    
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
    
        query_feats = self.feature_extractor.encode_text(query_list)
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None
        )
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
    
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )
    
        outputs = self.model(**model_inputs)
        # decode outputs
        # outputs = self.model(**model_inputs)
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

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            top_5_predictions = [[round(pred[0], 2), round(pred[1], 2), round(pred[2], 4)] for pred in cur_ranked_preds[:2]]

            cur_query_pred = dict(
                query=query_list[idx],  
                vid=video_path,
                pred_relevant_windows=top_5_predictions,  # Select Top-5 keyshots
                pred_saliency_scores=saliency_scores[idx]  
            )
            predictions.append(cur_query_pred)

        return predictions

"""
def calculate_f1_score(predicted_windows, gt_windows, iou_threshold=0.4):
    
    def iou(pred, gt):
        inter_start = max(pred[0], gt[0])
        inter_end = min(pred[1], gt[1])
        inter = max(0, inter_end - inter_start)
        union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
        return inter / union if union > 0 else 0

    y_true = []
    y_pred = []

    for gt in gt_windows:
        max_iou = max([iou(pred, gt) for pred in predicted_windows], default=0)
        y_true.append(1)
        y_pred.append(1 if max_iou >= iou_threshold else 0)

    for pred in predicted_windows:
        max_iou = max([iou(pred, gt) for gt in gt_windows], default=0)
        if max_iou < iou_threshold:
            y_true.append(0)
            y_pred.append(1)

    f1 = f1_score(y_true, y_pred)
    return f1
"""

def get_dynamic_iou_threshold(predicted_windows, gt_windows):
    iou_scores = []
    
    def iou(pred, gt):
        inter_start = max(pred[0], gt[0])
        inter_end = min(pred[1], gt[1])
        inter = max(0, inter_end - inter_start)
        union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
        return inter / union if union > 0 else 0

    for gt in gt_windows:
        for pred in predicted_windows:
            iou_scores.append(iou(pred, gt))

    if len(iou_scores) == 0:
        return 0.35  # Default if no overlaps

    dynamic_iou_threshold = max(iou_scores)  # Use median with min cap
    #print(f" Dynamic IoU Threshold: {dynamic_iou_threshold:.4f}")
    
    return dynamic_iou_threshold

def calculate_f1_score(predicted_windows, gt_windows):
    """
    Calculate F1-score based on IoU overlap with dynamic threshold.
    """
    def iou(pred, gt):
        inter_start = max(pred[0], gt[0])
        inter_end = min(pred[1], gt[1])
        inter = max(0, inter_end - inter_start)
        union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
        return inter / union if union > 0 else 0

    # Compute dynamic IoU threshold
    dynamic_iou_threshold = get_dynamic_iou_threshold(predicted_windows, gt_windows)

    y_true = []
    y_pred = []

    #print("\n IoU Debugging:")
    for gt in gt_windows:
        for pred in predicted_windows:
            iou_score = iou(pred, gt)
            #print(f"IoU between GT {gt} and Predicted {pred}: {iou_score:.4f}")

    for gt in gt_windows:
        max_iou = max([iou(pred, gt) for pred in predicted_windows], default=0)
        y_true.append(1)
        y_pred.append(1 if max_iou >= dynamic_iou_threshold else 0)

    for pred in predicted_windows:
        max_iou = max([iou(pred, gt) for gt in gt_windows], default=0)
        if max_iou < dynamic_iou_threshold:
            y_true.append(0)
            y_pred.append(1)

    f1 = f1_score(y_true, y_pred)
    f1 = f1*100
    #print(f"Final F1-score: {f1:.4f}\n")

    return f1


def run_example():
    # Load example data
    video_path = "E:/Ph_D_2020/Video_Summarization/Implementations/QDA-DETR/run_on_video/example/video_3.mp4"
    query_path = "E:/Ph_D_2020/Video_Summarization/Implementations/QDA-DETR/run_on_video/example/queries_v_3.jsonl"
    queries = load_jsonl(query_path)
    query_text_list = [e["query"] for e in queries]
    gt_windows_list = [e["relevant_windows"] for e in queries]

    ckpt_path = "E:/Ph_D_2020/Video_Summarization/Implementations/QDA-DETR/results_1/model_best.ckpt"

    #print("Build models...")
    moment_detr_predictor = MomentDETRPredictor(ckpt_path, "ViT-B/32", "cuda")

    print("\nPredictions:")
    predictions = moment_detr_predictor.localize_moment(video_path, query_list=query_text_list)

    for idx, query_data in enumerate(queries):
        print(f"\nQuery: {query_data['query']}")
        print(f"\nGround Truth Keyshots: {query_data['relevant_windows']}")
        print(f"Predicted Keyshots [start, stop, score]: \n{predictions[idx]['pred_relevant_windows']}")
        f1 = calculate_f1_score(predictions[idx]['pred_relevant_windows'], gt_windows_list[idx])
        print(f"\nF1-score: {f1:.2f}")


if __name__ == "__main__":
    run_example()
