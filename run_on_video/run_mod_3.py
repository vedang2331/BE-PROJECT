import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from utils.basic_utils import l2_normalize_np_array, load_jsonl

class QueryFocusedSummarizer:
    def __init__(self, ckpt_path, gt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2  # seconds
        self.device = device
        self.gt_data = self.load_ground_truth(gt_path)
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained Moment-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)
    
    def load_ground_truth(self, gt_path):
        gt_data = load_jsonl(gt_path)
        return {entry["query"]: entry for entry in gt_data}
    
    @torch.no_grad()
    def summarize_video(self, video_path, query_list, top_k=10):
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        
        # Add Temporal Encoding Features (TEF)
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)
        video_feats = torch.cat([video_feats, tef], dim=1)
        
        assert n_frames <= 75, "Pretrained MomentDETR supports up to 150-sec videos (75 clips)."
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        
        query_feats = self.feature_extractor.encode_text(query_list)
        query_feats, query_mask = pad_sequences_1d(query_feats, dtype=torch.float32, device=self.device)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )

        # Model inference
        outputs = self.model(**model_inputs)
        saliency_scores = outputs["saliency_scores"].squeeze(0).cpu().numpy()
        
        # Rank frames based on saliency scores
        top_frame_indices = np.argsort(saliency_scores)[-top_k:]
        top_frame_indices.sort()
        
        # Print selected clip information
        print("Selected Clips for Summarization:")
        for idx in top_frame_indices:
            start_time = idx * self.clip_len
            end_time = start_time + self.clip_len
            print(f"Frame Index: {idx}, Start: {start_time:.2f}s, End: {end_time:.2f}s, Importance Score: {saliency_scores[idx]:.4f}")
        
        # Compute F1-score
        f1_score = self.compute_f1(query_list, top_frame_indices)
        print(f"F1-score: {f1_score:.4f}")
        
        return top_frame_indices, saliency_scores, f1_score
    
    def compute_f1(self, query_list, predicted_clips):
        all_f1_scores = []
        for query in query_list:
            if query in self.gt_data:
                gt_clips = set(self.gt_data[query]["relevant_clip_ids"])
                pred_clips = set(predicted_clips)
                
                tp = len(gt_clips & pred_clips)
                fp = len(pred_clips - gt_clips)
                fn = len(gt_clips - pred_clips)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                all_f1_scores.append(f1)
        
        return np.mean(all_f1_scores) if all_f1_scores else 0
    
    def generate_summary_video(self, video_path, top_frame_indices, saliency_scores, output_video):
        temp_clips = []
        for idx, frame_idx in enumerate(top_frame_indices):
            start_time = frame_idx * self.clip_len
            end_time = start_time + self.clip_len
            temp_clip = f"Results/clip_{idx}.mp4"
            cmd = f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c:v libx264 -crf 30 {temp_clip}"
            os.system(cmd)
            temp_clips.append(temp_clip)
        
        # Merge clips into a final summarized video
        with open("Results/video_clips.txt", "w") as f:
            for clip in temp_clips:
                f.write(f"file '{clip}'\n")
        
        merge_cmd = f"ffmpeg -f concat -safe 0 -i Results/video_clips.txt -c copy {output_video}"
        os.system(merge_cmd)

if __name__ == "__main__":
    video_path = "E:/Ph_D_2020/Video_Summarization/Implementations/Exp3/run_on_video/example/video_5.mp4"
    query_path = "E:/Ph_D_2020/Video_Summarization/Implementations/Exp3/run_on_video/example/queries_v_5.jsonl"
    gt_path = "E:/Ph_D_2020/Video_Summarization/Implementations/Exp3/run_on_video/example/queries_v_5.jsonl"
    queries = load_jsonl(query_path)
    query_text_list = [e["query"] for e in queries]
    ckpt_path = "E:/Ph_D_2020/Video_Summarization/Implementations/Exp3/run_on_video/moment_detr_ckpt/model_best.ckpt"
    
    summarizer = QueryFocusedSummarizer(ckpt_path, gt_path)
    top_frames, saliency_scores, f1_score = summarizer.summarize_video(video_path, query_text_list, top_k=10)
    summarizer.generate_summary_video(video_path, top_frames, saliency_scores, "E:/Ph_D_2020/Video_Summarization/Implementations/Exp3/Results/summary_output.mp4")