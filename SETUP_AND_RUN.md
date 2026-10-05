# Setup and Run Guide — Content-Based Video Summarization

This guide provides instructions to set up the environment, install dependencies, and run the Content-Based Video Summarization project (Moment-DETR + CLIP).

---

## 1. Environment & Prerequisites

### Required Environment
- **Operating System:** Windows 10 / 11 (or Linux / macOS)
- **Python Version:** Python 3.8.20 (or 3.7.16)
- **PyTorch & CUDA:** PyTorch 1.13.1 + cu117 (CUDA 11.7)
- **GPU:** NVIDIA GPU with CUDA support (e.g., RTX 3050 Laptop GPU)

### Creating the Conda Environment
```bash
conda create -n be_project python=3.8 -y
conda activate be_project
```

### Installing PyTorch with CUDA 11.7
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Installing Project Dependencies
```bash
pip install -r requirements.txt
```

### External Requirement: FFmpeg & FFprobe
FFmpeg and FFprobe must be available in system PATH or in your active Conda environment:
```bash
conda install -c conda-forge ffmpeg -y
```

---

## 2. Project Execution Modes

### Mode A: Text Query Pipeline (CLI)
To run temporal moment retrieval on a video with text queries from `queries.jsonl`:
```bash
python run_text_query_2.py run_on_video/example/video_1.mp4
```
**Output:**
- Moment predictions printed to console with confidence scores.
- Generated highlight clips saved to `Results/output_0.mp4`, `Results/output_1.mp4`, etc.
- Final concatenated summary video saved to `Results/output.mp4`.

---

### Mode B: Custom Video Execution (CLI)
To retrieve moments for custom text queries on any input video:
```bash
python run_on_video/run_custom.py --video run_on_video/example/video_2.mp4 --query "Charging of Electrical Vehicles" "car driving"
```
Or using the portable batch launcher:
```cmd
run_custom.bat --video run_on_video/example/video_2.mp4 --query "Charging of Electrical Vehicles"
```

---

### Mode C: Audio Query Pipeline (CLI)
To summarize video using an audio query recorded via microphone (saved to `converted_text.jsonl`):
```bash
python run_audio_query.py run_on_video/example/video_1.mp4
```

---

### Mode D: Graphical User Interface (GUI)
To launch the interactive Tkinter GUI application:
```bash
python gui_audio_text_query_mod_2.py
```
Or using the portable batch launcher:
```cmd
gui.bat
```

#### GUI Instructions:
1. Click **"Load Input Video"** to select your target `.mp4` file.
2. Select **"Text Query"** radio button, enter a description in the text box, and click **"Save Query"**.
3. Click **"Video Summary using Text Query"** to start summarization.
4. The status label will update to *"Generating video summary... Please wait"*.
5. Once completed, `Results/output.mp4` will automatically connect to the output player canvas and begin playing!
6. Click **"Pause"** / **"Resume"** or **"Play Summarized Video"** to control playback.

---

## 3. Supported Model Specifications & Limits

| Specification | Value | Note |
| :--- | :--- | :--- |
| **Clip Length** | 2.0 seconds | Frame sampling rate = 0.5 Hz |
| **Max Video Clips** | 75 clips | $75 \times 2\text{s} = 150.0\text{ seconds}$ |
| **Max Video Duration** | 150.0 seconds | Videos $> 150\text{s}$ are cleanly rejected before inference |
| **Confidence Threshold** | 0.90 | Moments with score $> 0.90$ are selected for summary cutting |
| **Default Model Checkpoint** | `run_on_video/moment_detr_ckpt/model_best.ckpt` | Pretrained Moment-DETR model |

---

## 4. Troubleshooting & Expected Error Behavior

- **"Error: Video duration (180s) exceeds the maximum supported limit of 150 seconds..."**
  - *Reason:* Moment-DETR positional embeddings support up to 75 two-second clips (150s).
  - *Fix:* Provide an input video shorter than 150 seconds.
- **"Error: The supplied file is not a valid or readable video file."**
  - *Reason:* The file is missing, corrupt, empty, or lacks a readable video stream.
  - *Fix:* Ensure a valid `.mp4`, `.avi`, or `.mov` file is selected.
- **"Microphone hardware or PyAudio library is unavailable on this system."**
  - *Reason:* Microphone hardware is missing or PyAudio is not installed.
  - *Fix:* Use **Text Query** mode to enter text queries directly.
