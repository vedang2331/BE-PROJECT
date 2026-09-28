# Run Content-Based Video Summarization Project

This project is a **Content-Based Video Summarization** application powered by **Moment-DETR / QDA-DETR** and **OpenAI CLIP**. It provides a Tkinter Graphical User Interface (GUI) that allows users to input a video and search for specific moments via text or spoken audio queries, producing a summarized output video.

---

## Technical Audit & Current Environment Status

- **Project Type**: Machine Learning / Computer Vision (PyTorch, CLIP, OpenCV, SpeechRecognition, Tkinter GUI).
- **Python Version**: `3.13.2` installed locally.
- **Model Checkpoints**: Available in `run_on_video/moment_detr_ckpt/model_best.ckpt`.
- **Sample Data**: Videos (`video_1.mp4`, etc.) and sample query files (`queries_v_1.jsonl`, etc.) available in `run_on_video/example/`.
- **Missing Dependencies**:
  - `torch` & `torchvision` (PyTorch deep learning framework)
  - `ffmpeg-python` & `imageio-ffmpeg` (for video feature extraction probing)
  - `ftfy` & `regex` (for CLIP tokenizer)

---

## Open Questions / Notes

> [!NOTE]
> - Running PyTorch inference on CPU is supported by default. If an NVIDIA GPU with CUDA is available, execution will be significantly faster.
> - For Audio Query recording via microphone, a working microphone input is required. Alternatively, manual text queries can be entered directly into the GUI.

---

## Proposed Setup & Execution Plan

### Step 1: Install Required Python Packages

Run the following command in PowerShell to install PyTorch and video processing dependencies:

```powershell
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install ffmpeg-python imageio-ffmpeg ftfy regex speechrecognition opencv-python pillow numpy
```

*(If you have an NVIDIA GPU, replace the PyTorch index URL with CUDA support).*

### Step 2: Configure FFmpeg Executable Path

`ffmpeg` is required by `ffmpeg-python` to sample video frames. We will ensure the `imageio-ffmpeg` binary is accessible on the system path or referenced by `ffmpeg-python`.

### Step 3: Run the Application

#### Option A: Launch the Graphical User Interface (GUI) (Recommended)

Run the main GUI application:

```powershell
python gui_audio_text_query_mod_2.py
```

**Using the GUI Interface:**
1. Click **"Load Input Video"** and select a video file (e.g., `run_on_video/example/video_1.mp4`).
2. Choose Query Type:
   - **Text Query**: Type your prompt into the text box and click **"Save Query"**.
   - **Audio Query**: Click **"Record Audio"** and speak into your microphone.
3. Click **"Video Summary using Text Query"** (or **"Audio Query"**) to generate the summarized video.
4. Use the playback buttons to preview the original and summarized videos side-by-side.

#### Option B: Run Command-Line Test Script

To run a quick prediction test directly on an example video:

```powershell
python run.py
```

---

## Verification Plan

### Automated / Diagnostic Checks
1. Verify Python imports (`torch`, `cv2`, `ffmpeg`, `clip`) run cleanly without errors.
2. Verify model checkpoint loading from `run_on_video/moment_detr_ckpt/model_best.ckpt`.

### Manual Verification
1. Launch `python gui_audio_text_query_mod_2.py` and verify the window renders correctly.
2. Load sample `run_on_video/example/video_1.mp4` and run a sample query (e.g., `"a person talking"`).
