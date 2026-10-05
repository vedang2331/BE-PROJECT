import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import os
import sys
import json
import cv2
import threading
import time
import math
import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageTk

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.basic_utils import *
from utils.video_utils import validate_video_file

import speech_recognition as sr

file_path1 = None
file_path2 = None
is_paused = False
playing_video1 = False
playing_video2 = False
video1_stream_id = 0
video2_stream_id = 0
active_player = None


def stop_all_videos():
    global playing_video1, playing_video2, is_paused, active_player
    playing_video1 = False
    playing_video2 = False
    is_paused = False
    active_player = None
    if 'pause_button' in globals():
        pause_button.config(text="Pause")


def clear_results_folder():
    stop_all_videos()
    results_dir = os.path.abspath("Results")
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
    else:
        os.makedirs(results_dir, exist_ok=True)


def browse_video(label):
    global file_path1
    selected = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if selected:
        file_path1 = os.path.abspath(selected)
        label.config(text=file_path1)
        return file_path1
    return None


def load_summarised_video(label=None):
    global file_path2
    results_dir = os.path.abspath("Results")
    
    if not os.path.exists(results_dir):
        messagebox.showwarning("No Results", "Results folder does not exist. Please run a query first.")
        return None

    # Automatically find highest confidence output video saved in Results folder
    target_file = None
    output_mp4 = os.path.join(results_dir, "output.mp4")
    output_0_mp4 = os.path.join(results_dir, "output_0.mp4")

    if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 0:
        target_file = output_mp4
    elif os.path.exists(output_0_mp4) and os.path.getsize(output_0_mp4) > 0:
        target_file = output_0_mp4
    else:
        mp4_files = [
            os.path.join(results_dir, f) for f in os.listdir(results_dir)
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')) and os.path.getsize(os.path.join(results_dir, f)) > 0
        ]
        if mp4_files:
            mp4_files.sort()
            target_file = mp4_files[0]

    if not target_file:
        messagebox.showwarning("No Video Found", "No summarized video found in Results folder. Please run a query first.")
        return None

    file_path2 = os.path.abspath(target_file)
    if label:
        label.config(text=file_path2)
    elif 'video2_label' in globals():
        video2_label.config(text=file_path2)

    status_label.config(text=f"Loaded summarized video: {os.path.basename(file_path2)}")
    
    # Automatically play the summarized video
    start_playing_video2()
    return file_path2


def play_video_stream(video_path, canvas, canvas_width, canvas_height, player_flag_name, stream_id):
    """Safely play video on Tkinter canvas using OpenCV."""
    global is_paused
    if not os.path.exists(video_path):
        messagebox.showerror("Playback Error", f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Playback Error", f"Cannot open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or math.isnan(fps):
        fps = 25.0
    delay_ms = max(10, int(1000 / fps))

    def update_frame():
        global is_paused
        current_sid = video1_stream_id if player_flag_name == "playing_video1" else video2_stream_id
        if current_sid != stream_id or globals().get(player_flag_name, False) is False:
            cap.release()
            return

        if is_paused:
            root.after(100, update_frame)
            return

        ret, frame = cap.read()
        if not ret:
            cap.release()
            globals()[player_flag_name] = False
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (canvas_width, canvas_height))

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk  # Keep reference

        root.after(delay_ms, update_frame)

    globals()[player_flag_name] = True
    update_frame()


def start_playing_video1():
    global playing_video1, playing_video2, video1_stream_id, is_paused, active_player
    video1_path = video1_label.cget("text")
    if not video1_path or video1_path == "No video selected" or not os.path.exists(video1_path):
        messagebox.showerror("Error", "Please select a valid input video file before playing.")
        return

    playing_video2 = False  # Stop video 2
    playing_video1 = False  # Stop previous video 1 stream
    is_paused = False
    active_player = "video1"
    pause_button.config(text="Pause")

    video1_stream_id += 1
    sid = video1_stream_id
    root.after(50, lambda: play_video_stream(video1_path, video1_canvas, 300, 300, "playing_video1", sid))


def start_playing_video2():
    global playing_video1, playing_video2, video2_stream_id, is_paused, active_player
    video2_path = video2_label.cget("text")
    if not video2_path or video2_path == "No video selected" or not os.path.exists(video2_path):
        messagebox.showerror("Error", "No summarized video available. Please generate or load a summarized video.")
        return

    playing_video1 = False  # Stop video 1
    playing_video2 = False  # Stop previous video 2 stream
    is_paused = False
    active_player = "video2"
    pause_button.config(text="Pause")

    video2_stream_id += 1
    sid = video2_stream_id
    root.after(50, lambda: play_video_stream(video2_path, video2_canvas, 300, 300, "playing_video2", sid))


def pause_video():
    global is_paused
    if not playing_video1 and not playing_video2:
        return
    is_paused = not is_paused
    pause_button.config(text="Resume" if is_paused else "Pause")


def load_query():
    global file_path2
    file_path2 = filedialog.askopenfilename(filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")])
    if file_path2:
        try:
            with open(file_path2, 'r') as file:
                display_text.delete(1.0, tk.END)
                for line in file:
                    record = json.loads(line)
                    display_text.insert(tk.END, record.get("query", "") + '\n')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {e}")
    return file_path2


def record_audio():
    recognizer = sr.Recognizer()
    status_label.config(text="Initializing microphone...")
    root.update_idletasks()

    try:
        with sr.Microphone() as source:
            status_label.config(text="Listening... Speak now")
            root.update_idletasks()
            audio = recognizer.listen(source, timeout=5)
            status_label.config(text="Recognizing speech...")
            root.update_idletasks()
            text = recognizer.recognize_google(audio)
            status_label.config(text="Audio converted to text")
            display_text.insert(tk.END, text + '\n')
            
            with open('converted_text.jsonl', 'w') as file:
                json_record = json.dumps({"query": text})
                file.write(json_record + '\n')
            messagebox.showinfo("Speech Recognition", f"Audio Query Captured:\n\"{text}\"")

    except (AttributeError, OSError) as e:
        status_label.config(text="Microphone / PyAudio unavailable")
        messagebox.showwarning(
            "Microphone Limitation",
            "Microphone hardware or PyAudio library is unavailable on this system.\n\n"
            "Please use Text Query mode to enter your query manually."
        )
    except sr.WaitTimeoutError:
        status_label.config(text="Listening timed out (no speech detected)")
        messagebox.showwarning("Timeout", "No speech detected within 5 seconds. Please try again.")
    except sr.UnknownValueError:
        status_label.config(text="Could not understand audio")
        messagebox.showwarning("Recognition Error", "Could not understand audio. Please speak clearly and try again.")
    except sr.RequestError as e:
        status_label.config(text="Speech service request failed")
        messagebox.showerror("API Error", f"Speech recognition service error: {e}")
    except Exception as e:
        status_label.config(text=f"Audio error: {e}")
        messagebox.showerror("Error", f"Audio recording error: {e}")


def handle_choice():
    choice = query_type.get()
    if choice == "audio":
        record_audio()
    elif choice == "text":
        enter_query()


def update_button_label(*args):
    choice = query_type.get()
    if choice == "audio":
        process_button.config(text="Record Audio")
    elif choice == "text":
        process_button.config(text="Enter Query")


def enter_query():
    text = text_entry.get("1.0", tk.END).strip()
    if text:
        data = {"query": text}
        with open("queries.jsonl", "w") as file:
            file.write(json.dumps(data) + "\n")
        messagebox.showinfo("Success", "Query saved to queries.jsonl")
        display_text.insert(tk.END, f"{text}\n")
        text_entry.delete("1.0", tk.END)
        status_label.config(text=f"Saved text query: \"{text}\"")
    else:
        messagebox.showerror("Error", "Please enter a query string before saving.")


def set_ui_processing_state(is_processing):
    state = tk.DISABLED if is_processing else tk.NORMAL
    query_au_btn.config(state=state)
    query_tq_btn.config(state=state)
    browse_video1_button.config(state=state)
    browse_video2_button.config(state=state)
    process_button.config(state=state)
    save_button.config(state=state)


def run_pipeline_worker(script_name, input_video, query_file):
    """Run inference script in a background thread to prevent UI freezing."""
    def target():
        try:
            cmd = [sys.executable, script_name, input_video]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            def on_finish():
                set_ui_processing_state(False)
                if result.returncode == 0:
                    out_path = os.path.abspath("Results/output.mp4")
                    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                        out_path = os.path.abspath("Results/output_0.mp4")

                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        video2_label.config(text=out_path)
                        status_label.config(text="Summarization complete! Playing summary...")
                        messagebox.showinfo("Success", f"Video summarization completed successfully!\nSaved to: {out_path}")
                        start_playing_video2()
                    else:
                        status_label.config(text="Summarization finished, but output file missing.")
                        messagebox.showerror("Pipeline Error", "Summarization script completed, but output video was not produced in Results folder.")
                else:
                    err_output = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                    status_label.config(text="Summarization failed.")
                    messagebox.showerror("Pipeline Error", f"Summarization failed with error:\n\n{err_output}")

            root.after(0, on_finish)

        except Exception as e:
            error_msg = str(e)
            def on_exception():
                set_ui_processing_state(False)
                status_label.config(text=f"Error executing pipeline: {error_msg}")
                messagebox.showerror("Execution Error", f"Failed to execute summarization pipeline: {error_msg}")

            root.after(0, on_exception)

    set_ui_processing_state(True)
    status_label.config(text="Generating video summary... Please wait (this may take a moment)")
    threading.Thread(target=target, daemon=True).start()


def query_tq_call():
    target_video = video1_label.cget("text")
    if not target_video or target_video == "No video selected":
        messagebox.showerror("Input Error", "Please load an input video before running Text Query summarization.")
        return

    is_valid, err_msg, _ = validate_video_file(target_video, max_duration=150.0)
    if not is_valid:
        messagebox.showerror("Validation Error", err_msg)
        return

    if not os.path.exists("queries.jsonl"):
        messagebox.showerror("Query Error", "Query file 'queries.jsonl' not found. Please enter and save a text query first.")
        return

    # Delete previous saved results in folder before running new query
    clear_results_folder()
    video2_label.config(text="No video selected")

    run_pipeline_worker('run_text_query_2.py', target_video, 'queries.jsonl')


def query_aq_call():
    target_video = video1_label.cget("text")
    if not target_video or target_video == "No video selected":
        messagebox.showerror("Input Error", "Please load an input video before running Audio Query summarization.")
        return

    is_valid, err_msg, _ = validate_video_file(target_video, max_duration=150.0)
    if not is_valid:
        messagebox.showerror("Validation Error", err_msg)
        return

    if not os.path.exists("converted_text.jsonl"):
        messagebox.showerror("Query Error", "Audio query file 'converted_text.jsonl' not found. Please record an audio query first.")
        return

    # Delete previous saved results in folder before running new query
    clear_results_folder()
    video2_label.config(text="No video selected")

    run_pipeline_worker('run_audio_query.py', target_video, 'converted_text.jsonl')


root = tk.Tk()
root.geometry("1200x650")
root.title("Content-Based Video Summarization")

label = tk.Label(root, text="Content based Video Summarization", width=350, anchor=tk.W, font=("Cambria", 16))
label.place(x=450, y=2)

video1_label = tk.Label(root, text="No video selected", width=50, anchor=tk.W)
video1_label.place(x=50, y=150)

video2_label = tk.Label(root, text="No video selected", width=50, anchor=tk.W)
video2_label.place(x=800, y=200)

browse_video1_button = tk.Button(root, text="Load Input Video", command=lambda: browse_video(video1_label))
browse_video1_button.place(x=50, y=100)

browse_video2_button = tk.Button(root, text="Load Summarized Video", command=lambda: load_summarised_video(video2_label))
browse_video2_button.place(x=800, y=150)

play_video1_button = tk.Button(root, text="Play Input Video", command=start_playing_video1)
play_video1_button.place(x=50, y=200)

play_video2_button = tk.Button(root, text="Play Summarized Video", command=start_playing_video2)
play_video2_button.place(x=800, y=250)

pause_button = tk.Button(root, text="Pause", command=pause_video)
pause_button.place(x=250, y=200)

video1_canvas = tk.Canvas(root, width=300, height=300, bg="black")
video1_canvas.place(x=50, y=300)

video2_canvas = tk.Canvas(root, width=300, height=300, bg="black")
video2_canvas.place(x=800, y=300)

load_iq_btn = tk.Button(root, text="Input Query", bg='turquoise')
load_iq_btn.place(x=450, y=50)

status_label = tk.Label(root, text="Select query type and press the button", wraplength=300)
status_label.place(x=450, y=100)

query_type = tk.StringVar(value="audio")
query_type.trace("w", update_button_label)

radio_audio = tk.Radiobutton(root, text="Audio Query", variable=query_type, value="audio")
radio_audio.place(x=450, y=150)

radio_text = tk.Radiobutton(root, text="Text Query", variable=query_type, value="text")
radio_text.place(x=450, y=200)

process_button = tk.Button(root, text="Record Audio", command=handle_choice)
process_button.place(x=450, y=250)

text_entry = tk.Text(root, wrap=tk.WORD, width=30, height=4)
text_entry.place(x=450, y=300)

save_button = tk.Button(root, text="Save Query", command=enter_query)
save_button.place(x=450, y=400)

display_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=30, height=4)
display_text.place(x=450, y=450)

query_au_btn = tk.Button(root, text="Video Summary using Text Query", bg='indian red', command=query_tq_call)
query_au_btn.place(x=800, y=50)

query_tq_btn = tk.Button(root, text="Video Summary using Audio Query", bg='indian red', command=query_aq_call)
query_tq_btn.place(x=800, y=100)

if __name__ == "__main__":
    root.mainloop()
