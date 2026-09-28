import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
# from tkVideoPlayer import TkinterVideo  # Unused import; video playing uses OpenCV and Canvas
from tkinter import *
import os
import sys
from pathlib import Path
from utils.basic_utils import *
from tkinter import ttk
import subprocess
import speech_recognition as sr
from tkinter import scrolledtext, filedialog, messagebox
import json
import cv2
from PIL import Image, ImageTk
import threading
import time

is_paused = False

def browse_video(label):
    global file_path1
    file_path1 = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")])
    if file_path1:
        label.config(text=file_path1)
        return file_path1
    return None

def play_video(file_path1, canvas, canvas_width, canvas_height):
    global is_paused
    cap = cv2.VideoCapture(file_path1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 / fps

    while cap.isOpened():
        start_time = time.time()
        if is_paused:
            time.sleep(0.1)
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit canvas
        frame = cv2.resize(frame, (canvas_width, canvas_height))
        
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk  # Keep a reference to prevent garbage collection
        canvas.update()

        elapsed_time = time.time() - start_time
        time_to_wait = delay - elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

    cap.release()

def start_playing_video1():
    video1_path = video1_label.cget("text")
    if not video1_path or video1_path == "No video selected":
        messagebox.showerror("Error", "Please select the first video file before playing.")
        return

    threading.Thread(target=play_video, args=(video1_path, video1_canvas, 300, 240)).start()

def start_playing_video2():
    video2_path = video2_label.cget("text")
    if not video2_path or video2_path == "No video selected":
        messagebox.showerror("Error", "Please select the second video file before playing.")
        return

    threading.Thread(target=play_video, args=(video2_path, video2_canvas, 300, 240)).start()

def pause_video():
    global is_paused
    is_paused = not is_paused
    pause_button.config(text="Resume" if is_paused else "Pause")

def load_query():
    global file_path2
    file_path2 = filedialog.askopenfilename(filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")])
    print(file_path2)
    if file_path2:
        try:
            with open(file_path2, 'r') as file:
                display_text.delete(1.0, tk.END)  # Clear the current text
                for line in file:
                    record = json.loads(line)
                    display_text.insert(tk.END, record["query"] + '\n')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {e}")
    
    return file_path2

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        status_label.config(text="Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            status_label.config(text="Audio converted to text")
            display_text.insert(tk.END, text + '\n')
            
            # Save the text to a JSON Lines file, overwriting previous content
            with open('converted_text.jsonl', 'w') as file:
                json_record = json.dumps({"query": text})
                file.write(json_record + '\n')
                
        except sr.WaitTimeoutError:
            status_label.config(text="Listening timed out")
        except sr.UnknownValueError:
            status_label.config(text="Could not understand audio")
        except sr.RequestError:
            status_label.config(text="Could not request results; check your network connection")

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
    else:
        messagebox.showerror("Error", "Please enter a query")

def query_tq_call():  
    print(file_path1)
    subprocess.run([sys.executable, 'run_text_query_2.py', file_path1])

def query_aq_call():  
    print(file_path1)
    subprocess.run([sys.executable, 'run_audio_query.py', file_path1])

# GUI Styling
root = tk.Tk()
root.geometry("1200x660+100+50")
root.title("Video Summarization")
root.configure(bg="#2E3440")  # Set background color

# Colors
primary_bg = "#2E3440"
secondary_bg = "#4C566A"
primary_fg = "#FFFFFF"  # Set all text to white
button_color = "darkolivegreen"
highlight_color = "steelblue"

# Title Label
label = tk.Label(root, text="Content based Video Summarization", width=350, anchor=tk.W, font=("Cambria", 21, "bold"), bg=primary_bg, fg=primary_fg)  # Increased font size
label.place(x=350, y=20)

# Video Selection Labels
video1_label = tk.Label(root, text="Select Input Video for Summarization", width=50, anchor=tk.W, bg=primary_bg, fg=primary_fg, font=("Cambria", 14, "bold"))  # Increased font size
video1_label.place(x=50, y=100)

video1_label = tk.Label(root, text="No video selected", width=50, anchor=tk.W, bg=primary_bg, fg=primary_fg, font=("Cambria", 12, "bold"), wraplength=320)  # Increased font size
video1_label.place(x=50, y=200)
video2_label = tk.Label(root, text="No video selected", width=50, anchor=tk.W, bg=primary_bg, fg=primary_fg, font=("Cambria", 12, "bold"), wraplength=320)  # Increased font size
video2_label.place(x=800, y=250)

# Buttons for Loading Videos
browse_video1_button = tk.Button(root, text="Load Input Video", command=lambda: browse_video(video1_label), bg=button_color, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
browse_video1_button.place(x=50, y=150)
browse_video2_button = tk.Button(root, text="Load Summarized Video", command=lambda: browse_video(video2_label), bg=button_color, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
browse_video2_button.place(x=800, y=200)

# Video Playback Buttons
play_video1_button = tk.Button(root, text="Play Input Video", command=start_playing_video1, bg=button_color, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
play_video1_button.place(x=50, y=290)
play_video2_button = tk.Button(root, text="Play Summarized Video", command=start_playing_video2, bg=button_color, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
play_video2_button.place(x=800, y=300)
pause_button = tk.Button(root, text="Pause", command=pause_video, bg=button_color, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
pause_button.place(x=250, y=290)

# Canvases for Videos
video1_canvas = tk.Canvas(root, width=300, height=240, bg=secondary_bg)
video1_canvas.place(x=50, y=350)
video2_canvas = tk.Canvas(root, width=300, height=240, bg=secondary_bg)
video2_canvas.place(x=800, y=350)

# Query Type and Text/Audio Input
status_label = tk.Label(root, text="Select type of the query", bg=primary_bg, fg=primary_fg, font=("Cambria", 14, "bold"))  # Increased font size
status_label.place(x=450, y=100)
query_type = tk.StringVar(value="audio")
query_type.trace("w", update_button_label)
radio_audio = tk.Radiobutton(root, text="Audio Query", variable=query_type, value="audio", bg=primary_bg, fg=primary_fg, selectcolor=secondary_bg, font=("Cambria", 13, "bold"))  # Increased font size
radio_audio.place(x=450, y=150)
radio_text = tk.Radiobutton(root, text="Text Query", variable=query_type, value="text", bg=primary_bg, fg=primary_fg, selectcolor=secondary_bg, font=("Cambria", 13, "bold"))  # Increased font size
radio_text.place(x=450, y=200)
process_button = tk.Button(root, text="Record Audio", command=handle_choice, bg=button_color, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
process_button.place(x=450, y=250)

# Text Entry Box and Save Button
text_entry = tk.Text(root, wrap=tk.WORD, width=30, height=4, bg=secondary_bg, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
text_entry.place(x=450, y=300)
save_button = tk.Button(root, text="Save Query", command=enter_query, bg=button_color, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
save_button.place(x=450, y=400)

# Display Text Box
display_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=30, height=4, bg=secondary_bg, fg=primary_fg, font=("Cambria", 12, "bold"))  # Increased font size
display_text.place(x=450, y=450)

# Query Action Buttons
query_au_btn = tk.Button(root, text="Video Summary using Text Query", bg=highlight_color, fg=primary_fg, command=query_tq_call, font=("Cambria", 12, "bold"))  # Increased font size
query_au_btn.place(x=800, y=100)
query_tq_btn = tk.Button(root, text="Video Summary using Audio Query", bg=highlight_color, fg=primary_fg, command=query_aq_call, font=("Cambria", 12, "bold"))  # Increased font size
query_tq_btn.place(x=800, y=150)

root.mainloop()
