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

def play_video(file_path1, canvas):
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
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk  # Keep a reference to the image to prevent garbage collection
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

    threading.Thread(target=play_video, args=(video1_path, video1_canvas)).start()

def start_playing_video2():
    video2_path = video2_label.cget("text")
    if not video2_path or video2_path == "No video selected":
        messagebox.showerror("Error", "Please select the second video file before playing.")
        return

    threading.Thread(target=play_video, args=(video2_path, video2_canvas)).start()

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
    subprocess.run(['python', 'run_text_query_2.py', file_path1])

def query_aq_call():  
    print(file_path1)
    subprocess.run(['python', 'run_audio_query.py', file_path1])

root = tk.Tk()
root.geometry("1200x650")
root.title("Video Summarization")

label = tk.Label(root, text="Content based Video Summarization", width=350, anchor=tk.W, font=("Cambria", 16))
label.place(x = 450, y = 2)

video1_label = tk.Label(root, text="Select Input Video for Summarization", width=50, anchor=tk.W)
video1_label.place(x = 50, y = 50)

video1_label = tk.Label(root, text="No video selected", width=50, anchor=tk.W)
video1_label.place(x = 50, y = 150)
video2_label = tk.Label(root, text="No video selected", width=50, anchor=tk.W)
video2_label.place(x = 800, y = 200)

browse_video1_button = tk.Button(root, text="Load Input Video", command=lambda: browse_video(video1_label))
browse_video1_button.place(x = 50, y = 100)
browse_video2_button = tk.Button(root, text="Load Summarized Video", command=lambda: browse_video(video2_label))
browse_video2_button.place(x = 800, y = 150)

play_video1_button = tk.Button(root, text="Play Input Video", command=start_playing_video1)
play_video1_button.place(x = 50, y = 200)
play_video2_button = tk.Button(root, text="Play Summarized Video", command=start_playing_video2)
play_video2_button.place(x = 800, y = 250)

pause_button = tk.Button(root, text="Pause", command=pause_video)
pause_button.place(x = 250, y = 200)

video1_canvas = tk.Canvas(root, width=300, height=300, bg="black")
video1_canvas.place(x = 50, y = 300)
video2_canvas = tk.Canvas(root, width=300, height=300, bg="black")
video2_canvas.place(x = 800, y = 300)

load_iq_btn = tk.Button(root, text="Input Query", bg='turquoise')
load_iq_btn.place(x = 450, y = 50)
Label(root).pack(padx=5,  pady=10)

status_label = tk.Label(root, text="Select query type and press the button")
status_label.place(x = 450, y = 100)
Label(root).pack(padx=5,  pady=10)

query_type = tk.StringVar(value="audio")
query_type.trace("w", update_button_label)  # Call update_button_label when query_type changes

radio_audio = tk.Radiobutton(root, text="Audio Query", variable=query_type, value="audio")
radio_audio.place(x = 450, y = 150)
Label(root).pack(padx=5,  pady=10)

radio_text = tk.Radiobutton(root, text="Text Query", variable=query_type, value="text")
radio_text.place(x = 450, y = 200)
Label(root).pack(padx=5,  pady=10)

process_button = tk.Button(root, text="Record Audio", command=handle_choice)
process_button.place(x = 450, y = 250)
Label(root).pack(padx=5,  pady=10)

text_entry = tk.Text(root, wrap=tk.WORD, width=30, height=4)
text_entry.place(x=450, y=300)

# Create a button to save the text input to a JSONL file
save_button = tk.Button(root, text="Save Query", command=enter_query)
save_button.place(x=450, y=400)

display_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=30, height=4)
display_text.place(x = 450, y = 450)
Label(root).pack(padx=5,  pady=10)

query_au_btn = tk.Button(root, text="Video Summary using Text Query", bg='indian red', command= query_tq_call)
query_au_btn.place(x = 800, y = 50)
Label(root).pack(padx=5,  pady=10)

query_tq_btn = tk.Button(root, text="Video Summary using Audio Query", bg='indian red', command= query_aq_call)
query_tq_btn.place(x = 800, y = 100)
Label(root).pack(padx=5,  pady=10)

root.mainloop()
