import tkinter as tk
from tkinter import messagebox
import os
import threading
from detect_gesture import GestureDetectionApp

# Create the main window (Main Menu)
root = tk.Tk()
root.title("Sign Language Detection - Main Menu")
root.geometry("500x500")
root.configure(bg="#282c34")

gesture_app = GestureDetectionApp()

# Function to start detection in a separate thread
def start_detection():
    threading.Thread(target=gesture_app.start_detection, daemon=True).start()
    messagebox.showinfo("Gesture Detection", "Gesture detection started.")

def pause_detection():
    gesture_app.pause_detection()
    messagebox.showinfo("Gesture Detection", "Gesture detection paused.")

def stop_detection():
    gesture_app.stop_detection()
    messagebox.showinfo("Gesture Detection", "Gesture detection stopped.")

def open_gesture_control_window():
    gesture_window = tk.Toplevel(root)
    gesture_window.title("Gesture Detection Control")
    gesture_window.geometry("400x400")
    gesture_window.configure(bg="#f4f4f9")

    title_label = tk.Label(gesture_window, text="Gesture Detection", font=("Helvetica", 18, 'bold'), fg="#333", bg="#f4f4f9")
    title_label.pack(pady=20)

    btn_start = tk.Button(gesture_window, text="Start Detection", font=("Helvetica", 14), bg="#007BFF", fg="white", width=15, relief="solid", command=start_detection)
    btn_start.pack(pady=10)

    btn_pause = tk.Button(gesture_window, text="Pause Detection", font=("Helvetica", 14), bg="#FFC107", fg="black", width=15, relief="solid", command=pause_detection)
    btn_pause.pack(pady=10)

    btn_stop = tk.Button(gesture_window, text="Stop Detection", font=("Helvetica", 14), bg="#DC3545", fg="white", width=15, relief="solid", command=stop_detection)
    btn_stop.pack(pady=10)

def collect_data():
    def run_collect_data():
        os.system('python collect_data.py')

    threading.Thread(target=run_collect_data, daemon=True).start()
    messagebox.showinfo("Collect Data", "Collecting data...")

def train_model():
    def run_train_model():
        os.system('python train_model.py')

    threading.Thread(target=run_train_model, daemon=True).start()
    messagebox.showinfo("Train Model", "Training model...")

title_label = tk.Label(root, text="Sign Detection", font=("Helvetica", 28, 'bold'), fg="#61dafb", bg="#282c34")
title_label.pack(pady=30)

subtitle_label = tk.Label(root, text="Choose an action below", font=("Helvetica", 14), fg="#dcdcdc", bg="#282c34")
subtitle_label.pack(pady=5)

def create_button(text, command, color_bg, color_fg):
    return tk.Button(root, text=text, font=("Helvetica", 16), bg=color_bg, fg=color_fg, width=25, height=2, relief="solid", command=command)

btn_collect_data = create_button("Collect Data", collect_data, "#007BFF", "white")
btn_collect_data.pack(pady=10)

btn_train_model = create_button("Train Model", train_model, "#28A745", "white")
btn_train_model.pack(pady=10)

btn_detect_gesture = create_button("Detect Gesture", open_gesture_control_window, "#17A2B8", "white")
btn_detect_gesture.pack(pady=10)

btn_exit = create_button("Exit", root.quit, "#DC3545", "white")
btn_exit.pack(pady=10)

root.mainloop()
