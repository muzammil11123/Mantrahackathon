#!/usr/bin/env python
# video_processing_gui.py

import os
import re
import math
import cv2
import logging
import platform
import psutil
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import torch
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract
from TTS.api import TTS
from panns_inference import AudioTagging, labels as pann_labels
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import subprocess
import shutil
import warnings
import time
import base64
import ollama  # Requires the ollama Python package
import requests
import simpleaudio as sa

# --- Dynamic Path Setup (for Dockerization / cross-platform) ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # Set TESSDATA_PREFIX to the parent directory containing tessdata (not used anymore)
    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR"
    PANN_MODEL_PATH = r"C:\Users\arash\panns_data\cnn14.pth"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    PANN_MODEL_PATH = "models/cnn14.pth"

# --- Suppress extraneous warnings ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Setup Logging ---
LOG_FILE = "video_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- YOLO Class Mapping ---
CLASS_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "brocolli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    78: "hair drier", 79: "toothbrush"
}

# --- Global Constants & Variables ---
LLAVA_INTERVAL = 5  # LLava is removed for speed

# --- Helper: Convert Seconds to HH:MM:SS ---
def seconds_to_timestr(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# --- Hardware Usage Debug Print ---
def print_hardware_usage():
    print("=== Hardware Usage ===")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    mem = psutil.virtual_memory()
    print(f"Memory Usage: {mem.used / (1024 ** 3):.1f} GB / {mem.total / (1024 ** 3):.1f} GB ({mem.percent}%)")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
    print("======================\n")

# --- Audio Preprocessing ---
def preprocess_audio(audio_file, sr=16000):
    waveform, sr = librosa.load(audio_file, sr=sr)
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    return waveform, sr

# --- Utility Functions ---
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("No audio track found in the video.")
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.reader.close()
    clip.audio.reader.close_proc()

def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename

def transcribe_audio(audio_file, language=None):
    try:
        # Use a Speech-to-Text model
        stt = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=torch.cuda.is_available())
        transcription = stt.stt(audio_file)
        detected_language = "en"  # Assuming English for now
    except Exception as e:
        logging.error("Error in audio transcription: %s", str(e))
        transcription = ""
        detected_language = "unknown"
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return transcription, detected_language
    
def detect_audio_events(audio_file):
    try:
        waveform, sr = librosa.load(audio_file, sr=32000)
        segment_length = 5 * sr  # 5-second segments
        events = {}
        for i in range(0, len(waveform), segment_length):
            segment = waveform[i:i + segment_length]
            if len(segment) == 0:
                continue
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
            try:
                output = panns_model.inference(segment_tensor)
                if isinstance(output, dict) and "clipwise_output" in output:
                    clipwise_output = np.array(output["clipwise_output"], dtype=float)
                else:
                    clipwise_output = np.array(output, dtype=float)
                if np.max(clipwise_output) < 0.1:
                    continue
                top_idx = int(np.argmax(clipwise_output))
                event_label = pann_labels[top_idx] if top_idx < len(pann_labels) else "Unknown"
                timestamp = i / sr
                if event_label in events:
                    events[event_label].append(seconds_to_timestr(timestamp))
                else:
                    events[event_label] = [seconds_to_timestr(timestamp)]
            except Exception as e:
                logging.error(f"Error processing audio segment: {e}")
        if not events:
            return {"No event": []}
        return events
    except Exception as e:
        logging.error("Error in audio event detection: %s", str(e))
        return {"Error": []}

# --- (OCR functionality removed completely) ---

def clean_report(text):
    text = re.sub(r'[\u06F0-\u06F9]+', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text

def free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Helper to describe detection position ---
def describe_position(x_norm, y_norm):
    if x_norm < 0.33:
        horz = "left"
    elif x_norm < 0.66:
        horz = "center"
    else:
        horz = "right"
    if y_norm < 0.33:
        vert = "top"
    elif y_norm < 0.66:
        vert = "middle"
    else:
        vert = "bottom"
    return f"{horz}, {vert}"

def article_for(label):
    return "an" if label[0].lower() in "aeiou" else "a"

# --- Function to get available Ollama models dynamically ---
def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            logging.error("Error calling 'ollama list': %s", result.stderr)
            return []
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception as e:
        logging.error("Exception in get_ollama_models: %s", str(e))
        return []

# --- Global Model Loading (with GPU memory cleanup after each load) ---
logging.info("Loading PANNs audio detection model...")
panns_model = AudioTagging(checkpoint_path=PANN_MODEL_PATH)
free_gpu_memory()
print_hardware_usage()

# --- Modified LLM Integration Functions ---
def call_ollama(prompt, input_text, model):
    try:
        combined = prompt + "\n\n" + input_text
        result = subprocess.run(
            ["ollama", "run", model],
            input=combined,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode != 0:
            logging.error("Ollama call failed with code %d: %s", result.returncode, result.stderr)
            return "LLM call failed."
        output = re.sub(r'<think>.*?</think>', '', result.stdout, flags=re.DOTALL).strip()
        if not output:
            output = "No response received."
        print(f"LLM ({model}) output:\n{output}\n")
        return output
    except Exception as e:
        logging.error("Error calling Ollama: %s", str(e))
        return "Error in LLM call."

def ollama_summarize_report(report_file, model):
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report_text = f.read()
    except Exception as e:
        logging.error("Error reading report file: %s", str(e))
        return ""
    clean_text = clean_report(report_text)
    prompt = (
        '''
        You are an expert video content summarizer. Generate a cohesive, engaging, and concise narrative summary (less than 100 words) of the video based on the following report. Do not include timestamps, technical details, or model names—write in plain, natural language.
        '''
    )
    summary = call_ollama(prompt, clean_text, model=model)
    if "LLM call failed" in summary or "Error in LLM call" in summary:
        summary = "The video presents a dynamic scene with various events, blending spoken words and visuals into an engaging narrative."
    return summary

def generate_video_description(speak_summary=False):
    report_file = "report.txt"
    if not os.path.exists(report_file):
        logging.error("Report file not found for summarization.")
        return None, None
    summary = ollama_summarize_report(report_file, model=selected_summarization_model.get())
    description_file = "video_description.txt"
    audio_file = "summary_audio.wav"
    try:
        with open(description_file, "w", encoding="utf-8") as f:
            f.write("Video Narrative Summary:\n")
            f.write(summary)
        logging.info("Video description generated as %s", description_file)
        if speak_summary:
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())
            tts.tts_to_file(text=summary, file_path=audio_file)
            return description_file, audio_file

    except Exception as e:
        logging.error("Error generating video description: %s", str(e))
        return None, None
    return description_file, None

# --- GPU Memory Cleanup ---
def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Main Video Processing Function ---
def process_video(video_path, sample_rate=1, draw_boxes=True, save_video=False, show_video=False, ocr_languages="eng"):
    if not os.path.exists(video_path):
        logging.error("File does not exist: %s", video_path)
        return

    # --- Video Properties ---
    clip = VideoFileClip(video_path)
    fps = clip.fps
    frame_count = int(clip.reader.nframes)
    duration = clip.duration
    width, height = clip.size
    video_format = os.path.splitext(video_path)[1].lower()
    clip.reader.close()
    if clip.audio is not None:
        clip.audio.reader.close_proc()

    logging.info("Video properties: Duration: %s, Frames: %d, FPS: %.2f, Resolution: %dx%d, Format: %s",
                 seconds_to_timestr(duration), frame_count, fps, width, height, video_format)

    # --- Build Report Header ---
    report_lines = []
    report_lines.append("Video Processing Report")
    report_lines.append("-----------------------")
    report_lines.append(f"File: {video_path}")
    report_lines.append(f"Duration: {seconds_to_timestr(duration)} ({duration:.2f} seconds)")
    report_lines.append(f"Resolution: {width}x{height}")
    report_lines.append(f"FPS: {fps:.2f}")
    report_lines.append(f"Total Frames: {frame_count}")
    report_lines.append(f"Format: {video_format}")
    report_lines.append("")

    # --- Audio Analysis ---
    temp_audio_path = "temp_audio.wav"
    logging.info("Extracting audio from video...")
    try:
        extract_audio(video_path, temp_audio_path)
    except Exception as e:
        logging.error("Error extracting audio: %s", str(e))
        temp_audio_path = None

    audio_transcript = ""
    detected_lang = "unknown"
    audio_for_video = "audio_for_video.wav"
    if temp_audio_path and os.path.exists(temp_audio_path):
        shutil.copy(temp_audio_path, audio_for_video)
        try:
            logging.info("Transcribing audio with Coqui STT...")
            audio_transcript, detected_lang = transcribe_audio(temp_audio_path)
            logging.info("Audio transcription: %s", audio_transcript)
            logging.info("Detected language for transcription: %s", detected_lang)
        except Exception as e:
            logging.error("Error in audio transcription: %s", str(e))
        try:
            logging.info("Detecting audio events using PANNs (5-sec segments)...")
            audio_events = detect_audio_events(temp_audio_path)
            logging.info("Detected audio events: %s", audio_events)
        except Exception as e:
            logging.error("Error in audio event detection: %s", str(e))
            audio_events = {"Error": []}
        os.remove(temp_audio_path)
    else:
        logging.info("No audio extracted.")
        audio_events = {"No audio": []}

    report_lines.append("Audio Analysis:")
    report_lines.append(f"  Transcription: {audio_transcript if audio_transcript else 'N/A'}")
    if isinstance(audio_events, dict):
        for event, times in audio_events.items():
            times_str = ", ".join(times) if times else "N/A"
            report_lines.append(f"  Detected Audio Event - {event}: {times_str}")
    else:
        report_lines.append(f"  Detected Audio Event: {audio_events if audio_events else 'N/A'}")
    report_lines.append("")

    print_hardware_usage()
    free_gpu()

    # --- Load Advanced YOLO and BLIP Models (and free GPU memory after each load) ---
    logging.info("Loading advanced YOLO model (YOLOv8n)...")
    try:
        yolo_model = YOLO("yolov8n.pt")
        free_gpu()
    except Exception as e:
        logging.error("Error loading YOLO model: %s", str(e))
        return

    logging.info("Loading BLIP-2 captioning model (base variant)...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.to(device)
        free_gpu()
    except Exception as e:
        logging.error("Error loading BLIP model: %s", str(e))
        return

    # --- Prepare Annotated Video Writer if needed ---
    annotated_temp = "annotated_temp.mp4"
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_temp, fourcc, fps, (width, height))

    # --- Process Video Frames ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error: Could not open video file.")
        return

    processed_frame_count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            if save_video and writer is not None:
                writer.write(frame)
            continue

        processed_frame_count += 1
        current_time = frame_idx / fps
        time_str = seconds_to_timestr(current_time)

        # --- YOLO Detection ---
        results = yolo_model(frame)
        yolo_descriptions = []
        for result in results:
            if result.boxes is None or result.boxes.data is None:
                continue
            detections = result.boxes.data.cpu().numpy()
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cls_int = int(cls)
                label = CLASS_MAP.get(cls_int, f"Unknown")
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cx_norm = cx / width
                cy_norm = cy / height
                pos_descr = describe_position(cx_norm, cy_norm)
                phrase = f"{article_for(label)} {label} at {pos_descr}"
                yolo_descriptions.append(phrase)
        yolo_text = ", ".join(yolo_descriptions) if yolo_descriptions else None

        # --- BLIP Captioning ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        inputs = blip_processor(pil_img, return_tensors="pt").to(device)
        try:
            output_ids = blip_model.generate(
                **inputs,
                max_length=60,  # Allows more words (default is ~30)
                min_length=20,  # Forces captions to be at least 20 tokens long
                repetition_penalty=1.05,  # Reduces word repetition (higher value = less repetition)
                num_beams=5,  # Beam search improves caption quality (default is 1)
                length_penalty=0.6  # Encourages longer, more detailed captions
            )

            caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)

            # Remove adjacent repeated words from caption
            words = caption.split()
            if words:
                clean_words = [words[0]]
                for w in words[1:]:
                    if w.lower() != clean_words[-1].lower():
                        clean_words.append(w)
                caption = " ".join(clean_words)
            caption = caption if caption.strip() != "" else None
        except Exception as e:
            logging.error("Error generating BLIP caption: %s", str(e))
            caption = None

        # --- Build Log Line (only include non-N/A fields in plain language) ---
        log_fields = []
        if yolo_text:
            log_fields.append(yolo_text)
        if caption:
            log_fields.append(caption)
        log_line = f"Time {time_str}: " + " | ".join(log_fields)
        logging.info(log_line)
        report_lines.append(log_line)

        # --- Optionally, draw BLIP caption overlay on the frame ---
        if draw_boxes and caption:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0)
            margin = 10
            (txt_w, txt_h), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
            x_txt = int((width - txt_w) / 2)
            y_txt = height - margin
            cv2.rectangle(frame,
                          (x_txt - margin, y_txt - txt_h - margin),
                          (x_txt + txt_w + margin, y_txt + baseline + margin),
                          bg_color,
                          cv2.FILLED)
            cv2.putText(frame, caption, (x_txt, y_txt), font, font_scale, text_color, thickness, cv2.LINE_AA)

        if save_video and writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    logging.info("Finished processing video.")
    free_gpu()

    # --- Merge Annotated Video with Audio ---
    if save_video and os.path.exists(audio_for_video):
        try:
            video_clip = VideoFileClip(annotated_temp)
            audio_clip = AudioFileClip(audio_for_video)
            video_with_audio = video_clip.set_audio(audio_clip)
            final_video = os.path.splitext(video_path)[0] + "_annotated.mp4"
            video_with_audio.write_videofile(final_video, codec="libx264", audio_codec="aac")
            logging.info("Annotated video with audio saved as %s", final_video)
            os.remove(annotated_temp)
            os.remove(audio_for_video)
        except Exception as e:
            logging.error("Error merging audio with annotated video: %s", str(e))
            final_video = annotated_temp
            logging.info("Annotated video saved without audio as %s", final_video)

    # --- Write Final Report ---
    report_filename = "report.txt"
    try:
        with open(report_filename, "w", encoding="utf-8") as rpt:
            rpt.write("\n".join(report_lines))
        logging.info("Report generated as %s", report_filename)
    except Exception as e:
        logging.error("Error generating report: %s", str(e))

    logging.info("Final Audio Transcription: %s", audio_transcript)
    logging.info("Final Detected Audio Events: %s", audio_events)

    # --- Ollama Summarization Feature (using user-selected model) ---
    desc_file, _ = generate_video_description()
    if desc_file:
        logging.info("Video description generated: %s", desc_file)
    free_gpu()


# --- GUI Code ---
class VideoProcessingGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Video Analyzer")
        master.geometry("650x550")
        master.configure(bg="#2E2E2E")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#2E2E2E", foreground="#EAEAEA", font=("Helvetica", 11))
        style.configure("TButton", background="#007BFF", foreground="white", font=("Helvetica", 10, "bold"), borderwidth=1, relief="flat")
        style.map("TButton", background=[('active', '#0056b3')])
        style.configure("TCheckbutton", background="#2E2E2E", foreground="#EAEAEA", font=("Helvetica", 10), indicatorrelief="flat")
        style.configure("TEntry", fieldbackground="#3A3A3A", foreground="white", borderwidth=1, insertbackground="white")
        style.configure("TCombobox", fieldbackground="#3A3A3A", foreground="white", borderwidth=1, arrowcolor="white", selectbackground="#007BFF")
        style.configure("TSpinbox", fieldbackground="#3A3A3A", foreground="white", borderwidth=1, arrowcolor="white")
        style.configure("TFrame", background="#2E2E2E")
        style.configure("TLabelframe", background="#2E2E2E", foreground="white", bordercolor="#4A4A4A")
        style.configure("TLabelframe.Label", background="#2E2E2E", foreground="#EAEAEA", font=("Helvetica", 11, "bold"))

        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill="both", expand=True)

        # --- File Selection Frame ---
        file_frame = ttk.Labelframe(main_frame, text="1. Select Video", padding="10")
        file_frame.pack(fill="x", pady=5)
        self.video_path = tk.StringVar()
        self.video_entry = ttk.Entry(file_frame, textvariable=self.video_path, width=60)
        self.video_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.load_button = ttk.Button(file_frame, text="Browse...", command=self.load_video)
        self.load_button.pack(side="left")

        # --- Options Frame ---
        options_frame = ttk.Labelframe(main_frame, text="2. Set Options", padding="10")
        options_frame.pack(fill="x", pady=5)

        # Language Options
        lang_frame = ttk.Frame(options_frame)
        lang_frame.pack(fill="x", pady=5)
        self.auto_lang = tk.BooleanVar(value=True)
        self.auto_check = ttk.Checkbutton(lang_frame, text="Auto-detect language for transcription", variable=self.auto_lang, command=self.toggle_language_options)
        self.auto_check.pack(anchor="w")
        
        self.languages = [
            ("English", "en"), ("Spanish", "es"), ("French", "fr"), ("German", "de"), ("Italian", "it"),
            ("Portuguese", "pt"), ("Polish", "pl"), ("Turkish", "tr"), ("Russian", "ru"), ("Dutch", "nl"),
            ("Czech", "cs"), ("Arabic", "ar"), ("Chinese", "zh-cn"), ("Japanese", "ja"), ("Hungarian", "hu"),
            ("Korean", "ko"), ("None", "none")
        ]
        lang_names = [f"{name} ({code})" for name, code in self.languages]
        self.primary_lang = tk.StringVar(value=lang_names[0])
        
        prim_lang_frame = ttk.Frame(options_frame)
        prim_lang_frame.pack(fill="x", pady=2)
        ttk.Label(prim_lang_frame, text="Primary Language:").pack(side="left", padx=5)
        self.primary_menu = ttk.Combobox(prim_lang_frame, textvariable=self.primary_lang, values=lang_names, state="disabled")
        self.primary_menu.pack(side="left", fill="x", expand=True)

        # Other Options
        other_opts_frame = ttk.Frame(options_frame)
        other_opts_frame.pack(fill="x", pady=5)
        self.save_video = tk.BooleanVar(value=True)
        self.save_check = ttk.Checkbutton(other_opts_frame, text="Save Annotated Video", variable=self.save_video)
        self.save_check.pack(side="left", padx=5)
        
        ttk.Label(other_opts_frame, text="Frame Sample Rate:").pack(side="left", padx=(20, 5))
        self.sample_rate = tk.IntVar(value=90)
        self.sample_spin = ttk.Spinbox(other_opts_frame, from_=1, to=999, textvariable=self.sample_rate, width=5)
        self.sample_spin.pack(side="left")

        # --- Summarization Frame ---
        summary_frame = ttk.Labelframe(main_frame, text="3. Summarization", padding="10")
        summary_frame.pack(fill="x", pady=5)
        
        ttk.Label(summary_frame, text="LLM Model:").pack(side="left", padx=5)
        self.available_models = get_ollama_models()
        self.selected_model = tk.StringVar(value=self.available_models[0] if self.available_models else "N/A")
        self.model_menu = ttk.Combobox(summary_frame, textvariable=self.selected_model, values=self.available_models, state="readonly")
        self.model_menu.pack(side="left", fill="x", expand=True)

        # --- Control and Status Frame ---
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=5)
        
        self.progress = ttk.Progressbar(control_frame, orient="horizontal", mode="indeterminate", length=300)
        self.progress.pack(pady=5)
        
        self.status_label = ttk.Label(control_frame, text="Status: Ready", anchor="center")
        self.status_label.pack(pady=5)

        # --- Post-processing Frame ---
        post_frame = ttk.Labelframe(main_frame, text="4. View Results", padding="10")
        post_frame.pack(fill="x", pady=5)
        
        btn_frame = ttk.Frame(post_frame)
        btn_frame.pack()

        self.play_button = ttk.Button(btn_frame, text="Play Video", command=self.play_video, state="disabled")
        self.play_button.pack(side="left", padx=5)
        self.open_report_button = ttk.Button(btn_frame, text="Open Report", command=self.open_report, state="disabled")
        self.open_report_button.pack(side="left", padx=5)
        self.summarize_button = ttk.Button(btn_frame, text="Summarize & Speak", command=self.summarize_and_speak, state="disabled")
        self.summarize_button.pack(side="left", padx=5)

        self.toggle_language_options()
        self.annotated_video_path = None
        
    def load_video(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filepath:
            self.video_path.set(filepath)
            self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")

    def toggle_language_options(self):
        state = "disabled" if self.auto_lang.get() else "readonly"
        self.primary_menu.config(state=state)
        
    def start_processing(self):
        video_file = self.video_path.get()
        if not video_file or not os.path.exists(video_file):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        transcribe_language = None if self.auto_lang.get() else self.get_lang_code(self.primary_lang.get())
        
        sample_rate = self.sample_rate.get()
        save_video = self.save_video.get()

        global selected_summarization_model
        selected_summarization_model = self.selected_model

        self.status_label.config(text="Status: Processing started...")
        self.start_button.config(state="disabled")
        self.progress.start()

        def processing_task():
            process_video(video_file, sample_rate=sample_rate, draw_boxes=True, save_video=save_video, ocr_languages=transcribe_language)
            if save_video:
                self.annotated_video_path = os.path.splitext(video_file)[0] + "_annotated.mp4"
            self.master.after(0, self.processing_complete)

        threading.Thread(target=processing_task, daemon=True).start()

    def processing_complete(self):
        self.progress.stop()
        self.status_label.config(text="Status: Processing completed.")
        self.start_button.config(state="normal")
        self.play_button.config(state="normal" if self.annotated_video_path else "disabled")
        self.open_report_button.config(state="normal")
        self.summarize_button.config(state="normal")

    def get_lang_code(self, lang_display):
        match = re.search(r'\((\w+)\)', lang_display)
        return match.group(1) if match else "en"

    def play_video(self):
        if self.annotated_video_path and os.path.exists(self.annotated_video_path):
            try:
                if platform.system() == "Windows":
                    os.startfile(self.annotated_video_path)
                else:
                    subprocess.Popen(["open", self.annotated_video_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open video: {e}")
        else:
            messagebox.showerror("Error", "Annotated video not found.")

    def open_report(self):
        report_path = "report.txt"
        if os.path.exists(report_path):
            try:
                if platform.system() == "Windows":
                    os.startfile(report_path)
                else:
                    subprocess.Popen(["open", report_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open report file: {e}")
        else:
            messagebox.showerror("Error", "Report file not found.")

    def summarize_and_speak(self):
        self.status_label.config(text="Status: Summarizing and generating speech...")
        self.summarize_button.config(state="disabled")
        
        def task():
            desc_file, audio_file = generate_video_description(speak_summary=True)
            self.master.after(0, lambda: self.on_summarize_complete(desc_file, audio_file))
        
        threading.Thread(target=task, daemon=True).start()

    def on_summarize_complete(self, desc_file, audio_file):
        self.status_label.config(text="Status: Ready")
        self.summarize_button.config(state="normal")
        if audio_file and os.path.exists(audio_file):
            try:
                wave_obj = sa.WaveObject.from_wave_file(audio_file)
                play_obj = wave_obj.play()
                # We don't wait for it to finish, just start playing.
                # If you need to wait, use: play_obj.wait_done()
            except Exception as e:
                messagebox.showerror("Error", f"Could not play summary audio: {e}")
        elif desc_file:
            messagebox.showinfo("Summary Generated", f"Summary saved to {desc_file}, but could not generate audio.")
        else:
            messagebox.showerror("Error", "Could not generate video summary.")

if __name__ == "__main__":
    root = tk.Tk()
    gui = VideoProcessingGUI(root)
    root.mainloop()
