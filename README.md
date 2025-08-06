#  AI-Powered Video Analyzer for Mantrahackathon 

This application provides a comprehensive analysis of video files, including object detection, scene captioning, audio transcription, and an AI-generated summary. The user-friendly interface allows you to easily load a video, configure the analysis, and view the results.

## Features

- **Object Detection**: Identifies a wide range of objects in each sampled frame using the YOLOv8n model.
- **Scene Captioning**: Generates descriptive captions for video scenes using the BLIP-2 model.
- **Audio Transcription**: Transcribes spoken words from the video's audio track using Coqui STT.
- **AI-Generated Summary**: Creates a concise, narrative summary of the video's content using a local LLM (via Ollama).
- **Audio Playback**: Reads the generated summary aloud using text-to-speech.
- **Customizable Analysis**: Adjust the frame sample rate to balance processing speed and level of detail.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8+**: This application is built with Python. You can download it from [python.org](https://www.python.org/downloads/).
- **Ollama**: The AI summarization feature is powered by a local large language model (LLM) running on Ollama. You will need to have Ollama installed and have pulled a model.
  - [Download and install Ollama](https://ollama.ai/).
  - Pull a model by running the following command in your terminal (we recommend `llama2`):
    ```bash
    ollama pull llama2
    ```
- **FFmpeg**: This is required for audio extraction from video files.
  - **macOS (using Homebrew)**: `brew install ffmpeg`
  - **Windows (using Chocolatey)**: `choco install ffmpeg`
  - **Linux (using apt)**: `sudo apt update && sudo apt install ffmpeg`

## Installation

To get the AI-Powered Video Analyzer up and running, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/arashsajjadi/ai-powered-video-analyzer.git
    cd ai-powered-video-analyzer
    ```
2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## How to Use

1.  **Run the Application**:
    ```bash
    python3 video_processing_gui.py
    ```
2.  **Load a Video**: Click the **Browse...** button to select an MP4 video file.
3.  **Set Options**:
    - **Frame Sample Rate**: Adjust the slider to determine how many frames to skip. A higher rate means faster processing but less detail.
    - **Save Annotated Video**: Check this box if you want to save a new video file with object detection boxes and captions overlaid.
4.  **Start Processing**: Click the **Start Processing** button. The progress bar will indicate that the analysis is underway.
5.  **View Results**:
    - **Play Video**: If you chose to save the annotated video, this button will play it.
    - **Open Report**: This will open a detailed, timestamped report of all detected objects and scene captions.
    - **Summarize & Speak**: This will generate an AI summary, save it as `video_description.txt`, and read it aloud.

## Included Models

This application automatically downloads and caches the following models upon first use:

- **YOLOv8n**: For object detection.
- **BLIP-2**: For image captioning.
- **PANNs**: For audio event detection.
- **Coqui STT/TTS**: For speech-to-text and text-to-speech.
