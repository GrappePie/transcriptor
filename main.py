import sys
import sounddevice as sd
import numpy as np
import whisper
import torch
from collections import deque
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, QTimer
import difflib

# Whisper and Device Configuration
BUFFER_DURATION = 5  # Buffer duration in seconds
SAMPLE_RATE = 16000  # Sampling rate in Hz
TARGET_NAME = "SteelSeries Sonar - Stream"  # Audio device name (I am using a SteelSeries Sonar) for audio activity detection
VOLUME_THRESHOLD = 0.01  # Volume threshold to avoid false positives
CLEAR_TEXT_DELAY = 10000  # Delay time to clear text (milliseconds)
SMOOTHING_WINDOW = 5  # Smoothing window to combine subtitles
TEMPERATURE = 0.8  # Temperature to enhance transcription diversity
BEAM_SIZE = 5  # Beam size for the decoder
LANGUAGE = "es"  # Language for transcription

# Load the model on CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=device)

# Check if the device has audio activity
def device_has_audio_activity(device_index, samplerate):
    try:
        with sd.InputStream(device=device_index, channels=1, samplerate=samplerate):
            return True
    except Exception:
        return False

# Search for active devices by name
def find_active_device(target_name):
    devices = sd.query_devices()
    active_devices = []

    for i, device in enumerate(devices):
        if target_name in device['name'] and device['max_input_channels'] > 0:
            samplerate = int(device['default_samplerate'])
            if device_has_audio_activity(i, samplerate):
                active_devices.append(i)
                print(
                    f"Index: {i}, Name: {device['name']}, Input: {device['max_input_channels']}, Output: {device['max_output_channels']}")
    return active_devices

# Function to compare differences between subtitles
def are_subtitles_similar(text1, text2, threshold=0.5):
    """Check if two subtitles are similar based on a threshold"""
    return difflib.SequenceMatcher(None, text1, text2).ratio() >= threshold

# Function to apply smoothing to subtitles
def smooth_subtitles(subtitle_buffer):
    """Combine recent subtitles to improve fluency"""
    return ' '.join(subtitle_buffer)

# Class to handle speech recognition in a separate thread
class SpeechRecognitionThread(QThread):
    text_ready = Signal(str)

    def __init__(self, device_index):
        super().__init__()
        self.device_index = device_index
        self.buffer = deque(maxlen=int(BUFFER_DURATION * SAMPLE_RATE))
        self.subtitle_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.previous_text = ""

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        # Convert to float and normalize to range [-1, 1]
        audio_float = indata[:, 0].astype(np.float32) / np.iinfo(np.int16).max
        self.buffer.extend(audio_float)

    def run(self):
        with sd.InputStream(callback=self.audio_callback, dtype='int16', channels=1,
                            samplerate=SAMPLE_RATE, device=self.device_index):
            while True:
                if len(self.buffer) >= SAMPLE_RATE * BUFFER_DURATION:
                    audio_data = np.array(self.buffer, dtype=np.float32)
                    audio_data = whisper.pad_or_trim(audio_data)

                    # Evaluate volume to avoid false positives
                    volume = np.sqrt(np.mean(audio_data ** 2))
                    if volume < VOLUME_THRESHOLD:
                        continue  # Skip processing if volume is too low

                    mel = whisper.log_mel_spectrogram(audio_data).to(device)

                    # Perform transcription with enhanced decoding
                    options = whisper.DecodingOptions(language=LANGUAGE, beam_size=BEAM_SIZE, temperature=TEMPERATURE, without_timestamps=True)
                    result = model.decode(mel, options)

                    # Add to smoothing buffer
                    new_text = result.text.strip()
                    self.subtitle_buffer.append(new_text)

                    # Apply smoothing and compare with previous subtitles
                    smoothed_text = smooth_subtitles(self.subtitle_buffer)
                    if not are_subtitles_similar(smoothed_text, self.previous_text):
                        self.previous_text = smoothed_text
                        self.text_ready.emit(smoothed_text)

# Class to display subtitles in full screen
class SubtitleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.clear_subtitle)

    def initUI(self):
        # Configure the window
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        screen = QtWidgets.QApplication.primaryScreen()
        screen_resolution = screen.size()
        self.setGeometry(0, 0, screen_resolution.width(),
                         screen_resolution.height())  # Adjust to screen size
        self.move(0, 850)  # Move to the bottom of the screen

        # Configure the label for subtitles
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)  # Enable word wrapping
        self.label.setStyleSheet("QLabel { color : white; font-size: 36px; font-weight: bold; }")
        max_width = int(screen_resolution.width() * 0.8)  # Limit to 80% of screen width
        max_height = int(screen_resolution.height() * 0.2)  # Limit to 20% of screen height
        self.label.setGeometry((screen_resolution.width() - max_width) // 2,
                               screen_resolution.height() - max_height - 80,
                               max_width, max_height)

    def update_subtitle(self, text):
        self.label.setText(text)
        # Restart the timer to clear the text
        self.timer.start(CLEAR_TEXT_DELAY)

    def clear_subtitle(self):
        self.label.setText("")

# Main function to run the speech recognition application
def main():
    # Find the active device
    active_devices = find_active_device(TARGET_NAME)

    if not active_devices:
        print(f"No input devices with audio activity containing the name '{TARGET_NAME}'.")
        sys.exit(1)

    # Use the first active device found
    microphone = active_devices[0]

    # Start the PySide6 application
    app = QApplication(sys.argv)
    window = SubtitleWindow()
    window.showFullScreen()

    # Create and start the speech recognition thread
    speech_thread = SpeechRecognitionThread(microphone)
    speech_thread.text_ready.connect(window.update_subtitle)
    speech_thread.start()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
