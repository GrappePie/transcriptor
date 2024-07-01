
# Whisper Real-Time Subtitles

This project provides a real-time subtitles application using the Whisper model for speech recognition, SoundDevice for audio capture, and PySide6 for the graphical user interface.

## Requirements

- Python 3.7 or higher
- CUDA (optional, for GPU usage)
- Python dependencies (see below)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/GrappePie/transcriptor.git
cd transcriptor
```

2. Install the required dependencies:

```bash
pip install torch sounddevice numpy whisper pyside6
```

## Usage

1. Ensure your audio input device (e.g., SteelSeries Sonar - Stream) is connected and configured.

2. Modify the configuration in the script if necessary:

- `TARGET_NAME`: The name of the audio device you want to use.
- `BUFFER_DURATION`: Duration of the buffer in seconds.
- `SAMPLE_RATE`: Sampling rate in Hz.
- `VOLUME_THRESHOLD`: Volume threshold to avoid false positives.
- `CLEAR_TEXT_DELAY`: Delay time to clear text in milliseconds.
- `SMOOTHING_WINDOW`: Smoothing window to combine subtitles.
- `TEMPERATURE`: Temperature to enhance transcription diversity.
- `BEAM_SIZE`: Beam size for the decoder.
- `LANGUAGE`: Language for transcription.

3. Run the application:

```bash
python main.py
```

## Code Structure

- `device_has_audio_activity`: Checks if a device has audio activity.
- `find_active_device`: Searches for active devices by name.
- `are_subtitles_similar`: Compares differences between subtitles.
- `smooth_subtitles`: Applies smoothing to subtitles.
- `SpeechRecognitionThread`: Handles speech recognition in a separate thread.
- `SubtitleWindow`: Displays subtitles in full screen.
- `main`: Main function to run the application.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- [OpenAI](https://www.openai.com/) for the Whisper model.
- [PySide](https://pyside.org/) for the GUI libraries.
- [SoundDevice](https://python-sounddevice.readthedocs.io/) for audio capture.
