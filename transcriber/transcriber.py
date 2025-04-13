import numpy as np
import speech_recognition as sr
import torch
import whisper
import threading
import pyaudio
import time
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from clearvoice.clearvoice import ClearVoice
from .utilities import save_to_wav, trim_silence
from .streaming_audio_source import StreamingAudioSource

# # Audio Config
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 8192

# # Initialize PyAudio
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

print(torch.version.cuda)
print(torch.version.__version__)
print(torch.cuda.is_available())


class SpeechTranscriber:
    def __init__(
        self,
        username,
        flush_callback,
        model="small.en",
        energy_threshold=200,
        record_timeout=3,
        phrase_timeout=5,
    ):
        self.username = username
        self.model_name = model
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.flush_callback = flush_callback
        self.last_callback_print = datetime.utcnow()
        self.last_processing_print = datetime.utcnow()
        self.last_flush_print = datetime.utcnow()

        self.last_phrase_time = None
        self.needs_flush = False
        self.processed_data_queue = Queue()
        self.clearvoice = ClearVoice(
            task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"]
        )
        self.data_queue = Queue()
        self.transcription = ""
        self.running = False

        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        recorder = sr.Recognizer()
        recorder.energy_threshold = self.energy_threshold

        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        recorder.dynamic_energy_threshold = False

        source = StreamingAudioSource(frame_queue=self.data_queue)

        self.audio_model = whisper.load_model(self.model_name)
        print("Model loaded and ready to receive audio frames.")

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes, run noise suppression, and push the result into the thread safe queue.
            data = audio.get_raw_data()

            # Enhance audio so it is only voice
            suppressed = self.clearvoice.process_bytes(data)
            found_voice, trimmed = trim_silence(suppressed)

            if found_voice:
                now = datetime.utcnow()
                # print("callback: ", now - self.last_callback_print)
                self.last_callback_print = now
                file_name = f'request_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}_original.wav'
                save_to_wav(file_name, data)
                file_name = f'request_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}_suppressed.wav'
                save_to_wav(file_name, suppressed)
                file_name = f'request_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}_trimmed.wav'
                save_to_wav(file_name, trimmed)
                self.processed_data_queue.put(suppressed)
                # stream.write(suppressed)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        recorder.listen_in_background(
            source, record_callback, phrase_time_limit=record_timeout
        )

    def add_audio_frame(self, audio_data: bytes):
        """External method to add audio frames for processing."""
        self.data_queue.put(audio_data)

    def process_audio(self):
        while self.running:
            now = datetime.utcnow()
            if (
                self.needs_flush
                and self.last_phrase_time
                and now - self.last_phrase_time > timedelta(seconds=self.phrase_timeout)
            ):
                debug_now = datetime.utcnow()
                # print("flush: ", debug_now - self.last_flush_print)
                self.last_flush_print = debug_now

                self.needs_flush = False
                self.flush(self.transcription)
                self.transcription = ""

            if not self.processed_data_queue.empty():
                self.last_phrase_time = now

                debug_now = datetime.utcnow()
                # print("whisper loop: ", debug_now - self.last_processing_print)
                self.last_processing_print = debug_now

                audio_data = b"".join(self.processed_data_queue.queue)
                self.processed_data_queue.queue.clear()

                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(
                        np.float32)
                    / 32768.0
                )
                result = self.audio_model.transcribe(
                    audio_np, fp16=torch.cuda.is_available()
                )
                text = result["text"].strip()

                self.transcription += " " + text
                self.needs_flush = True
            else:
                sleep(0.25)

    def start_processing(self):
        """Starts audio processing in a separate thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(
                target=self.process_audio, daemon=True)
            self.thread.start()
            print("Processing started in background thread.")

    def stop_processing(self):
        """Stops audio processing."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

        print("Processing stopped.")

    def flush(self, text: str):
        print(f"Flushing: {text}")
        self.flush_callback(self.username, text)
        # pyautogui.write("say " + text + "\n", interval=0.01)  # Simulates key presses
