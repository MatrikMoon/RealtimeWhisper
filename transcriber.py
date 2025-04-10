import numpy as np
import speech_recognition as sr
import torch
import whisper
import wave
import threading
import pyaudio
import time
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from clearvoice.clearvoice import ClearVoice

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

def save_to_wav(filename, audio_bytes, sample_rate=16000):
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)  # mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)


class StreamingAudioSource(sr.AudioSource):
    def __init__(self, frame_queue, rate=16000, channels=1, sample_width=2):
        self.frame_queue = frame_queue
        self.SAMPLE_RATE = rate
        self.CHANNELS = channels
        self.SAMPLE_WIDTH = sample_width  # 16-bit PCM
        self.CHUNK = 8192
        self.stream = self.AudioFrameStream(self.frame_queue, self.CHUNK)

    def __enter__(self):
        return self  # Allows using 'with' statements

    def __exit__(self, exc_type, exc_value, traceback):
        pass  # No cleanup needed

    class AudioFrameStream(object):
        def __init__(self, frame_queue: Queue, chunk: int):
            self.frame_queue = frame_queue
            self.lock = threading.Lock()
            self.CHUNK = chunk

        def read(self, size):
            """Read from the queue, returning silence if no data is available."""
            with self.lock:
                frames = []

                # We're still slightly unclear on why the *2 is necessary, but it may have something to do with the sample
                # originally being 96000 instaed of 48000?
                remaining_size = size * 2

                while not frames:
                    while remaining_size > 0:
                        if self.frame_queue.empty():
                            time.sleep(0.1)

                        chunk = self.frame_queue.get()
                        # print(f'Rame queue: {self.frame_queue.qsize()} {size} {remaining_size} {len(chunk)}')
                        frames.append(chunk)
                        remaining_size -= len(chunk)

                return b"".join(frames)


class SpeechTranscriber:
    def __init__(
        self,
        flush_callback,
        model="small.en",
        energy_threshold=200,
        record_timeout=2,
        phrase_timeout=4,
    ):
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
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
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
            self.thread = threading.Thread(target=self.process_audio, daemon=True)
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
        self.flush_callback(text)
        # pyautogui.write("say " + text + "\n", interval=0.01)  # Simulates key presses


def trim_silence(audio_np, sample_rate=16000, threshold=100, silence_duration_ms=800):
    abs_audio = np.abs(audio_np)
    is_loud = abs_audio > threshold
    silence_samples = int((silence_duration_ms / 1000) * sample_rate)

    silent_counter = 0
    speech_indices = []
    current_segment = []
    has_loud = False  # Track if current_segment contains loud audio

    for idx, loud in enumerate(is_loud):
        if loud:
            if silent_counter >= silence_samples and current_segment:
                speech_indices.extend(current_segment)
                current_segment = []
            silent_counter = 0
            current_segment.append(idx)
            has_loud = True
        else:
            silent_counter += 1
            if silent_counter < silence_samples:
                current_segment.append(idx)

    # Only add the final segment if it contains loud audio
    if current_segment and has_loud:
        speech_indices.extend(current_segment)

    trimmed_audio = audio_np[speech_indices]
    has_audio = trimmed_audio.size > 0

    return has_audio, trimmed_audio