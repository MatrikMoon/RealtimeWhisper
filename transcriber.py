import numpy as np
import speech_recognition as sr
import torch
import whisper
import wave
import threading
import pyaudio
import time
import httpx
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

# Audio Config
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 1

# Initialize PyAudio
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

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
                remaining_size = size

                while not frames:
                    while remaining_size > 0:
                        if self.frame_queue.empty():
                            time.sleep(0.1)
                        
                        chunk = self.frame_queue.get()
                        # print(f'Rame queue: {self.frame_queue.qsize()} {size} {remaining_size} {len(chunk)}')
                        frames.append(chunk)
                        remaining_size -= len(chunk)

                return b''.join(frames)

class SpeechTranscriber:
    def __init__(self, flush_callback, model="medium.en", energy_threshold=300, record_timeout=3, phrase_timeout=4):
        self.model_name = model
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.flush_callback = flush_callback

        self.phrase_time = None
        self.needs_flush = False
        self.processed_data_queue = Queue()
        self.data_queue = Queue()
        self.transcription = ['']
        self.running = False

        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        recorder = sr.Recognizer()
        recorder.energy_threshold = self.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        recorder.dynamic_energy_threshold = False

        source = StreamingAudioSource(frame_queue=self.data_queue)

        self.audio_model = whisper.load_model(self.model_name)
        print("Model loaded and ready to receive audio frames.")

        # with source:
        #     recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio:sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            self.processed_data_queue.put(data)
            # stream.write(data)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    def add_audio_frame(self, audio_data: bytes):
        """External method to add audio frames for processing."""
        self.data_queue.put(audio_data)

    def pcm_to_wav_log(self, pcm_data_queue: Queue):
        """ Convert raw PCM audio to WAV format """
        with wave.open('log.wav', "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(16000)  # 16kHz sample rate
            while not pcm_data_queue.empty():
                item = pcm_data_queue.get()
                wav_file.writeframes(item)
                pcm_data_queue.task_done()            
            wav_file.close()

    def process_audio(self):
        while self.running:
            now = datetime.utcnow()
            if self.needs_flush and self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                self.needs_flush = False
                self.flush('\n'.join(self.transcription))
                self.transcription = ['']

            if not self.processed_data_queue.empty():
                phrase_complete = False
                if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                    phrase_complete = True
                self.phrase_time = now
                
                audio_data = b''.join(self.processed_data_queue.queue)
                self.processed_data_queue.queue.clear()
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    # print('(pause)')
                    self.transcription.append(text)
                    self.needs_flush = True
                else:
                    # print('(no pause)')
                    self.transcription[-1] += " " + text
                    self.needs_flush = True

                # os.system('cls' if os.name=='nt' else 'clear')
                # for line in self.transcription:
                #     print(line)
                # print('', end='', flush=True)
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
        
        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)
        
        self.pcm_to_wav_log(self.data_queue)

        print("Processing stopped.")

    def flush(self, text: str):
        print(f"Flushing: {text}")
        bot_host = "http://192.168.1.102:8080/processVoice"
        payload = {
            "prompt": text,
            "userId": "moon323",
        }
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
        }
        with httpx.Client() as client:
            response = client.post(bot_host, json=payload, headers=headers, timeout=360)
            response.raise_for_status()

            # If the bot decides not to respond, just move on with life
            if response.status_code == 204:
                return
            
            json = response.json()
            self.flush_callback(json)
            print(f'Response: {json["response"]}')
        # pyautogui.write("say " + text + "\n", interval=0.01)  # Simulates key presses

if __name__ == "__main__":
    transcriber = SpeechTranscriber()
    transcriber.start_processing()
    input("Press Enter to stop...\n")
    transcriber.stop_processing()
