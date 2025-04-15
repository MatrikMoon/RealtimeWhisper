import numpy as np
import torch
import whisper
import threading
import pyaudio
from datetime import datetime, timedelta
from queue import Queue
from clearvoice.clearvoice import ClearVoice
from .utilities import detect_noise, save_to_wav, trim_silence
from .streaming_audio_source import StreamingAudioSource

# # Audio Config
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 8192

# # Initialize PyAudio
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS,
# rate=RATE, output=True, frames_per_buffer=CHUNK)

print(torch.version.cuda)
print(torch.version.__version__)
print(torch.cuda.is_available())

audio_model = whisper.load_model("small.en")
clearvoice = ClearVoice(
    task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"]
)


class SpeechTranscriber:
    def __init__(
        self,
        username,
        personality,
        gender,
        sourcematerial,
        flush_callback,
        record_timeout=3,
        flush_after_silence_duration=2,
        max_recording_duration=60
    ):
        self.username = username
        self.personality = personality
        self.gender = gender
        self.sourcematerial = sourcematerial

        self.record_timeout = record_timeout
        self.flush_after_silence_duration = flush_after_silence_duration
        self.max_recording_duration = max_recording_duration
        self.flush_callback = flush_callback
        self.last_loud_audio_detected = None
        self.last_voice_detected = None
        self.record_at_least_this_much_audio = 0
        self.needs_whisper = False
        self.post_flush = False
        self.data_queue = Queue()
        self.loud_data_queue = Queue()
        self.voice_data_queue = Queue()
        self.running = False
        self.source = StreamingAudioSource(frame_queue=self.data_queue)

    def add_audio_frame(self, audio_data: bytes):
        """External method to add audio frames for processing."""
        self.data_queue.put(audio_data)

    def process_audio(self):
        while self.running:
            now = datetime.utcnow()

            # If the last loop was a flush, there's a really good chance that we just
            # spent a lot of time blocked, and there may be some pending loud packets
            # in the queue. In this case, we should reset the timers so that we don't
            # end up flushing due to silence immediately
            if self.post_flush:
                self.last_loud_audio_detected = None
                self.last_voice_detected = None
                self.post_flush = False

            seconds_per_frame = float(
                self.source.CHUNK) / self.source.SAMPLE_RATE

            # Read frame data, then detect loud audio
            frame_data = self.source.stream.read(self.source.CHUNK)

            # If loud audio is detected, we'll add the audio to the voice detection queue,
            # as well as at least record_timeout seconds worth afterwards
            if detect_noise(frame_data, self.source.SAMPLE_WIDTH):
                self.record_at_least_this_much_audio = self.source.SAMPLE_RATE * self.record_timeout
                self.last_loud_audio_detected = now

            if self.record_at_least_this_much_audio > 0:
                self.record_at_least_this_much_audio -= self.source.CHUNK
                self.loud_data_queue.put(frame_data)

            # This variable will be true when the environment is quiet.
            # It would be nice to be able to check whether the user has also stopped speaking for this,
            # but if we did, that condition would trip automatically when flush_after_silence_duration
            # is less than record_timeout
            # TODO: Perhaps we could check for recent voice when this is tripped, and only set it if none
            # is detected... But really, isn't that the same thing as just reducing the record_timeout?
            environment_is_quiet = (
                self.last_loud_audio_detected
                and now - self.last_loud_audio_detected > timedelta(seconds=self.flush_after_silence_duration)
            )

            # If the loud data queue has at least record_timeout seconds of audio,
            # we'll test it to see if there's voice in there (by suppressing it and seeing
            # if anything is left over). If there is voice, we'll add the suppressed audio
            # to the processing queue for whisper
            if (self.loud_data_queue.qsize() * seconds_per_frame) > self.record_timeout or (not self.loud_data_queue.empty() and environment_is_quiet):
                combined_audio_data = b''.join(self.loud_data_queue.queue)
                self.loud_data_queue.queue.clear()
                suppressed = clearvoice.process_bytes(combined_audio_data)
                found_voice = detect_noise(
                    suppressed.tobytes(), self.source.SAMPLE_WIDTH)
                if found_voice:
                    self.voice_data_queue.put(suppressed)
                    self.last_voice_detected = now
                    self.needs_whisper = True

            # This variable will be true when the user has stopped speaking.
            # Note: we check last_voice_detected against the max of record_timeout and
            # flush_after_silence_duration because if record_timeout is longer than
            # flush_after_silence_duration, we can't expect any voice processing within the
            # window of flush_after_silence_duration, and if flush_after_silence_duration is
            # longer than record_timeout, we don't want to flush yet anyway
            user_stopped_speaking = (
                environment_is_quiet
                or (
                    self.last_voice_detected
                    and now - self.last_voice_detected > timedelta(seconds=max(self.record_timeout, self.flush_after_silence_duration))
                )
            )

            # This variable will be True when we should expedite a flush.
            # It will indicate that we know the user has stopped talking, so we should
            # just go ahead and suppress the rest of the audio in the queue and hand
            # it to whisper
            # Also, just in case it's picking up a TV or something, we'll flush if we
            # hot 60 seconds of recording
            # Note: Voice data queue is populated with suppressed data, which is
            # accumulated in chunks of length record_timeout, so... Here we are
            # TODO: Does this actually fix the TV problem? What if the user starts talking
            # at the end of the 60 seconds, do we lose that?
            flush_now = self.needs_whisper and (
                user_stopped_speaking
                or (
                    self.max_recording_duration
                    and self.voice_data_queue.qsize() * self.record_timeout > self.max_recording_duration
                )
            )

            # If there has been silence for more than flush_after_silence_duration seconds,
            # or the loud audio has not been voice for more than flush_after_silence_duration seconds,
            # or if the user has been speaking for more than max_whisper_processing_duration seconds,
            # process the voice_data_queue with whisper
            if flush_now:
                self.needs_whisper = False

                audio_data = b"".join(self.voice_data_queue.queue)
                self.voice_data_queue.queue.clear()

                # file_name = f'request_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}_suppressed.wav'
                # save_to_wav(file_name, audio_data, self.source.SAMPLE_RATE)

                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(
                        np.float32)
                    / 32768.0
                )
                result = audio_model.transcribe(
                    audio_np, fp16=torch.cuda.is_available()
                )
                text = result["text"].strip()
                self.flush(text)
                self.post_flush = True

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
        self.flush_callback(self.username, self.personality,
                            self.gender, self.sourcematerial, text)
        # pyautogui.write("say " + text + "\n", interval=0.01)  # Simulates key presses
