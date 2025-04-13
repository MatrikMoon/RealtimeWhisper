import speech_recognition as sr
import threading
import time
from queue import Queue


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
