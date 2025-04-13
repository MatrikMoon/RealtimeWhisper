import asyncio
from fractions import Fraction
from queue import Queue
from av import AudioFrame
import time
import wave
import io
from aiortc import (
    MediaStreamTrack,
)


class MediaStreamError(Exception):
    pass


class DynamicWavAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self, sample_rate=48000, channels=2, sample_width=2, frame_duration=0.02
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.frame_duration = frame_duration
        self.start_time = time.time()
        self.queue = Queue()  # Queue to hold WAV file bytes
        self.current_wav = None
        self.timestamp = 0

    def enqueue_wav(self, wav_bytes: bytes):
        # Call this method to add new WAV file bytes to the track
        self.queue.put(wav_bytes)

    def enqueue_wav_bytes(self, wav_bytes: bytes):
        # Call this method to add new WAV file bytes to the track
        self.queue.put(wav_bytes)

    async def recv(self):
        # If no WAV file is currently playing, try to load one from the queue
        if self.current_wav is None:
            if not self.queue.empty():
                wav_bytes = self.queue.get()
                self.current_wav = wave.open(io.BytesIO(wav_bytes), "rb")
                self.sample_rate = self.current_wav.getframerate()
                self.channels = self.current_wav.getnchannels()
                self.sample_width = self.current_wav.getsampwidth()
            else:
                # No WAV data available; output silence
                return await self._create_silence_frame()

        # Read a chunk corresponding to frame_duration
        num_samples = int(self.sample_rate * self.frame_duration)
        raw_data = self.current_wav.readframes(num_samples)
        if not raw_data:
            # Finished current WAV file; clear it so next call can load a new one
            self.current_wav = None
            return await self._create_silence_frame()

        try:
            self.timestamp += num_samples
            fmt = "s16" if self.sample_width == 2 else "s32"
            layout = "stereo" if self.channels == 2 else "mono"
            frame = AudioFrame(format=fmt, layout=layout, samples=num_samples)
            for p in frame.planes:
                p.update(raw_data)

            frame.sample_rate = self.sample_rate
            frame.pts = self.timestamp
            frame.time_base = Fraction(1, self.sample_rate)

            wait = self.start_time + \
                (self.timestamp / self.sample_rate) - time.time()
            # print(f'{wait}  \t---  {self.start_time}\t{frame.pts}\t{self.sample_rate}')
            await asyncio.sleep(wait)
        except Exception as e:
            print(e)

        # print('fram')
        return frame

    async def _create_silence_frame(self):
        """
        Receive the next :class:`~av.audio.frame.AudioFrame`.

        The base implementation just reads silence, subclass
        :class:`AudioStreamTrack` to provide a useful implementation.
        """
        if self.readyState != "live":
            raise MediaStreamError

        samples = int(self.frame_duration * self.sample_rate)

        self.timestamp += samples
        wait = self.start_time + \
            (self.timestamp / self.sample_rate) - time.time()
        # print(f'{wait}  \t---  {self.start_time}\t{self.timestamp}\t{self.sample_rate}')
        await asyncio.sleep(wait)

        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = self.timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        return frame
