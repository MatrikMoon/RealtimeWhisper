from av import AudioFrame
import numpy as np
from aiortc import (
    MediaStreamTrack,
)

from transcriber.transcriber import SpeechTranscriber


class TranscribingAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, transcriber: SpeechTranscriber, track: MediaStreamTrack):
        super().__init__()
        self.transcriber = transcriber
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        pcm_bytes = self.frame_to_pcm(frame)
        self.transcriber.add_audio_frame(pcm_bytes)
        return frame

    def frame_to_pcm(self, frame: AudioFrame):
        """Extract PCM bytes from an audio frame and resample to 16000Hz if needed."""
        pcm_bytes = frame.planes[0]
        if frame.sample_rate != 16000:
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            # Compute the resampling factor based on the input sample rate
            factor = (
                16000 / 96000
            )  # it should be frame.sample_rate, but for my current implementation, I've found that it lies
            new_length = int(len(pcm_array) * factor)
            resampled_array = np.interp(
                np.linspace(0, len(pcm_array), new_length, endpoint=False),
                np.arange(len(pcm_array)),
                pcm_array,
            ).astype(np.int16)
            return resampled_array.tobytes()
        return pcm_bytes
