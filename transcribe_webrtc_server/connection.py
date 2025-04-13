from aiortc import (
    RTCDataChannel,
    RTCPeerConnection,
)

from transcriber.transcriber import SpeechTranscriber
from .dynamic_wav_audio_track import DynamicWavAudioTrack
from dataclasses import dataclass


@dataclass
class Connection:
    peer_connection: RTCPeerConnection
    reply_track: DynamicWavAudioTrack
    data_channel: RTCDataChannel
    transcriber: SpeechTranscriber
    name: str
