import asyncio
from fractions import Fraction
from queue import Queue
import ssl
import base64
from datetime import datetime
from av import AudioFrame
import numpy as np
import time
import av
import wave
import io
from aiohttp import web
from aiohttp_cors import setup as setup_cors, ResourceOptions
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.media import MediaRecorder, MediaPlayer
from transcriber import SpeechTranscriber

# Global set of active peer connections
connections = set()

def flush_callback(response):
    # Save the decoded audio bytes to a WAV file
    audio_bytes = base64.b64decode(response["audio"])
    file_name = f'response_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.wav'
    with open(file_name, 'wb') as f:
        f.write(audio_bytes)
    
    # Create a MediaPlayer to play the response audio
    for (_, reply_track) in connections:
        casted_track: DynamicWavAudioTrack = reply_track
        print(f'Playing response: {response["response"]}')
        casted_track.enqueue_wav(audio_bytes)
        # casted_track.enqueue_wav(MediaPlayer(file_name).audio)

transcriber = SpeechTranscriber(flush_callback)

class MediaStreamError(Exception):
    pass

class DynamicWavAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000, channels=2, sample_width=2, frame_duration=0.02):
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
                self.current_wav = wave.open(io.BytesIO(wav_bytes), 'rb')
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
            fmt = 's16' if self.sample_width == 2 else 's32'
            layout = 'stereo' if self.channels == 2 else 'mono'
            frame = AudioFrame(format=fmt, layout=layout, samples=num_samples)
            for p in frame.planes:
                p.update(raw_data)

            frame.sample_rate = self.sample_rate
            frame.pts = self.timestamp
            frame.time_base = Fraction(1, self.sample_rate)

            wait = self.start_time + (self.timestamp / self.sample_rate) - time.time()
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
        wait = self.start_time + (self.timestamp / self.sample_rate) - time.time()
        # print(f'{wait}  \t---  {self.start_time}\t{self.timestamp}\t{self.sample_rate}')
        await asyncio.sleep(wait)

        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = self.timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        return frame

class TranscribingAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        pcm_bytes = self.frame_to_pcm(frame)
        transcriber.add_audio_frame(pcm_bytes)
        return frame

    def frame_to_pcm(self, frame: AudioFrame):
        """Extract PCM bytes from an audio frame and resample to 16000Hz if needed."""
        pcm_bytes = frame.planes[0]
        if frame.sample_rate != 16000:
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            # Compute the resampling factor based on the input sample rate
            factor = 16000 / 96000 # it should be frame.sample_rate, but for my current implementation, I've found that it lies
            new_length = int(len(pcm_array) * factor)
            resampled_array = np.interp(
                np.linspace(0, len(pcm_array), new_length, endpoint=False),
                np.arange(len(pcm_array)),
                pcm_array
            ).astype(np.int16)
            return resampled_array.tobytes()
        return pcm_bytes

async def offer(request):
    data = await request.json()
    pc = RTCPeerConnection()
    reply_track = DynamicWavAudioTrack()

    connections.add((pc, reply_track))
    pc.addTrack(reply_track)

    @pc.on("track")
    async def on_track(track):
        print(f"🎧 Received track: {track.kind}")
        if track.kind == "audio":
            print("🎙️ Audio track received. Starting processing...")
            # recorder = MediaRecorder(f"output_{id(pc)}.wav")
            # recorder.addTrack(TranscribingAudioTrack(track))
            # await recorder.start()

            transcribing_track = TranscribingAudioTrack(track)
            try:
                while True:
                    await transcribing_track.recv()
            except Exception as e:
                print("Audio processing ended:", e)

    @pc.on("connectionstatechange")
    def on_connection_state_change():
        if pc.connectionState in ["closed", "failed", "disconnected"]:
            print(f"🔴 Connection {id(pc)} closed. Cleaning up.")
            connections.discard((pc, reply_track))

    offer_desc = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

    print("✅ Received WebRTC offer, sending answer.")
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    print("📡 Sending WebRTC answer.")
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def ice_candidate(request):
    data = await request.json()
    print("🌍 Received ICE candidate:", data)
    if "candidate" in data:
        candidate = RTCIceCandidate(
            component=1,
            foundation="0",
            priority=1,
            protocol="udp",
            type="host",
            ip=data["candidate"],
            port=5000,
            sdpMid=data["sdpMid"],
            sdpMLineIndex=data["sdpMLineIndex"]
        )
        if connections:
            # Use the last created connection
            (pc, _) = list(connections)[-1]
            await pc.addIceCandidate(candidate)
    return web.Response()

async def cleanup():
    """Periodically remove inactive peer connections."""
    while True:
        await asyncio.sleep(10)
        for (pc, reply_track) in list(connections):
            if pc.connectionState in ["closed", "failed", "disconnected"]:
                print(f"Cleaning up connection {id(pc)}")
                connections.discard((pc, reply_track))

async def start_server():
    # Start the cleanup task in the background
    asyncio.create_task(cleanup())

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile="cert1.pem", keyfile="privkey1.pem")

    app = web.Application()
    cors = setup_cors(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["POST", "GET", "OPTIONS"],
        )
    })

    app.router.add_post("/offer", offer)
    app.router.add_post("/ice-candidate", ice_candidate)
    for route in list(app.router.routes()):
        cors.add(route)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, port=5000, ssl_context=ssl_context)
    print("WebRTC server is running on HTTPS!")
    await site.start()

    transcriber.start_processing()

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        transcriber.stop_processing()
        await runner.shutdown()
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(start_server())
