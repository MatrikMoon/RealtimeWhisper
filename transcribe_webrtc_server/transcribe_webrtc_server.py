import asyncio
import re
import ssl
import base64
from datetime import datetime
import httpx
from aiohttp import web
from aiohttp_cors import setup as setup_cors, ResourceOptions
from aiortc import (
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
)
from .dynamic_wav_audio_track import DynamicWavAudioTrack
from .transcribing_audio_track import TranscribingAudioTrack
from .connection import Connection
from transcriber.transcriber import SpeechTranscriber


class TranscribeWebRTCServer:
    def __init__(self):
        super().__init__()
        self.connections: list[Connection] = []

    def flush_callback(self, username, personality, gender, source_material, transcribed_text):
        bot_host = "http://192.168.1.102:8080/processVoice"
        payload = {
            "prompt": transcribed_text,
            "userId": username,
            "personality": personality,
            "gender": gender,
            "sourceMaterial": source_material
        }
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
        }
        with httpx.Client() as client:
            response = client.post(bot_host, json=payload,
                                   headers=headers, timeout=360)
            response.raise_for_status()

            # If the bot decides not to respond, just move on with life
            if response.status_code == 204:
                return

            json = response.json()
            print(f'Response: {json["response"]}')

            # Save the decoded audio bytes to a WAV file
            audio_bytes = base64.b64decode(json["audio"])
            file_name = f'response_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.wav'
            with open(file_name, "wb") as f:
                f.write(audio_bytes)

            # Create a MediaPlayer to play the response audio
            for connection in self.connections:
                print(f'Playing response for {connection.name}')
                connection.reply_track.enqueue_wav(audio_bytes)

                self.loop.call_soon_threadsafe(
                    connection.data_channel.send, "FROMUSER>" + json["respondingTo"])
                self.loop.call_soon_threadsafe(
                    connection.data_channel.send, "FROMBOT>" + json["response"] + "AUDIO>" + json["audio"])
                # connection.reply_track.enqueue_wav(MediaPlayer(file_name).audio)

    async def offer(self, request):
        data = await request.json()
        peer_connection = RTCPeerConnection()
        reply_track = DynamicWavAudioTrack()

        peer_connection.addTrack(reply_track)

        self.connections.append(Connection(
            peer_connection, reply_track, None, None, None))

        @peer_connection.on("track")
        async def on_track(track):
            print(f"🎧 Received track: {track.kind}")
            if track.kind == "audio":
                print("🎙️ Audio track received. Starting processing...")
                # recorder = MediaRecorder(f"output_{id(pc)}.wav")
                # recorder.addTrack(TranscribingAudioTrack(track))
                # await recorder.start()

                found = next(
                    (c for c in self.connections if c.peer_connection == peer_connection), None)

                # Sleep until we either fail, or the transcriber is initialized
                while found.transcriber is None and peer_connection.connectionState not in ["closed", "failed", "disconnected"]:
                    await asyncio.sleep(1)

                transcribing_track = TranscribingAudioTrack(
                    found.transcriber, track)
                try:
                    while True:
                        await transcribing_track.recv()
                except Exception as e:
                    print("Audio processing ended:", e)

        @peer_connection.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            found = next(
                (c for c in self.connections if c.peer_connection == peer_connection), None)
            found.data_channel = channel

            @channel.on("message")
            def on_message(message):
                def extract_fields(s):
                    pattern = r'([A-Z]+)>(.*?)(?=[A-Z]+>|$)'
                    return {key: value.strip() for key, value in re.findall(pattern, s)}

                if isinstance(message, str) and message.startswith("NAME>"):
                    fields = extract_fields(message)
                    found.name = fields["NAME"]
                    found.transcriber = SpeechTranscriber(
                        found.name, fields["PERSONALITY"], fields["GENDER"], fields["SOURCEMATERIAL"], self.flush_callback)
                    found.transcriber.start_processing()
                    channel.send("<TRANSCRIBERWARMEDUP>")

        @peer_connection.on("connectionstatechange")
        def on_connection_state_change():
            if peer_connection.connectionState in ["closed", "failed", "disconnected"]:
                print(
                    f"🔴 Connection {id(peer_connection)} closed. Cleaning up.")
                self.connections = [
                    c for c in self.connections if c.peer_connection != peer_connection]

        offer_desc = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

        print("✅ Received WebRTC offer, sending answer.")
        await peer_connection.setRemoteDescription(offer_desc)
        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)

        print("📡 Sending WebRTC answer.")
        return web.json_response(
            {"sdp": peer_connection.localDescription.sdp,
                "type": peer_connection.localDescription.type}
        )

    async def ice_candidate(self, request):
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
                sdpMLineIndex=data["sdpMLineIndex"],
            )
            if self.connections:
                # Use the last created connection
                await self.connections[-1].peer_connection.addIceCandidate(candidate)
        return web.Response()

    async def cleanup(self):
        """Periodically remove inactive peer connections."""
        while True:
            await asyncio.sleep(10)
            for connection in self.connections:
                if connection.peer_connection.connectionState in ["closed", "failed", "disconnected"]:
                    print(
                        f"Cleaning up connection {id(connection.peer_connection)}")
                    self.connections = [
                        c for c in self.connections if c != connection]

    async def start_server(self):
        # Start the cleanup task in the background
        asyncio.create_task(self.cleanup())

        self.loop = asyncio.get_event_loop()

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(
            certfile="cert1.pem", keyfile="privkey1.pem")

        app = web.Application()
        cors = setup_cors(
            app,
            defaults={
                "*": ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods=["POST", "GET", "OPTIONS"],
                )
            },
        )

        app.router.add_post("/offer", self.offer)
        app.router.add_post("/ice-candidate", self.ice_candidate)
        for route in list(app.router.routes()):
            cors.add(route)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, port=5000, ssl_context=ssl_context)
        print("WebRTC server is running on HTTPS!")
        await site.start()

        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            for connection in self.connections:
                if connection.transcriber is not None:
                    connection.transcriber.stop_processing()

            await runner.shutdown()
            await runner.cleanup()
