import asyncio
import ssl
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.media import MediaRecorder
from aiohttp_cors import setup as setup_cors, ResourceOptions

# Store active peer connections
pcs = set()  

class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        # print("🎤 Received audio frame!")  # ✅ Debug: Log received audio
        return frame

async def offer(request):
    global pcs  # Keep track of connections

    data = await request.json()
    pc = RTCPeerConnection()  # ✅ Create a NEW WebRTC connection per client
    pcs.add(pc)  # Store connection

    # Handle incoming tracks
    @pc.on("track")
    async def on_track(track):
        print(f"🎧 Received track: {track.kind}")  # ✅ Debug received track

        if track.kind == "audio":
            print("🎙️ Audio track received! Starting processing...")
            recorder = MediaRecorder(f"output_{id(pc)}.wav")  # Save per connection
            recorder.addTrack(AudioTrack(track))
            await recorder.start()

    # Handle disconnections (Cleanup)
    @pc.on("connectionstatechange")
    def on_connection_state_change():
        if pc.connectionState in ["closed", "failed", "disconnected"]:
            print(f"🔴 Connection {id(pc)} closed. Cleaning up.")
            pcs.discard(pc)
    
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    print("✅ Received WebRTC offer, sending answer.")
    
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print("📡 Sending WebRTC answer.")

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def ice_candidate(request):
    data = await request.json()
    print("🌍 Received ICE candidate:", data)  # ✅ Debug ICE candidates

    # Ensure the data is correctly structured
    if "candidate" in data:
        candidate = RTCIceCandidate(
            component=1,  # Usually 1 for RTP/RTCP
            foundation="0",  # A default value
            priority=1,  # Default priority
            protocol="udp",  # Use UDP
            type="host",  # Default type
            ip=data["candidate"],  # The actual ICE candidate string
            port=5000,  # The default port (change if needed)
            sdpMid=data["sdpMid"],
            sdpMLineIndex=data["sdpMLineIndex"]
        )

        # Apply ICE candidates to the most recent peer connection
        if pcs:
            pc = list(pcs)[-1]  # Use the last created connection
            await pc.addIceCandidate(candidate)

    return web.Response()

async def cleanup():
    """ Periodically clean up old connections """
    while True:
        await asyncio.sleep(10)
        for pc in list(pcs):
            if pc.connectionState in ["closed", "failed", "disconnected"]:
                print(f"🗑️ Cleaning up connection {id(pc)}")
                pcs.discard(pc)

app = web.Application()

# Enable CORS for all routes
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

# Apply CORS to all routes
for route in list(app.router.routes()):
    cors.add(route)

async def start_server():
    """ Starts the WebRTC server and cleanup task """
    # Start cleanup in background
    asyncio.create_task(cleanup())  # ✅ Use create_task() instead of ensure_future()

    # Load SSL certificate & key
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile="cert1.pem", keyfile="privkey1.pem")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, port=5000, ssl_context=ssl_context)
    print("🚀 WebRTC server is running on HTTPS!")
    await site.start()

    # ✅ Keep the event loop alive
    while True:
        await asyncio.sleep(3600)  # Sleep indefinitely to keep server running

# ✅ Properly start asyncio event loop
asyncio.run(start_server())