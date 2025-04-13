

import asyncio
from transcribe_webrtc_server.transcribe_webrtc_server import TranscribeWebRTCServer


if __name__ == "__main__":
    server = TranscribeWebRTCServer()
    asyncio.run(server.start_server())
