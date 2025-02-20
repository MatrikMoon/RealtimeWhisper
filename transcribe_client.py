import socket
import pyaudio

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 960  # Same as Opus frame size, but it's raw PCM now

# Receiver (Server) Info
SERVER_IP = "127.0.0.1"  # Replace with actual receiver IP
PORT = 5000

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Setup UDP Socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Streaming raw PCM audio to {SERVER_IP}:{PORT}...")

while True:
    try:
        raw_audio = stream.read(CHUNK, exception_on_overflow=False)
        client_socket.sendto(raw_audio, (SERVER_IP, PORT))
    except Exception as e:
        print(f"Error: {e}")
        break
