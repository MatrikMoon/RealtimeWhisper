import socket
import pyaudio

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 960  # Same size as the sender

# Server Config (UDP)
HOST = "0.0.0.0"
PORT = 5000

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

# Setup UDP Socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

print(f"Listening for raw PCM audio on {HOST}:{PORT}...")

while True:
    try:
        raw_data, _ = server_socket.recvfrom(CHUNK * 2)  # Each sample is 2 bytes (16-bit PCM)
        stream.write(raw_data)
    except Exception as e:
        print(f"Error: {e}")
        break