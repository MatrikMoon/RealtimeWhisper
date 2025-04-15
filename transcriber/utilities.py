import numpy as np
import wave
import audioop


def detect_noise(audio_bytes: bytes, sample_width=2, threshold=400, debug=False):
    energy = audioop.rms(audio_bytes, sample_width)

    if debug:
        print(energy, energy > threshold)

    return energy > threshold


def trim_silence(audio_np: np.ndarray, sample_rate=16000, threshold=100, silence_duration_ms=800):
    abs_audio = np.abs(audio_np)
    is_loud = abs_audio > threshold
    silence_samples = int((silence_duration_ms / 1000) * sample_rate)

    silent_counter = 0
    speech_indices = []
    current_segment = []
    has_loud = False  # Track if current_segment contains loud audio

    for idx, loud in enumerate(is_loud):
        if loud:
            if silent_counter >= silence_samples and current_segment:
                speech_indices.extend(current_segment)
                current_segment = []
            silent_counter = 0
            current_segment.append(idx)
            has_loud = True
        else:
            silent_counter += 1
            if silent_counter < silence_samples:
                current_segment.append(idx)

    # Only add the final segment if it contains loud audio
    if current_segment and has_loud:
        speech_indices.extend(current_segment)

    trimmed_audio = audio_np[speech_indices]
    has_audio = trimmed_audio.size > 0

    return has_audio, trimmed_audio


def save_to_wav(filename, audio_bytes, sample_rate=16000):
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)  # mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
