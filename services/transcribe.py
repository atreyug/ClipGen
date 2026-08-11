from faster_whisper import WhisperModel


model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

def transcribe(path: str):
    segments, info = model.transcribe(
        path,
        word_timestamps=True
    )

    transcript = []

    for segment in segments:
        transcript.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    return transcript