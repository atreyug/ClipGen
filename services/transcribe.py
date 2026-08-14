from faster_whisper import WhisperModel

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            "tiny",        # "tiny" uses ~150MB vs "small" ~500MB
            device="cpu",
            compute_type="int8"
        )
    return _model


def transcribe(path: str):
    model = _get_model()
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
