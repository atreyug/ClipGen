from faster_whisper import WhisperModel

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            "tiny",
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
    all_words = []

    for segment in segments:
        transcript.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

        if not segment.words:
            continue

        for word in segment.words:
            word_text = word.word.strip()

            if not word_text:
                continue

            all_words.append({
                "text": word_text,
                "start": round(word.start, 3),
                "end": round(word.end, 3),
            })

    return transcript, all_words