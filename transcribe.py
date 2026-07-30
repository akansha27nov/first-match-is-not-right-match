import os
from openai import OpenAI
from config import OPENAI_API_KEY, PODCAST_AUDIO_PATH, PODCAST_TRANSCRIPT_PATH

client = OpenAI(api_key=OPENAI_API_KEY)
MAX_BYTES = 25 * 1024 * 1024

def transcribe():
    if os.path.exists(PODCAST_TRANSCRIPT_PATH):
        print("Transcript already exists, skipping API call.")
        return

    size = os.path.getsize(PODCAST_AUDIO_PATH)
    if size > MAX_BYTES:
        raise ValueError(
            f"{PODCAST_AUDIO_PATH} is {size / 1e6:.1f}MB, over the 25MB Whisper API limit. "
            "Re-encode with -codec:a libmp3lame and a lower bitrate."
        )

    with open(PODCAST_AUDIO_PATH, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )

    with open(PODCAST_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(resp)

    print(f"Transcript saved: {len(resp)} chars")


if __name__ == "__main__":
    transcribe()