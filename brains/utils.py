import re
from io import BytesIO

from elevenlabs import generate, play
from openai import OpenAI

from brains import args

client = OpenAI()


def play_text(text: str):

    voice = generate(text=clear_markdown(text), voice=args.elevenlabs_voice_id)
    play(voice)


def get_transcription(audio: BytesIO) -> str:

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
        prompt="Do not transcribe if you think the audio is noise.",
    )
    return response.text


def clear_markdown(text):
    # Remove Markdown URLS: ![alt text](URL) or [link text](URL)
    text = re.sub(r"!\[[^\]]*\]\((.*?)\)|\[[^\]]*\]\((.*?)\)", "", text)

    # Remove inline Markdown links (just display the link text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove images: ![alt text](URL)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove headers, lists, blockquotes, in-line codes, and bold/italic markers
    text = re.sub(r"#|\*|>|`|_|~", "", text)
    # text = re.sub(r"#|\*|\-|>|`|_|~", "", text)

    # Remove HTML tags, if any
    text = re.sub(r"<.*?>", "", text)

    # Optional: Remove extra spaces and lines
    text = re.sub(r"\s+", " ", text).strip()

    return text
