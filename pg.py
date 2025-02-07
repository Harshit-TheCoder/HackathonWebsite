# import pafy
# import os

# # Set pafy to use yt-dlp
# pafy.backend = "internal"  # Ensures it works with yt-dlp



# video = pafy.new(url)
# DOWNLOAD_PATH = "static/uploads2"

# # Get best audio stream
# bestaudio = video.getbestaudio()
# bestaudio.download(filepath=os.path.join(DOWNLOAD_PATH, bestaudio.filename))

# print("Download complete!")

import whisper
import os
import yt_dlp
from pytube import YouTube

url = "https://www.youtube.com/watch?v=W6wVU5b5nQk"
def extract_audio(youtube_url):
    title = "audio"
    ydl_opts = {
        "format": "bestaudio/best",  # Best quality audio
        "outtmpl": f"static/uploads1/{title}.%(ext)s",  # Save with video title
        "postprocessors": [{
            "key": "FFmpegExtractAudio",  # Extract audio using FFmpeg
            "preferredcodec": "mp3",  # Convert to MP3
            "preferredquality": "192",  # Set quality
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    model = whisper.load_model("small")

    URL = "static/uploads1/audio.mp3"
    model = whisper.load_model("small")
    result = model.transcribe(URL)
    print(result["text"])


# Example usage

extract_audio(url)
