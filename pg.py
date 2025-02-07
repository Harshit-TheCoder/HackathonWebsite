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

from pytube import YouTube  

url = "https://www.youtube.com/watch?v=W6wVU5b5nQk"
def extract_audio(youtube_url):

    yt = YouTube(youtube_url) 

    audio_stream = yt.streams.filter(only_audio=True).first()  

    audio_stream.download()  



# Example usage

extract_audio(url)
