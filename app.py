from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import asyncio
import pickle as pkl
from googletrans import Translator
import whisper
import google.generativeai as genai
from moviepy.video.io.VideoFileClip import VideoFileClip
from pytube import YouTube
from dotenv import load_dotenv
import os
import yt_dlp

# Initialize the translator
translator = Translator()
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
UPLOAD_FOLDER1 = "static/uploads1"
UPLOAD_FOLDER2 = "static/uploads2"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["UPLOAD_FOLDER1"] = UPLOAD_FOLDER1
app.config["UPLOAD_FOLDER2"] = UPLOAD_FOLDER2
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER1, exist_ok=True)
os.makedirs(UPLOAD_FOLDER2, exist_ok=True)
whisper_model = whisper.load_model("small")

emotion_array = ['sadness', 'joy' , 'love', 'anger', 'fear', 'suppride']
emotion_emoji = [ '😞 or 😢', '😊 or 😄', '❤️ or 😍', '😡 or 😠', '😨 or 😱', '😌 or 🦚']

model = load_model('trained_weights/lstm_best_model.h5')
tkn = pkl.load(open('trained_weights/wordpiece.pkl', 'rb'))
v = joblib.load("trained_weights/tfv.pkl")
news = joblib.load("trained_weights/fake_news_model.pkl")
genai_model = genai.GenerativeModel("gemini-pro")

chat = genai_model.start_chat()



@app.route("/chatbot", methods=['POST'])
def chatbot():
    question  = request.form.get("question")
    response = chat.send_message(question, stream=True)
    response.resolve()
    answer=""
    for chunk in response:
        answer += chunk.text

    # result = f"You: {question}<br><b>Bot:</b> {answer}<br><br>"
    print(answer)
    return str(answer)

@app.route("/video_emotion_detection", methods=['POST'])
def video_emotion_detection():
    url = request.form.get("yt_url")
    print("Route hit")
    title = "audio"
    ydl_opts = {
        "format": "bestaudio/best",  
        "outtmpl": f"static/uploads1/{title}.%(ext)s",  
        "postprocessors": [{
            "key": "FFmpegExtractAudio",  
            "preferredcodec": "mp3", 
            "preferredquality": "192",  
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    model = whisper.load_model("small")

    URL = "static/uploads1/audio.mp3"
    model = whisper.load_model("small")
    result = model.transcribe(URL)
    print(result["text"])

    return result["text"]


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file1" not in request.files:
        return jsonify({"error": "No file uploaded!"})
    file = request.files["file1"]
    if file.filename == "":
        return jsonify({"error": "No selected file!"})
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    result = whisper_model.transcribe(filepath)
    transcript = result["text"]
    return transcript

@app.route("/translate", methods=["POST"])
def translate():
    if "file2" not in request.files:
        return jsonify({"error": "No file uploaded!"})
    file = request.files["file2"]
    if file.filename == "":
        return jsonify({"error": "No selected file!"})
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    result = whisper_model.transcribe(filepath, task="translate")
    transcript = result["text"]
    return transcript

@app.route('/predict', methods=['POST'])
async def predict():
    # sample_text = "i would think that whomever would be lucky enough to stay in this suite must feel like it is the most romantic place on earth"
    sample_text = request.form.get('sentence')
    print(f"Type of sample_text: {type(sample_text)}, Value: {sample_text}")
    language = request.form.get('language')
    print(f"Detected language: {language}")
    
    # Translate the text if it is not already in English
    if language != 'en':
        translated_text =await translator.translate(sample_text, src=language, dest='en')
    else:
        translated_text = sample_text

    print(f"Translated text: {translated_text}")
    output = tkn.encode(sample_text)
    print("Tokens:", output.tokens)
    print("Token IDs:", output.ids)
    data = output.ids
    # Convert the list to a NumPy array and reshape it to add batch dimension
    data = np.array(data)  # Convert list to array
    data = np.reshape(data, (1, -1))  # Add batch dimension (1, len(data))
    # Now you can make the prediction
    result = model.predict(data)[0]
    print(result)
    answer_idx = np.argmax(result)
    print(f"Emotion: {emotion_array[answer_idx]}")
    print(f"Emoji: {emotion_emoji[answer_idx]}")
    answer = str(emotion_array[answer_idx]) + emotion_emoji[answer_idx]
    return answer


@app.route('/detectFakeReal', methods=['POST'])
def detectFakeReal():
    data = request.form.get('review')
    data = list(data)
    data = v.transform(data)
    y_pred = news.predict(data)[0]
    print(y_pred)
    # idx = np.argmax(y_pred)
    # result = idx
    result=""
    if(y_pred == 1):
        result = "News is Fake"
    else:
        result = "News is Real"
    return result

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)