from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import asyncio
import pickle as pkl
from googletrans import Translator

# Initialize the translator
translator = Translator()

app = Flask(__name__)

emotion_array = ['sadness', 'joy' , 'love', 'anger', 'fear', 'suppride']
emotion_emoji = [ '😞 or 😢', '😊 or 😄', '❤️ or 😍', '😡 or 😠', '😨 or 😱', '😌 or 🦚']

model = load_model('trained_weights/lstm_best_model.h5')
tkn = pkl.load(open('trained_weights/wordpiece.pkl', 'rb'))
v = joblib.load("trained_weights/tfv.pkl")

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


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)