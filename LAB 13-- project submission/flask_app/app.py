import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('flask_app/model/model_svc.pkl', 'rb'))
genre_mapping = {
    'lofi': 0,
    'sad': 1,
    'bittersweet': 2,
    'happy': 3,
    'energetic': 4,
    'romantic': 5,
    'chill': 6,
    'upbeat': 7,
    'melancholy': 8,
    'party': 9
}

artist_mapping = {
    0: 'Jungkook',
    1: 'RM',
    2: 'J-Hopeft.J.Cole',
    3: 'Jimin',
    4: 'J-Hope',
    5: 'Jin',
    6: 'V',
    7: 'Suga',
    8: 'Jungkookft.CharliePuth',
    9: 'Sugaft.IU',
    10: 'Retro Resonance',
    11: 'Urban Rhapsody',
    12: 'Sofia Carter',
    13: 'ElectroPulse',
    14: 'Firefly Symphony',
    15: 'Neon Vortex',
    16: 'Liam Harper',
    17: 'Jay Zenith',
    18: 'The Midnight Howl',
    19: 'Shadow Beats',
    20: 'Samantha Lee',
    21: 'Aurora Soundwave',
    22: 'DJ Thunder',
    23: 'Horizon Flow',
    24: 'EchoSync',
    25: 'Ava & The Ocean',
    26: 'Crimson Echo',
    27: 'Celestial Harmony',
    28: 'Bass Surge',
    29: 'Luna Nova'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        genre = request.form['genre'].lower()
        if genre not in genre_mapping:
            raise ValueError("Invalid genre selected.")
        
        tiktok_virality = float(request.form['tiktok'])
        release_year = int(request.form['release'])
        genre_value = genre_mapping[genre]
        features = np.array([[genre_value, tiktok_virality, release_year]])
        prediction = model.predict(features)
        artist_name = artist_mapping.get(prediction[0], "Unknown Artist")
        return render_template('index.html', prediction_text=f'Predicted Artist: {artist_name}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)