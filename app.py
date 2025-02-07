from flask import Flask, request, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load tokenizer dan model
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load model LSTM
model = load_model('lstm_model.h5')  # Ganti dengan nama model LSTM Anda yang disimpan

# Fungsi untuk mengubah teks ke dalam urutan angka (sequences)
def text_to_sequence(text, tokenizer, max_len=100):
    # Tokenisasi teks menggunakan tokenizer yang sudah disimpan
    sequences = tokenizer.texts_to_sequences([text])
    
    # Padding agar panjang urutan sama
    padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequence

# Fungsi untuk label sentimen dan probabilitas
def sentimen_label_with_probability(prediction):
    probability_positive = prediction  # Probabilitas untuk sentimen positif
    probability_negative = 1 - prediction  # Probabilitas untuk sentimen negatif
    
    
    # Tentukan label berdasarkan probabilitas
    sentiment_label = 'Positif' if probability_positive > 0.6 else 'Negatif'
    
    return sentiment_label, probability_positive, probability_negative

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            text = request.form['text']  # Ambil teks dari form

            # Ubah teks menjadi sequence
            processed_text = text_to_sequence(text, tokenizer)
            
            # Prediksi sentimen menggunakan model LSTM
            predicted_sentimen = model.predict(processed_text)[0][0]
            predicted_sentimen_label, prob_pos, prob_neg = sentimen_label_with_probability(predicted_sentimen)
            
            # Kirim hasil prediksi ke halaman index.html
            return render_template('index.html', text=text, prediksi=predicted_sentimen_label,
                                   prob_pos=round(prob_pos, 4), prob_neg=round(prob_neg, 4))
        except Exception as e:
            error_message = "Terjadi kesalahan: " + str(e)
            return render_template('index.html', error=error_message)
    return render_template('index.html')

if __name__ == "__main__":
    app.run()

