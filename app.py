from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import logging

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pemetaan huruf ke angka
LINE_MAPPING = {'A': 0, 'B': 1, 'C': 2, 'D1': 3, 'E': 4, 'F1': 5, 'G1': 6, 'H1': 7, 'I': 8}
REVERSE_MAPPING = {v: k for k, v in LINE_MAPPING.items()}

# Load model dan scaler menggunakan joblib
try:
    rf_model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    logger.info("Model dan scaler berhasil dimuat")
except Exception as e:
    logger.error(f"Gagal memuat model atau scaler: {e}")
    rf_model = None
    scaler = None

def preprocess_data(input_data):
    # Konversi LINE dari huruf ke angka
    input_data['LINE'] = input_data['LINE'].map(LINE_MAPPING)

    # Mengonversi OUTPUT(KG) dan OEE menjadi integer
    input_data['OUTPUT(KG)'] = input_data['OUTPUT(KG)'].astype(int)
    input_data['OEE'] = input_data['OEE'].astype(int)

    # Transformasi log pada fitur input
    epsilon = 1e-10
    input_log = input_data.copy()
    input_log['LINE'] = np.log(input_log['LINE'] + epsilon)
    input_log['OUTPUT(KG)'] = np.log(input_log['OUTPUT(KG)'] + epsilon)
    input_log['OEE'] = np.log(input_log['OEE'] + epsilon)

    # Fitur rekayasa
    input_log['LINE_OUTPUT'] = input_log['LINE'] * input_log['OUTPUT(KG)']
    input_log['LINE_OEE'] = input_log['LINE'] * input_log['OEE']
    input_log['OUTPUT_OEE_RATIO'] = input_log['OUTPUT(KG)'] / input_log['OEE']
    input_log['OUTPUT_KG_SQUARED'] = input_log['OUTPUT(KG)'] ** 2

    return input_log

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request
        data = request.json
        
        # Validasi data
        if not data or 'predictions' not in data:
            return jsonify({'error': 'Data tidak valid'}), 400

        # Konversi data ke DataFrame
        input_data = pd.DataFrame(data['predictions'])
        
        # Preprocessing data
        input_log = preprocess_data(input_data)

        # Standarisasi fitur input
        input_scaled = scaler.transform(input_log)

        # Melakukan prediksi
        predicted_conversion_cost = rf_model.predict(input_scaled)

        # Menyiapkan hasil
        result_df = input_data.copy()
        result_df['Predicted Conversion Cost'] = predicted_conversion_cost.astype(int)
        result_df['LINE'] = result_df['LINE'].map(REVERSE_MAPPING)

        # Konversi hasil ke list dictionary
        results = result_df.to_dict('records')

        return jsonify(results)

    except Exception as e:
        logger.error(f"Kesalahan dalam prediksi: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)