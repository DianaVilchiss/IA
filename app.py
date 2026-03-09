import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

models = {
    'scaler': None,
    'kmeans': None,
    'pca': None,
    'df': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_models', methods=['POST'])
def upload_models():
    try:
        f_scaler = request.files.get('scaler')
        f_kmeans = request.files.get('kmeans')
        f_pca = request.files.get('pca')
        f_csv = request.files.get('dataset')

        if not all([f_scaler, f_kmeans, f_pca, f_csv]):
            return jsonify({"success": False, "error": "Faltan archivos en el formulario"})

        models['scaler'] = joblib.load(f_scaler)
        models['kmeans'] = joblib.load(f_kmeans)
        models['pca'] = joblib.load(f_pca)
        models['df'] = pd.read_csv(f_csv, encoding="latin-1")

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if models['kmeans'] is None:
        return jsonify({"success": False, "error": "Primero debes subir los modelos"})
    
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = models['scaler'].transform(features)
        cluster = models['kmeans'].predict(features_scaled)
        coords = models['pca'].transform(features_scaled)

        return jsonify({
            "success": True,
            "cluster": int(cluster[0]),
            "pca1": float(coords[0][0]),
            "pca2": float(coords[0][1])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/dataset_stats')
def dataset_stats():
    if models['df'] is None:
        return jsonify({"success": False, "error": "No hay datos cargados"})
    
    df = models['df']
    cols = ["SALES", "QUANTITYORDERED", "PRICEEACH", "MSRP"]
    corr = df[cols].corr().values.tolist()
 
    cluster_counts = [0, 0, 0, 0]
    for v in df["SALES"]:
        if v < 2000: cluster_counts[0] += 1
        elif v < 5000: cluster_counts[1] += 1
        elif v < 9000: cluster_counts[2] += 1
        else: cluster_counts[3] += 1

    return jsonify({
        "success": True,
        "sales_hist": df["SALES"].tolist(),
        "corr_matrix": corr,
        "corr_labels": cols,
        "cluster_labels": ["Bajo", "Medio", "Alto", "Premium"],
        "cluster_counts": cluster_counts,
        "total_rows": int(len(df)),
        "avg_sales": float(df["SALES"].mean()),
        "max_sales": float(df["SALES"].max()),
        "min_sales": float(df["SALES"].min()),
        "std_sales": float(df["SALES"].std()),
        "null_count": int(df.isnull().sum().sum()) 
    })

if __name__ == '__main__':
    app.run(debug=True)