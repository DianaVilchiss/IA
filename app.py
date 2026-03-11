import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

models = {'scaler': None, 'kmeans': None, 'pca': None, 'df': None, 'df_cleaned': None}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_models', methods=['POST'])
def upload_models():
    try:
        models['scaler'] = joblib.load(request.files.get('scaler'))
        models['kmeans'] = joblib.load(request.files['kmeans'])
        models['pca'] = joblib.load(request.files['pca'])
        
        raw_df = pd.read_csv(request.files['dataset'], encoding="latin-1")
        models['df'] = raw_df
        
        df_c = raw_df.copy()
        df_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY', 
                   'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 
                   'CUSTOMERNAME', 'ORDERNUMBER', 'STATUS']
        df_c.drop(columns=[c for c in df_drop if c in df_c.columns], inplace=True)
        
        models['df_cleaned'] = df_c.select_dtypes(include=[np.number]).dropna()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/dataset_stats')
def dataset_stats():
    if models['df'] is None: return jsonify({"success": False})
    
    df = models['df']
    df_num = models['df_cleaned']
    
    # PCA 3D
    sc = StandardScaler()
    scaled = sc.fit_transform(df_num)
    pca_res = PCA(n_components=3).fit_transform(scaled)
    
    # WCSS
    wcss = []
    sample = df_num.sample(n=min(1000, len(df_num)))
    for i in range(1, 11):
        km = KMeans(n_clusters=i, n_init=10, random_state=42).fit(sample)
        wcss.append(float(km.inertia_))

    return jsonify({
        "success": True,
        "pca_3d": {"x": pca_res[:,0].tolist(), "y": pca_res[:,1].tolist(), "z": pca_res[:,2].tolist()},
        "wcss": wcss,
        "sales_month": df.groupby(pd.to_datetime(df['ORDERDATE']).dt.month)['SALES'].sum().tolist(),
        "countries": df['COUNTRY'].value_counts().index.tolist(),
        "country_counts": df['COUNTRY'].value_counts().tolist(),
        "table": df.head(12).fillna('').to_dict('records'),
        "corr": df_num.corr().values.tolist(),
        "corr_labels": df_num.columns.tolist()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)