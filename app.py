import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

models = {'df': None, 'df_cleaned': None}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    df = pd.read_csv(request.files['dataset'], encoding="latin-1")
    models['df'] = df

    df_c = df.copy()
    df_drop = ['ADDRESSLINE1','ADDRESSLINE2','POSTALCODE','CITY','TERRITORY',
               'PHONE','STATE','CONTACTFIRSTNAME','CONTACTLASTNAME',
               'CUSTOMERNAME','ORDERNUMBER','STATUS']

    df_c.drop(columns=[c for c in df_drop if c in df_c.columns], inplace=True)

    df_num = df_c.select_dtypes(include=[np.number]).dropna()
    models['df_cleaned'] = df_num

    return jsonify({"success": True})


@app.route('/graph', methods=['POST'])
def graph():

    t = request.json['type']

    df = models['df']
    df_num = models['df_cleaned']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_num)

    pca2 = PCA(n_components=2).fit_transform(scaled)
    pca3 = PCA(n_components=3).fit_transform(scaled)

    kmeans = KMeans(n_clusters=5, n_init=10)
    labels = kmeans.fit_predict(scaled)

    # ----------- GRAFICAS -----------

    if t == "timeline":
        data = df.groupby(pd.to_datetime(df['ORDERDATE']))['SALES'].sum()
        return jsonify({"x": data.index.astype(str).tolist(), "y": data.values.tolist()})

    elif t == "pca3d":
        return jsonify({
            "x": pca3[:,0].tolist(),
            "y": pca3[:,1].tolist(),
            "z": pca3[:,2].tolist(),
            "color": labels.tolist()
        })

    elif t == "clusters":
        return jsonify({
            "x": pca2[:,0].tolist(),
            "y": pca2[:,1].tolist(),
            "labels": labels.tolist()
        })

    elif t == "elbow":
        wcss=[]
        for i in range(1,11):
            km = KMeans(n_clusters=i,n_init=10).fit(scaled)
            wcss.append(float(km.inertia_))
        return jsonify({"x":list(range(1,11)),"y":wcss})

    elif t == "countries":
        data = df['COUNTRY'].value_counts()
        return jsonify({"x":data.index.tolist(),"y":data.values.tolist()})

    elif t == "product":
        data = df.groupby('PRODUCTLINE')['SALES'].sum()
        return jsonify({
            "x":data.index.tolist(),
            "y":data.values.tolist(),
            "color": list(range(len(data)))
        })

    elif t == "corr":
        return jsonify({
            "z":df_num.corr().values.tolist(),
            "labels":df_num.columns.tolist()
        })

    elif t == "scatter":
        sample = df_num.sample(n=min(300,len(df_num)))
        data = {col: sample[col].tolist() for col in sample.columns[:5]}
        return jsonify(data)

    

    elif t == "dataset":
        return jsonify(df.head(15).fillna("").to_dict("records"))

    return jsonify({})


if __name__ == '__main__':
    app.run(debug=True)