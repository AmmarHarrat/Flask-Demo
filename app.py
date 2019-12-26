import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    list_str = []
    list_str.append(request.form['name'])
    array =  np.asarray(list_str)
    cat_data = pd.read_csv('demofile2.csv', sep=',|\t', engine='python')
    prediction = model.predict(array)
    predicted  = cat_data.cat[cat_data['code'] == prediction[0]].item()
    
    return render_template('index.html', prediction_text=predicted)


if __name__ == "__main__":
    app.run(debug=False)

