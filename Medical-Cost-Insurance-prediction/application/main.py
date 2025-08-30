import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('glass_identification_classification.pkl','rb'))


@app.route('/classify-glass',methods=['POST'])
def classify():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
     # Convert input data to DataFrame with consistent feature names
    df_input = pd.DataFrame(data['features'], columns=['RI:refractive index', 'Na:Sodium', 'Al:Aluminum', 'Si:Silicon', 'Ca:Calcium'])
     # Make prediction
    prediction = model.predict(df_input)
    
    return jsonify({'prediction': prediction.tolist()})



if __name__ == '__main__':
    app.run(debug=True)

