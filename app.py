import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = tf.keras.models.load_model(
    'DBPredictorModel.h5', custom_objects=None, compile=True, options=None)
scFeatures = pickle.load(open('scFeatures_DBassignment.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    preg = float(request.form['preg'])
    plas = float(request.form['plas'])
    pres = float(request.form['pres'])
    skin = float(request.form['skin'])
    test = float(request.form['test'])
    mass = float(request.form['mass'])
    pedi = float(request.form['pedi'])
    age = float(request.form['age'])

    inputvar = np.array([[preg, plas, pres, skin, test, mass, pedi, age]])
    inputvarScaled = scFeatures.transform(inputvar)
    if model.predict(inputvarScaled)[0][0] > 0.5:
        prediction = 'Diabetes'
    else:
        prediction = 'no Diabetes'

    return render_template('index.html', prediction_text='Whether diabetes or not: The patient has {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)