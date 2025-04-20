import numpy as np 
import pandas as pd
import pickle
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

#import ridege regressor and standard scaler 
ridge_model = pickle.load(open('C:/Users/IMMANUEL RAJ KUMAR/Desktop/New folder (2)/myenv/machine_learning/project1/models/ridge.pkl','rb'))
standard_scalar = pickle.load(open('C:/Users/IMMANUEL RAJ KUMAR/Desktop/New folder (2)/myenv/machine_learning/project1/models/scaler.pkl','rb')) 

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict_datapoint():
    if request.method =='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes= float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        #apply scaling
        new_data = standard_scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data)
        return render_template('home.html',results = result[0])
    
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host = '0.0.0.0')