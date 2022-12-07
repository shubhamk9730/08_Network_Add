from flask import Flask, render_template, request
import pickle as pkl
import numpy as np
import pandas as pd

model = pkl.load(open('RF_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pre', methods=['POST'])
def pred():
    Gender = request.form['Gender']
    Age = request.form['Age']
    EstimatedSalary = request.form['EstimatedSalary']

    project_data = {"Gender":{'male':1, 'female':0}, 'columns' :['Gender', 'Age', 'EstimatedSalary']}

    arr = np.array([[project_data['Gender'][Gender], Age, EstimatedSalary]])
    arr2 = np.array(arr, dtype=float)
    predi = str(model.predict(arr2)[0])

    if predi=='0':
        return 'not purchased'
    if predi == '1':
        return 'purchased'

if __name__=='__main__':
    app.run(host='0.0.0.0', port= 2121, debug=True)

