from flask import Flask, flash, request, redirect, url_for, render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Loading Models
heart_model = pickle.load(open('models/heart_model.dat', "rb"))


app = Flask(__name__)
########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')

@app.route('/featurespage')
def features():
    return render_template('featurespage.html')

@app.route('/showtable')
def showtable():
    return render_template('showtable.html')  

@app.route('/correlation')
def correlation():
    return render_template('correlation.html')

@app.route('/age_analysis')
def age_analysis():
    return render_template('age_analysis.html') 

@app.route('/gender_analysis')
def gender_analysis():
    return render_template('gender_analysis.html')  

@app.route('/cp_analysis')
def cp_analysis():
    return render_template('cp_analysis.html') 

@app.route('/thal_analysis')
def thal_analysis():
    return render_template('thal_analysis.html')    

@app.route('/googleform')
def googleform():
    return render_template('googleform.html')            

########################### Result Functions ########################################

@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        gender = request.form['gender']
        ca = request.form['nmv']
        cp = request.form['tcp']
        exang = request.form['eia']
        thal = request.form['thal']
        oldpeak = request.form['op']
        thalach = request.form['mhra']
        age = request.form['age']
        pred = heart_model.predict(np.array([ca, cp, exang, thal, oldpeak, thalach, age]).reshape(1, -1))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


if __name__ == '__main__':
    app.run(debug=True)
