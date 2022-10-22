from flask import Flask, flash, request, redirect, url_for, render_template
import pickle
# from pushbullet import PushBullet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Loading Models
heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
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
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        pred = heart_model.predict(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run(debug=True)
