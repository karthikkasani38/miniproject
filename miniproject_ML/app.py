import numpy as np
import pickle
from flask import Flask, request, render_template

# creating flask app

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    values = [np.array(features)]
    prediction = model.predict(values)
    return render_template("index.html", prediction_text="The predicted crop is {}".format(prediction))


@app.route("/predict1", methods=["POST"])

def predict1():
    temperature = int(request.form['temperature'])
    humidity = int(request.form['humidity'])
    moisture = int(request.form['moisture'])
    n = int(request.form['N'])
    p = int(request.form['P'])
    k = int(request.form['K'])
    s=0
    c=0
    soil_type = str(request.form['Soil'])

    if(soil_type=='Black'):
        s=0
    elif(soil_type=='Clayey'):
        s=1
    elif(soil_type=='Loamy'):
        s=2
    elif(soil_type=='Red') :
        s=3
    else:
        s=4

    crop_type = str(request.form['Crop'])

    if(crop_type == 'Barley'):
        c=0
    elif(crop_type == 'Cotton'):
        c=1
    elif(crop_type == 'Ground Nuts'):
        c=2
    elif(crop_type == 'Maize'):
        c=3
    elif(crop_type == 'Millets'):
        c=4
    elif(crop_type == 'Oil seeds'):
        c=5
    elif(crop_type == 'Paddy'):
        c=6
    elif(crop_type == 'Pulses'):
        c=7
    elif(crop_type == 'Sugarcane'):
        c=8
    elif(crop_type == 'Tobacco'):
        c=9
    else:
        c=10
    prediction = model1.predict([[temperature, moisture, humidity, s, c, n, p, k]])
    return render_template("index.html", prediction_text_fertilizer="The predicted Fertilizer is {}".format(prediction))

"""
def predict1():
    features = [int(x) for x in request.form.values()]
    values = [np.array(features)]
    prediction = model1.predict(values)
    return render_template("index.html", prediction_text_fertilizer="The predicted Fertilizer is {}".format(prediction))"""


if __name__ == "__main__":
    app.run(debug=True)
home()
