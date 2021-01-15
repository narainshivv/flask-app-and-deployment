from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
import pandas as pd

application = Flask(__name__)

model = pickle.load(open('modelForPrediction.pkl', 'rb'))

@application.route('/')
def hello_world():
   return render_template('index.html')


@application.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features3 = [float(x) for x in request.form.values()]
    final3 = [np.array(int_features3)]
    print(int_features3)
    print(final3)
    prediction = model.predict(final3)

    if prediction == 0:
        return render_template('index.html', pred='Survived')
    else:
        return render_template('index.html', pred='Did Not Survive')


if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
