#Importing the required python libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initializing the application name [here, the name is app]
app = Flask(__name__)

#Loading the model created in model.py
model = pickle.load(open('model.pkl', 'rb'))

#Starting the app by rendering the index.html page
@app.route('/')
def home():
    return render_template('index.html')

#Calling the prediction function using the POST method
@app.route('/predict',methods=['POST'])
def predict():
    
    #Reading the inputs from the index.html page to pass it while calling the model
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0][0], 2)
    
    return render_template('index.html', prediction_text='Predicted House Price is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
