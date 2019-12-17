#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[4]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[5]:


@app.route('/')
def home():
    return render_template('index.html')


# In[6]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text='Predicted House Price is $ {}'.format(output))


# In[7]:


if __name__ == "__main__":
    app.run(debug=True)

