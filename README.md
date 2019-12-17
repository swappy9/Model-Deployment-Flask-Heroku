# Basic-Heroku-Deployment

The repository is created to showcase the steps involved in the basic deployment of machine learning models using the Heroku and flask.

## Overall flow

1. Build the Machine Learning model
2. Create a WebApp using Flask
3. Create a repo in Github [Use to deploy with Herolu]
4. Create an app in free Heroku account
5. Deploy the model in heroku app

## Building ML Model to deploy

Used Kaggle Houe Price prediction dataset to create a simple machine learning linear regression model to work.
Kaggle Link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

train.csv: Training dataset with 81 columns
test.csv: Test dataset with 80 columns (Not used in the implementation)
Model.py: Script building and saving the linear model in pickle file

## Creating WebApp

Created a simple HTML web page which takes the input using forms [available under templates as index.html]
Once the HTML page created, a Flask app - app.py takes the input from the browser (index.html) and calls the model to get the prediction and shows the results on the index page

## Create Github Repository

Upload all the required files to the repo [Only need model.pkl, index.html, app.py]

## Create an Heroku app

Create an app in Heroku free account, connect the above GitHub repo to the application.
NOTE: Before connecting the repo to the heroku, need requirements.txt and Procfile
      requirements.txt contains all the python dependencies with versions we used while creating the model or in the app.py
      Procfile contains the commands that are executed by the heroku app on startup with the application name
      

## Deploy the model

Once the app is deployed successfuly, anyone can use the provided URL to access the model.
URL: https://house-price-prediction-api.herokuapp.com/

