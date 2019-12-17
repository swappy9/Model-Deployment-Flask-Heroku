
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import pickle

def SelectTopN(df, N):
    df = df.select_dtypes(include=['float64', 'int64'])
    X = df.drop(['SalePrice'], axis=1)
    X = X.fillna(X.mean(), inplace=True)
    y = df[['SalePrice']]
    
    bestfeatures = SelectKBest(score_func=chi2, k=N)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    
    return featureScores.nlargest(N,'Score')



def linear_model(X_train, y_train, X_test, y_test):
    clf1 = LinearRegression()
    linear_model = clf1.fit(X_train, y_train)
    linear_model_predictions = linear_model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, linear_model_predictions))
    pickle.dump(linear_model, open('model.pkl', 'wb'))
    
    return linear_model_predictions, RMSE


def get_prediction(arr, model):
    #Making the arrray 2D
    arr2D = [arr]
    return model.predict(arr2D)


def main():
    
    #Loading the dataset
    train = pd.read_csv('train.csv')
    #test = pd.read_csv('test.csv')
    print('\n***********************************')
    print('Train dataset shape: ', train.shape)
    #print('Test dataset shape: ', test.shape)
    print('Columns in training dataset: ', train.columns)
    #print('Columns in training dataset: ', train.dtypes)
    
    
    #Calulating higher explainability variables
    print('\n*********************************************\n')
    FeatureScores = SelectTopN(train, 5)
    print('Top 5 features in the training dataset with higher explainability: \n',FeatureScores)
    
    
    #Creating subset of data
    print('\n*********************************************\n')
    FeatureList = list(FeatureScores['Specs'])
    SP_train = list(train['SalePrice'])
    
    train = train[FeatureList]
    train['SalePrice'] = SP_train
    #test = test[FeatureList]
    print('Train dataset shape after selecting top variables: ',train.shape)
    #print('Test dataset shape after selecting top variables: ',test.shape)
    
    
    #Splitting the dataset in train and test 
    print('\n*********************************************\n')
    X_train, X_test, y_train, y_test = train_test_split(train.drop(['SalePrice'], axis=1), train[['SalePrice']].astype(int), 
                                                        test_size=0.2, random_state=0)
    
    print('Train and test dataset shape: ',X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    #Calling the Linear Regression model
    predictions, RMSE = linear_model(X_train, y_train, X_test, y_test)
    print('Model RMSE: ',RMSE)
    
    #Testing the dowanloaded model
    print(get_prediction([11622, 400, 800, 750, 300], pickle.load(open('model.pkl', 'rb'))))
    
if __name__ == '__main__':
    main()
