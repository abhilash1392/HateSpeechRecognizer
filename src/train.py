# Importing the libraries
import numpy as np 
import pandas as pd 
import textCleaning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
import joblib
import argparse
from  sklearn.feature_extraction.text import CountVectorizer

# Building the run function
def run(fold):
    # Importing the dataset
    df = pd.read_csv('../data/data_folds.csv')

    # Splitting the data using fold
    df_train  = df[df['kfold']!=fold].reset_index(drop=True)
    df_valid  = df[df['kfold']==fold].reset_index(drop=True)
    # creating X_train and X_valid
    X_train = df_train['tweet']
    X_valid = df_valid['tweet']
    y_train = df_train['class']
    y_valid = df_valid['class']
    # concating the whole data
    X_all = pd.concat([X_train,X_valid])
    # creating vectorizer
    vct = CountVectorizer(analyzer=textCleaning.clean_text)
    vct.fit(X_all)
    X_train_transformed = vct.transform(X_train)
    X_valid_transformed = vct.transform(X_valid) 

    # Using Logistic Regression model it is a great model for NLP
    lr = LogisticRegression(max_iter = 1000000)
    lr.fit(X_train_transformed,y_train)
    y_pred = lr.predict(X_valid_transformed)

    # printing the model accuracy
    print('Fold: {} Accuracy:{:.3f}'.format(fold,metrics.accuracy_score(y_valid,y_pred)))
    # Saving the model
    joblib.dump(lr,f'../models/dt_{fold}.bin')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',type=int)
    args = parser.parse_args()
    run(fold = args.fold)
