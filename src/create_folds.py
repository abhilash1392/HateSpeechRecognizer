#src/create_folds.py
# Importing the libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold

if __name__=="__main__":
    # Loading the dataset
    df = pd.read_csv('../data/labeled_data.csv')
    # Creating a column kfold and setting all it values as -1
    df['kfold']=-1
    # Shuffling the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    # Creating the kf 
    kf = StratifiedKFold(n_splits = 5)
    X = df.drop(['class'],axis=1)
    y = df['class'].values

    for f, (t_,v_) in enumerate(kf.split(X=X,y=y)):
        df.loc[v_,'kfold']=f 

    df.to_csv('../data/data_folds.csv',index=False)

