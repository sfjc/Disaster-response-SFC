# import packages
import sys

# import libraries
import pandas as pd
import numpy as np
import random
import requests
import json   

import sqlalchemy

print("test")

#Import tokenization things
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib




nltk.download(['punkt', 'wordnet'])

import warnings
warnings.filterwarnings('ignore')


# function to load data from database



def load_data(path='data/response.db',tablename="data/dfall"):
    
    '''This loads the previously processed data for analysis'''

    # read in file

    engine = sqlalchemy.create_engine('sqlite:///'+path).connect()

    df = pd.read_sql_table(tablename, engine)
    X = df['message']
    y = df.drop(['message','id','original','genre','related'], axis = 1)
    cat_names = y.columns

    return X, y

# function to create tokens

def tokenize(text):

    '''This creates tokens from the data'''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    
# function to build the model


def build_model():

    '''This creates a predictive model from the processed and tokenized data'''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  RandomForestClassifier())
    ]) 
    
    #        'clf__estimator__min_samples_leaf' : [1,2,3],
#        'clf__estimator__max_depth' : [2,4],
#        'tfidf__use_idf':[True, False],


    parameters = {'clf__min_samples_split': [2,4],
        'clf__n_estimators': [100,200]
                    }

    cv=GridSearchCV(pipeline,param_grid=parameters, verbose=3)
                      
#    print("training")
#    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=23)
#    cv.fit(X_train, y_train)
#    new_mod = cv.best_estimator_

    return cv


# Function to test the model



def modeltest(model, X_test, y_test):

    '''This checks model performance '''

    y_pred = model.predict(X_test)
    
#    display_results(model, y_test, y_pred)
    
    for i, colnum in enumerate(y_test):
        print(colnum)
        print(classification_report(y_test[colnum],y_pred[:, i]))

        
########

#New pipeline trial

#pipeline_new = Pipeline([
#        ('vect', CountVectorizer()),
#        ('tfidf', TfidfTransformer()),
#        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
#])




#pipeline_new.get_params()

#parameters_new = {
#        'clf__estimator__min_samples_leaf' : [1,2,3],
#        'clf__estimator__max_depth' : [2,4],
#        'tfidf__use_idf':[True, False],
#    }



#cv_new = GridSearchCV(pipeline_new, param_grid = parameters_new, verbose= 3)

#cv_new.fit(x_train, y_train)


############

# Function to export the model

def export_model(model):

    '''This 'pickles' the Python object hierarchy, creating a .pkl file  '''

    # Export model as a pickle file
    pickle.dump(model, open('classifier.pkl', 'wb'))


# Function to run the pipeline


def run_pipeline(data_file):

    '''Main function calling other functions. The pipeline.  '''

    print("loading")
    X,y= load_data(data_file)  # run ETL pipeline 
    y=y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

    print("building")
    cv = build_model()  # build model pipeline
    print("training")
    cv.fit(X_train, y_train)
    
    print("testing")
    modeltest(cv, X_test, y_test) #test model
    print("exporting")
    export_model(cv)  # save model
    print("done")

if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline




