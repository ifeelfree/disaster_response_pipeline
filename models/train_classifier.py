#!/usr/bin/env python
# coding: utf-8

 


# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

 


import numpy as np
import pandas as pd
import sys
import os
import re
from sqlalchemy import create_engine
import pickle
from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

def tokenize(text, url_place_holder_string="urlplaceholder"):
    """
    This function is used to convert the text into a list of words
    
    :param text: input texts
    :param url_place_holder_string: this word will replace the urls texts
    """
     # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


def train_model(database_name, model_file_name):
    """
    This function is used to train a classification model

    :param database_name data base name
    :param model_file_name model file name

    """

    # step 1: get the data
    table_name = "disaster_response_table"
    engine = create_engine('sqlite:///'+database_name)
    df = pd.read_sql_table(table_name, engine)

    # step 2: data cleaning
    col_to_removed = []
    for key, value in df.describe().items():
        if value['max']<1:
            col_to_removed.append(key)
    cleaned_df = df.drop(col_to_removed, axis=1)

    X = df['message']
    Y = df.iloc[:,4:]
    y=Y.astype('bool')

    # step 3: perform machine learning taining
    pipeline_basic = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count', CountVectorizer(tokenizer=tokenize)),
                ('tfid', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipeline_fitted = pipeline_basic.fit(X_train, y_train)

    # step 4: perform testing
    y_prediction_test = pipeline_fitted.predict(X_test)
    print(classification_report(y_test.values.astype('bool'), 
                      y_prediction_test.astype('bool'), target_names=Y.columns.values))


    # step 5: save the model
   # model_file_name ='classifier.pkl'
    with open(model_file_name, "wb") as f:
        pickle.dump(pipeline_fitted, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate classification model')
    parser.add_argument('--database', metavar='path', required=False,
                        help='path to .db database')

    parser.add_argument('--model', metavar='path', required=False,
                        help='path to category csv file')
    
    args = parser.parse_args()
    db_file_name = args.database
    model_file_name = args.model 


    if model_file_name is None:
        model_file_name = "classifier.pkl"
    
    if db_file_name is None:
        db_file_name = "../data/DisasterResponse.db"

    assert os.path.exists(db_file_name)

    train_model(db_file_name, model_file_name) 









 


   