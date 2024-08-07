import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from kedro.pipeline import node, Pipeline


file_path = "/vscode/practices/data/Youtube01-Psy.csv "

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data[["CONTENT", "CLASS"]]
    data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam Comment"})

    x = np.array(data["CONTENT"])
    y = np.array(data["CLASS"])

    cv = CountVectorizer()
    x = cv.fit_transform(x) 

    return x, y, cv

def train_model(x, y, test_size=0.2, random_state=42 ):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = test_size, random_state=random_state)
    model = BernoulliNB()
    model.fit(xtrain, ytrain)

    return model, xtrain, xtest, ytrain, ytest


def evaluate_model(model, xtest, ytest):
    score = model.score(xtest, ytest)
    return score


def predict_sample(model, cv, sample):
    data = cv.transform([sample]).toarray()
    prediction = model.predict(data)[0]
    return prediction


                 
