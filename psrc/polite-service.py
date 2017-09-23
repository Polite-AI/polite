import pandas as pd
import urllib
import os, sys
import json

from learning_helpers import load_pipeline

from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from keras.regularizers import l2

from flask import Flask, request, abort


def check(input):
        return({
            "aggression" : 1, # aggression.predict(input)[0],
            "attack" : 0, #attack.predict(input)[0],
            "toxicity" : 1 #toxicity.predict(input)[0]
        })


#    return({
#        "aggression" : aggression.predict(input)[0],
#        "attack" : attack.predict(input)[0],
#        "toxicity" : toxicity.predict(input)[0]
#    })


#aggression = load_pipeline('models', 'model-new-aggression')
#attack = load_pipeline('models', 'model-new-attack')
#toxicity = load_pipeline('models', 'model-new-toxicity')

app = Flask(__name__)


@app.route('/api/1.0/classify', methods=['POST'])
def classify():
    if not request.json or not 'text' in request.json:
        abort(400)
    else:
        response = {
            'request' : { 'text' : request.json['text'] },
            'results' : check(request.json['text'])
        }
        return(json.dumps(response))

app.run()
