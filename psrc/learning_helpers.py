import pandas as pd
import os, sys
import keras
import keras.wrappers
import keras.wrappers.scikit_learn
import copy
import joblib
import json
from sklearn.base import TransformerMixin, BaseEstimator
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2

def get_best_params(data_dir, model_type, ngram_type, label_type):
    cv_results = pd.read_csv(os.path.join(data_dir, 'cv_results.csv'))
    query = "model_type == '%s' and ngram_type == '%s' and label_type == '%s'" % (model_type, ngram_type, label_type)
    params = cv_results.query(query)['best_params'].iloc[0]
    return(json.loads(params))

def parse_training_data(data_dir, task):

    """
    Computes labels from annotations and aligns comments and labels for training
    """

    COMMENTS_FILE = "%s_annotated_comments.tsv" % task
    LABELS_FILE = "%s_annotations.tsv" % task

    comments = pd.read_csv(os.path.join(data_dir, COMMENTS_FILE), sep = '\t', index_col = 0)
    # remove special newline and tab tokens

    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))


    annotations = pd.read_csv(os.path.join(data_dir, LABELS_FILE),  sep = '\t', index_col = 0)
    labels = empirical_dist(annotations[task])

    X = comments.sort_index()['comment'].values
    y = labels.sort_index().values

    assert(X.shape[0] == y.shape[0])
    return X, y

def empirical_dist(l, w = 0.0, index = None):

    """
    Compute empirical distribution over all classes
    using all labels with the same rev_id
    """
    if not index:
        index = sorted(list(set(l.dropna().values)))

    data = {}
    for k, g in l.groupby(l.index):
        data[k] = g.value_counts().reindex(index).fillna(0) + w

    labels = pd.DataFrame(data).T
    labels = labels.fillna(0)
    labels = labels.div(labels.sum(axis=1), axis=0)
    return labels


def save_sklearn_pipeline(pipeline, directory, name):
    joblib.dump(pipeline, os.path.join(directory, '%s_pipeline.pkl' % name))

def load_sklearn_pipeline(directory, name):
    return joblib.load(os.path.join(directory, '%s_pipeline.pkl' % name))

def save_keras_pipeline(pipeline, directory, name):
    # save classifier
    clf = pipeline.steps[-1][1]
    keras_model = clf.model
    keras_model.save(os.path.join(directory, '%s_clf.h5' % name))
    # save pipeline, without model
    clf.model = None
    joblib.dump(pipeline, os.path.join(directory, '%s_extractor.pkl' % name))
    clf.model = keras_model

def load_keras_pipeline(directory, name):
    model = keras.models.load_model(os.path.join(directory, '%s_clf.h5' % name))
    pipeline = joblib.load(os.path.join(directory, '%s_extractor.pkl' % name))
    pipeline.steps[-1][1].model = model
    return pipeline

def save_pipeline(pipeline, directory, name):

    clf = pipeline.steps[-1][1]
    is_keras = type(clf) == keras.wrappers.scikit_learn.KerasClassifier
    if is_keras and hasattr(clf, 'model'):
        save_keras_pipeline(pipeline, directory, name)
    else:
        save_sklearn_pipepine(pipeline, directory, name)

def load_pipeline(directory, name):
    sklearn_file = os.path.join(directory, '%s_pipeline.pkl' % name)
    keras_clf_file = os.path.join(directory, '%s_clf.h5' % name)
    keras_extractor_file = os.path.join(directory, '%s_extractor.pkl' % name)

    if os.path.isfile(sklearn_file):
        return load_sklearn_pipeline(directory, name)
    elif os.path.isfile(keras_clf_file) and os.path.isfile(keras_extractor_file):
        return load_keras_pipeline(directory, name)
    else:
        print('Pipeline not saved')
        return None

def make_mlp(hidden_layer_sizes = [], output_dim = None, input_dim = None, alpha = 0.0001, softmax = True):
    architecture = [input_dim] + list(hidden_layer_sizes) + [output_dim]
    # create model
    model = Sequential()
    layers = list(zip(architecture, architecture[1:]))

    for i, o in layers[:-1]:
        model.add(Dense(input_dim=i,output_dim=o, init='normal', W_regularizer = l2(alpha)))
        model.add(Activation("relu"))

    i, o = layers[-1]
    model.add(Dense(input_dim=i,output_dim=o, init='normal', W_regularizer = l2(alpha)))
    if softmax:
        model.add(Activation("softmax"))

    # Compile model
    if softmax:
        model.compile(loss='kullback_leibler_divergence', optimizer='adam', metrics = ['mse'])
    else:
        model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
    return model



class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None):
        return X.todense()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=False):
        return {}
