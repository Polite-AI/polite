
import urllib


from learning_helpers import parse_training_data, DenseTransformer, make_mlp, save_pipeline, load_pipeline, get_best_params

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import TransformerMixin, BaseEstimator



tasks = [ 'attack', 'aggression', 'toxicity' ]
tasks = [ 'attack' ]
for task in tasks:
    X, y = parse_training_data('models', task)

    label_type = 'ed'
    model_type = 'mlp'
    ngram_type = 'char'

    clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('to_dense', DenseTransformer()),
            ('clf', KerasClassifier(build_fn=make_mlp, output_dim = y.shape[1], verbose=False)),
        ])

    params = get_best_params('models', model_type, ngram_type, label_type)
    print(params)
    print('training', task)
    engine = clf.set_params(**params).fit(X,y)
    print('trained', task, 'saving')
    save_pipeline(clf, 'models', 'model-'+task)
    print('saved', task)
print ('done all')
