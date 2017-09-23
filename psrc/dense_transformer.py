from sklearn.base import TransformerMixin, BaseEstimator

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
