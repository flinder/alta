import pickle
import os

from sklearn.base import BaseEstimator, TransformerMixin


class DtmSelector(BaseEstimator, TransformerMixin):
    '''
    Selector class to load dtm with specific feature set.

    Args:
    feature_set: List of features (see 'config.yaml' for details)
    data_path: location of dtms

    Returns:
    scipy.sparse.csr_matrix

    '''

    def __init__(self, fname):
            self.fname = fname

    def fit(self, x, y=None):
            return self

    def transform(self, indexes):
        with open(self.fname, 'rb') as infile:
            dtm = pickle.load(infile)
        try:
            return(dtm[indexes, :] )
        except:
            print(type(indexes))
            raise