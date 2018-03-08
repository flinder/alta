import pickle
import os

from sklearn.base import BaseEstimator, TransformerMixin


class DtmSelector(BaseEstimator, TransformerMixin):
    '''
    Selector class to load dtm with specific feature set.

    Args:
    fname: Name of binary file containing feature matrix
    data_path: location of dtms

    Returns:
    scipy.sparse.csr_matrix

    '''

    def __init__(self, feature_set, data_path = '../data/dtms/'):
            fname = '_'.join([str(x) for x in feature_set]) + '_dtm.pkl'
            self.fname = os.path.join(data_path, fname)

    def fit(self, x, y=None):
            return self

    def transform(self, data_dict):
        with open(self.fname, 'rb') as infile:
            return pickle.load(infile)
