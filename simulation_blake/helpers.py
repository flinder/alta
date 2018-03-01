import pickle

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

    def __init__(self, fname, data_path = '../data/dtms/'):
            self.data_path = data_path
            self.fname = f'../data/dtms/{fname}'

    def fit(self, x, y=None):
            return self

    def transform(self, data_dict):
        with open(self.fname, 'rb') as infile:
            return pickle.load(infile)
