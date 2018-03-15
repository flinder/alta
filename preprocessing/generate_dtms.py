'''
Take all data inputs and transform create the feature matrices
'''

import yaml
import itertools
import pandas as pd
import os
import pickle

from text_processing import TextProcessor


if __name__ == "__main__":

    CONFIG = '../config.yaml'
    DATA_PATH = '../data'
   
    # load config
    with open(CONFIG) as config_file:
        config = yaml.load(config_file)

    # get feature sets
    features = config['text_features']
    ## These will be in the order as they appear in config.yaml this can only be 
    ## changed if also changing the order of arguments in
    ## TextProcessor.vectorize, a bit hacky, maybe implement with kwargs TODO.
    feature_sets = list(itertools.product(*features.values()))

    german_text_processor = TextProcessor('de')
    english_text_processor = TextProcessor('en')
    
    for data_set_name in config['data_sets']:
        print(f'Processing {data_set_name}')
        data_set = config['data_sets'][data_set_name]
        df = pd.read_csv(os.path.join(DATA_PATH, data_set['fname']))
        s = df.shape[0]
        df.dropna(inplace=True)
        n_dropped = s - df.shape[0]
        if n_dropped > 0:
            print(f'Dropped {n_dropped} rows due to missing values')
        #df = pd.read_csv('../data/sample.csv')
        text = df[data_set['text_col']]
        
        dtms = []
        for feature_set in feature_sets:
            feature_string = '_'.join([str(x) for x in feature_set])
            outfname = f'{data_set_name}_{feature_string}_dtm.pkl'
            outpath = os.path.join(DATA_PATH, 'dtms/', outfname)
            if data_set['language'] == 'en':
                out = english_text_processor.vectorize(text, *feature_set)
            else:
                out = german_text_processor.vectorize(text, *feature_set)
            with open(outpath, 'wb') as outfile:
                pickle.dump(out, outfile)
