import csv
import re
import logging
import sys
import itertools
import pandas as pd
import numpy as np
import time

sys.path.append('../../dissertation/dissdat/database/')
from db import make_session

logging.basicConfig(level=logging.INFO)

# Create sql connection
logging.info('Connecting to DB...')
_, engine = make_session()

# Get the data from db
query = ('SELECT cr_results.tweet_id,' 
         '       cr_results.annotation,' 
         '       cr_results.trust,'
         '       tweets.text '
         'FROM cr_results '
         'INNER JOIN tweets on tweets.id=cr_results.tweet_id')

logging.info('Getting the data...')
df = pd.read_sql(sql=query, con=engine)

df.replace(to_replace=['relevant', 'irrelevant', 'None'], 
           value=[1,0,np.nan], inplace=True)

# Select only judgements from trusted contributors
df = df[['tweet_id', 'annotation', 'text']].loc[df['trust'] > 0.8]

# Aggregate to one judgement per tweet
def f(x):
     return pd.Series({'annotation': x['annotation'].mean(),
                       'text': x['text'].iloc[0],
                       'tweet_id': x['tweet_id'].iloc[0]})

logging.info('Aggregating...')
df = df[['annotation', 'text', 'tweet_id']].groupby('tweet_id').apply(f)
df = df[['annotation', 'text']]

# Make annotations binary
df.loc[df['annotation'] >= 0.5, 'annotation'] = 1
df.loc[df['annotation'] < 0.5, 'annotation'] = 0

df.to_csv('../data/annotated_german_refugee_tweets.csv')
