import pandas as pd
import numpy as np
import random

data = pd.read_csv('../data/raw_data/breitbart_data_full.csv')
neg = data.loc[data.muslim_identity == 0,]
pos = data.loc[data.muslim_identity == 1,]
data = neg.sample(24000 - len(pos), random_state=55433)
data = data.append(pos)
inds = np.arange(len(data))
random.shuffle(inds)
data = data.ix[inds]
data.to_csv('../data/breitbart_data.csv', index=False)