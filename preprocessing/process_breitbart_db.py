import pandas as pd

data = pd.read_csv('../data/raw_data/breitbart_data_full.csv')
data = data.sample(24000, random_state=55433)
data.to_csv('../data/breitbart_data.csv', index=False)