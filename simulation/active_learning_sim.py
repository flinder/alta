import argparse
import itertools
import pandas as pd
import numpy as np
import random
import yaml
from scipy.stats import expon, beta

from collections import Counter
from multiprocessing import Pool

from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

## Suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier

# Internal imports
from helpers import DtmSelector

parser = argparse.ArgumentParser()
parser.add_argument("data", help="Valid choices are 'tweets' and 'wikipedia_hate_speech'", type=str)
parser.add_argument("--random", help="Random sampling.", dest='random', action='store_true')

args = parser.parse_args()

# Load config
CONFIG = '../config.yaml'

# load config
with open(CONFIG) as config_file:
    config = yaml.load(config_file)

text_feature_sets = list(itertools.product(*config['text_features'].values()))
text_feature_sets = ['../data/dtms/' + '_'.join([args.data]+[str(x) for x in tf]) + '_dtm.pkl' for tf in text_feature_sets]

#############################################################################
# INITIAL SAMPLING / DEV SET
#############################################################################

fname = config['data_sets'][args.data]['fname']
y_col = config['data_sets'][args.data]['y_col']
data = pd.read_csv("../data/%s" % fname, dtype={y_col: 'int'})

n_records = config['data_sets'][args.data]['n_records']
train, test, train_y, test_y = train_test_split(range(0, n_records), data[y_col], test_size=0.2, random_state=1988)

#############################################################################
# MODEL
#############################################################################

svm = SGDClassifier(penalty='elasticnet', loss='log', alpha=0.0001, 
                        l1_ratio=0.77, random_state=26661)

## Hyperparameter distributions for randomized search
parameters = {
	'clf__class_weight': ['balanced', None],
	'clf__alpha': expon(1e-4),
	'clf__l1_ratio': beta(4,1),
	'text__selector__fname': text_feature_sets,
}

## Model pipeline
pipeline = Pipeline([
		('text', Pipeline([
			('selector', DtmSelector()),
			('tfidf', TfidfTransformer())
		])),
		('clf', svm),
])

#############################################################################
# ACTIVE LEARNING SIMULATION
#############################################################################

runs = []
n_steps = 200
stepsize = n_records // n_steps

## Get initial labels
labeled_ids = set([])
init_pos = data.loc[data[y_col] == 1, ].sample(20).index
init_neg = data.loc[data[y_col] == 0, ].sample(20).index
labeled_ids.update(init_pos)
labeled_ids.update(init_neg)

for i in range(200):
	to_code = []

	labeled = data.ix[list(labeled_ids), :]
	y_true = labeled[y_col]

	unlabeled_ids = set(train) - set(labeled_ids)
	unlabeled = data.ix[list(unlabeled_ids), :]

	y_unlabeled_true = unlabeled[y_col]

	vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 2))

	X_train, X_test, y_train, y_test = train_test_split(list(labeled_ids), y_true, test_size=0.20, random_state=64)
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	X_pred = np.array(unlabeled_ids)
	X_dev = np.array(test)

	## Randomized hyperparameter search
	grid = RandomizedSearchCV(pipeline, parameters, n_iter=2, scoring=make_scorer(f1_score), n_jobs=2, random_state=1988)
	grid.fit(X_train, y_train)

	y_pred = grid.predict(X_dev)

	## Evaluate F1 on development set, save in dictionary
	f1 = f1_score(test_y, y_pred)
	support = len([y for y in y_true if y == 1])
	print("Batch %s - F_1: %.2f, Support: %d" % (i+1, f1, support))
	run = {'f1' : f1, 'support' : support, 'batch' : i}
	runs.append(run)

	if args.random == False:
		try:
			dist_to_hp = grid.decision_function(X_pred)

			dist_uc = list(zip(unlabeled_ids, dist_to_hp))
			dist_uc_pos = [d for d in dist_uc if d[1] >= 0]
			dist_uc_neg = [d for d in dist_uc if d[1] <= 0]

			pos_ids, _ = list(zip(*sorted(dist_uc_pos, key=lambda x: x[1])))
			neg_ids, _ = list(zip(*sorted(dist_uc_neg, key=lambda x: x[1])))
			
			to_code.extend(pos_ids[:int(stepsize/2)])
			to_code.extend(neg_ids[:int(stepsize/2)])
		except IndexError:
			print("INDEX ERROR: Not enough observations on each side of hyperplane.")
			dist_to_hp = grid.decision_function(X_pred)
			dist_to_hp = abs(dist_to_hp)
			dist_uc = list(zip(unlabeled_ids, dist_to_hp))
			dist_uc = sorted(dist_uc, key=lambda x: x[1])
			sorted_ids = list(zip(*dist_uc))[0]
			to_code.extend(sorted_ids[:n])
	else:
		to_code = random.sample(unlabeled_ids, stepsize)
	labeled_ids.update(to_code)

## Save runs from list of dictionaries to CSV
simulation_data = pd.DataFrame(runs)
if args.random:
	simulation_data.to_csv('../data/random_simulation_data.csv', index=False)
else:
	simulation_data.to_csv('../data/active_simulation_data.csv', index=False)
