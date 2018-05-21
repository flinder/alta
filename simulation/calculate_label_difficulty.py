import glob
import itertools
import pandas as pd
import numpy as np
import os
import yaml
from scipy.stats import expon, beta, rv_continuous

from collections import Counter

## Suppress sklearn warnings
def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
														 accuracy_score, make_scorer)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

# Internal imports
from helpers import DtmSelector

dataset = 'tweets'

# Load config
CONFIG = '../config.yaml'

# load config
with open(CONFIG) as config_file:
	config = yaml.load(config_file)

text_feature_sets = list(
											itertools.product(
													config["text_features"]["tfidf"],
													config["text_features"]["stem"],
													config["text_features"]["token_type"]
												)
											)
text_feature_sets = ['../data/dtms/' + '_'.join([dataset] +
										[str(x) for x in tf]) + '_dtm.pkl'
										for tf in text_feature_sets]

fname = config['data_sets'][dataset]['fname']
y_col = config['data_sets'][dataset]['y_col']
data = pd.read_csv("../data/%s" % fname, dtype={y_col: 'int'})

all_idxs = data.index

result_fn = '../data/difficulty_simulation_%s.csv' % dataset
if os.path.isfile(result_fn):
	all_results = pd.read_csv(result_fn)
all_results = pd.DataFrame()
for i in range(1000):
	print("Starting iteration %d" % i)
	train, test, train_y, test_y = train_test_split(
																		all_idxs,
																		data.ix[all_idxs, y_col],
																		test_size=0.4,
																	)

	svm = LinearSVC(penalty='l2', class_weight='balanced', random_state=1988)
	clf = CalibratedClassifierCV(svm)

	parameters = {
		'clf__base_estimator__C': expon(50),
		'clf__base_estimator__class_weight': ['balanced', None],
		'text__selector__fname': text_feature_sets,
	}

	## Model pipeline
	pipeline = Pipeline([
			('text', Pipeline([
				('selector', DtmSelector(fname=text_feature_sets[0])),
				('tfidf', TfidfTransformer())
			])),
			('clf', clf),
	])

	## Grid hyperparameter search
	grid = RandomizedSearchCV(
					pipeline,
					parameters,
					n_iter=5,
					scoring=make_scorer(f1_score),
					n_jobs=1,
					random_state=1988
				)

	grid.fit(train, train_y)
	y_proba = clf.predict_proba(test)
	results = {'id' : test, 'p' : y_proba}
	results = pd.DataFrame(results)
	results['i'] = i
	all_results = all_results.append(results)
	all_results.to_csv(result_fn, index=False)