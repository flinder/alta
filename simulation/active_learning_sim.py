import argparse
import glob
import itertools
import pandas as pd
import numpy as np
import os
import pickle
import random
import yaml
from scipy.stats import expon, beta, rv_continuous

from collections import Counter
from functools import reduce
from operator import or_

def merge(*dicts):
		return({ k: reduce(lambda d, x: x.get(k, d), dicts, None)
							for k in reduce(or_, map(lambda x: x.keys(), dicts), set()) })

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

# Internal imports
from helpers import DtmSelector

parser = argparse.ArgumentParser()

parser.add_argument("data",
	help="Valid choices are 'tweets' and 'wikipedia_hate_speech' 'breitbart'",
	type=str
)
parser.add_argument("--balance",
	help="Float, proportion of positive observations",
	type=float
)
parser.add_argument("--iter",
	help="Integer, the current iteration.",
	type=int
)
parser.add_argument("--icr",
	help="Float, probability of correctly labeling an observation.",
	type=float
)
parser.add_argument("--mode",
	help="'random' for random sampling or 'active' for active learning.",
	type=str,
)

args = parser.parse_args()

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
text_feature_sets = ['../data/dtms/' + '_'.join([args.data] +
										[str(x) for x in tf]) + '_dtm.pkl'
										for tf in text_feature_sets]

#############################################################################
# CLASS BALANCE
#############################################################################

def icr_distort(ys, p):
	new_y = []
	for y in ys:
		new_y.append(y if random.random() < p else abs(y-1))
	return new_y

def balance_data(dat, balance):

	# Get index of negative and postive rows
	p_max = len(dat.loc[dat[y_col] == 1, ])
	n_max = len(dat.loc[dat[y_col] == 0, ])

	# Get the balance in the data
	original_balance = p_max / (n_max + p_max)

	# Pick the data
	if balance >= original_balance:
		# Use all positives and sample negatives
		n_neg = int(p_max * (1-balance) // balance)
		neg_idx = dat[dat[y_col] == 0].sample(n_neg).index
		pos_idx = dat[dat[y_col] == 1].index
	else:
		# Use all negatives and sample positives
		n_pos = int(balance * n_max // (1 - balance))
		neg_idx = dat[dat[y_col] == 0].index
		pos_idx = dat[dat[y_col] == 1].sample(n_pos).index

	return neg_idx.append(pos_idx)

#############################################################################
# SAVING SIMULATION DATA
#############################################################################

# save file to appropriate filename
rand = args.mode
if args.balance:
	if args.icr:
		fn = '../data/runs/%s/%s/%s_simulation_data_%s_icr_%s.csv' % (args.data, str(args.iter), rand, str(args.balance), str(args.icr))
	else:
		fn = '../data/runs/%s/%s/%s_simulation_data_%s.csv' % (args.data, str(args.iter), rand, str(args.balance))
else:
	if args.icr:
		fn = '../data/runs/%s/%s/%s_simulation_data_icr_%s.csv' % (args.data, str(args.iter), rand, str(args.icr))
	else:
		fn = '../data/runs/%s/%s/%s_simulation_data.csv' % (args.data, str(args.iter), rand)

if os.path.isfile(fn):
	print("Simulation already completed: %s" % fn)
	quit()

#############################################################################
# INITIAL SAMPLING / DEV SET
#############################################################################

fname = config['data_sets'][args.data]['fname']
y_col = config['data_sets'][args.data]['y_col']
data = pd.read_csv("../data/%s" % fname, dtype={y_col: 'int'})

if config['data_sets'][args.data]['n_cap'] is not None:
	data = data.sample(config['data_sets'][args.data]['n_cap'])
if args.balance is not None:
	all_idxs = balance_data(data, args.balance)
	print("Class balance: %.2f" % args.balance)
	n_records = len(all_idxs)
else:
	all_idxs = data.index
	n_records = len(data)

train, test, train_y, test_y = train_test_split(
																	all_idxs,
																	data.ix[all_idxs, y_col],
																	test_size=0.2,
																	random_state=1988
																)

if args.icr:
	train_y = icr_distort(train_y, args.icr)
	data.ix[train, y_col + '_icr'] = train_y

class_counter = Counter(train_y)
print("Positive observations: %d" % class_counter[1])
print("Negative observations: %d\n" % class_counter[0])

#############################################################################
# MODEL
#############################################################################

# Transformation to ensure alpha and C are equivalent for SGD/GD
class alpha(rv_continuous):
	def _pdf(self, x):
		val = 1. / (np.random.exponential(50) * (n_records * .8))
		return val

if config['data_sets'][args.data]['sgd'] == True:
	svm = SGDClassifier(loss="hinge", random_state=1988)
	parameters = {
		'clf__alpha': alpha(),
		'clf__class_weight': ['balanced', None],
		'text__selector__fname': text_feature_sets,
	}
else:
	svm = LinearSVC(penalty='l2', class_weight='balanced', random_state=1988)
	parameters = {
		'clf__C': expon(50),
		'clf__class_weight': ['balanced', None],
		'text__selector__fname': text_feature_sets,
	}

## Model pipeline
pipeline = Pipeline([
		('text', Pipeline([
			('selector', DtmSelector(fname=text_feature_sets[0])),
			('tfidf', TfidfTransformer())
		])),
		('clf', svm),
])

#############################################################################
# ACTIVE LEARNING SIMULATION
#############################################################################

## Batch size of 20
## Start with random sample and handle exception

runs = []
stepsize = 20
max_records = min(n_records // 5, 5000)
n_steps = max_records // stepsize
if n_steps < 200:
	n_steps = min(n_records // stepsize, 200)

## Get initial labels
labeled_ids = set(random.sample(list(train), 20))

print('beginning steps')
for i in range(n_steps):

	to_code = []

	labeled = data.ix[list(labeled_ids), :]
	if args.icr:
		y_true = labeled[y_col + "_icr"]
	else:
		y_true = labeled[y_col]

	unlabeled_ids = list(set(train) - set(labeled_ids))
	unlabeled = data.ix[unlabeled_ids, :]

	y_unlabeled_true = unlabeled[y_col]

	X_train, X_test, y_train, y_test = train_test_split(
																			list(labeled_ids),
																			y_true,
																			test_size=0.20,
																			random_state=64
																		)
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	X_pred = np.array(unlabeled_ids)
	X_dev = np.array(test)

	n_jobs = config['data_sets'][args.data]['n_jobs']
	## Randomized hyperparameter search
	grid = RandomizedSearchCV(
					pipeline,
					parameters,
					n_iter=20,
					scoring=make_scorer(f1_score),
					n_jobs=n_jobs,
					random_state=1988
				)
	try:
		grid.fit(X_train, y_train)
	except ValueError:
		run = {'support' : 0, 'batch' : i}
		runs.append(run)
		labeled_ids.update(set(random.sample(unlabeled_ids, 20)))
		continue

	y_pred = grid.predict(X_dev)

	## Evaluate F1 on development set, save in dictionary
	f1 = f1_score(test_y, y_pred)
	p = precision_score(test_y, y_pred)
	r = recall_score(test_y, y_pred)
	n_pos = np.sum(y_pred)

	support = len([y for y in y_true if y == 1])
	out = "%d/%d - F: %.2f, P: %.2f, R: %.2f, n_pos_pred: %d, pos_sup: %d, sup: %d" % (i+1, n_steps+1, f1, p, r, n_pos, support, (i+1) * stepsize)
	print(out)
	run = {	'f1' : f1,
					'p' : p,
					'r' : r,
					'n_pos_pred' : n_pos,
					'support' :
					support, 'batch' : i
				}
	run = merge(run, grid.best_params_)
	runs.append(run)

	if ((i+1) * stepsize) % 500 == 0:
		## Save runs from list of dictionaries to CSV
		simulation_data = pd.DataFrame(runs)
		simulation_data.to_csv(fn, index=False)

	if args.mode == 'active':
		try:
			dist_to_hp = grid.decision_function(X_pred)
		except ValueError:
			print("All positive documents have been labeled. Breaking out of simulation")
			break

		dist_uc = list(zip(unlabeled_ids, dist_to_hp))
		dist_uc_pos = [d for d in dist_uc if d[1] >= 0]
		dist_uc_neg = [d for d in dist_uc if d[1] <= 0]
		if len(dist_uc_neg) >= stepsize // 2 and len(dist_uc_pos) >= stepsize // 2:
			pos_ids, _ = list(zip(*sorted(dist_uc_pos, key=lambda x: x[1])))
			neg_ids, _ = list(zip(*sorted(dist_uc_neg, key=lambda x: x[1])))
			to_code.extend(pos_ids[:stepsize // 2])
			to_code.extend(neg_ids[:stepsize // 2])
		else:
			print("Not enough observations on each side of hyperplane.")
			dist_to_hp = grid.decision_function(X_pred)
			dist_to_hp = abs(dist_to_hp)
			dist_uc = list(zip(unlabeled_ids, dist_to_hp))
			dist_uc = sorted(dist_uc, key=lambda x: x[1])
			sorted_ids = list(zip(*dist_uc))[0]
			to_code.extend(sorted_ids[:stepsize])
	else:
		n_to_code = len(unlabeled_ids)
		if n_to_code == 0:
			break
		if n_to_code < stepsize:
			to_code = unlabeled_ids
		else:
			to_code = random.sample(unlabeled_ids, stepsize)
	labeled_ids.update(to_code)

simulation_data = pd.DataFrame(runs)
simulation_data.to_csv(fn, index=False)