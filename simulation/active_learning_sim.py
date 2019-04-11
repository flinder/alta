import argparse
import copy
import glob
import itertools
import pandas as pd
import numpy as np
import os
import pickle
import random
import yaml
from scipy.stats import expon, beta, rv_continuous

from datetime import datetime 
from collections import Counter
from functools import reduce
from multiprocessing import Pool
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron, ElasticNet
from sklearn.naive_bayes import MultinomialNB

# Internal imports
from helpers import DtmSelector

parser = argparse.ArgumentParser()

parser.add_argument("data",
	help="Valid choices are 'tweets,' 'wikipedia_hate_speech,' and 'breitbart'",
	type=str
)
parser.add_argument("--balance",
	help="Float, proportion of positive observations",
	type=float
)
parser.add_argument("--query_strat",
	help="Valid choices are 'committee' and 'margin'",
	type=str
)
parser.add_argument("--iter",
	help="Integer, the current iteration.",
	type=int
)
parser.add_argument("--icr",
	help="Float, probability of correctly labeling an observation.",
	type=float
)
parser.add_argument("--pct_random",
	help="Float, percent of each batch that is randomly sampled.",
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

def get_random(unlabeled_ids, stepsize):
		n_to_code = len(unlabeled_ids)
		if n_to_code == 0:
			return []
		if n_to_code < stepsize:
			return unlabeled_ids
		else:
			return random.sample(unlabeled_ids, stepsize)

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
		if args.pct_random is not None:
			fn = '../data/runs/%s/%s/%s_simulation_data_%s_%s_icr_%s_rand_%s.csv' % (args.data, str(args.iter), rand, args.query_strat, str(args.balance), str(args.icr), str(args.pct_random))
		else:
			fn = '../data/runs/%s/%s/%s_simulation_data_%s_%s_icr_%s.csv' % (args.data, str(args.iter), rand, args.query_strat, str(args.balance), str(args.icr))
	else:
		if args.pct_random is not None:
			fn = '../data/runs/%s/%s/%s_simulation_data_%s_%s_rand_%s.csv' % (args.data, str(args.iter), rand, args.query_strat, str(args.balance), str(args.pct_random))
		else:
			fn = '../data/runs/%s/%s/%s_simulation_data_%s_%s.csv' % (args.data, str(args.iter), rand, args.query_strat, str(args.balance))
else:
	if args.icr:
		if args.pct_random is not None:
			fn = '../data/runs/%s/%s/%s_simulation_data_%s_icr_%s_rand_%s.csv' % (args.data, str(args.iter), rand, args.query_strat, str(args.icr), str(args.pct_random))
		else:
			fn = '../data/runs/%s/%s/%s_simulation_data_%s_icr_%s.csv' % (args.data, str(args.iter), rand, args.query_strat, str(args.icr))
	else:
		if args.pct_random is not None:
			fn = '../data/runs/%s/%s/%s_simulation_data_%s_rand_%s.csv' % (args.data, str(args.iter), rand, args.query_strat, str(args.pct_random))
		else:
			fn = '../data/runs/%s/%s/%s_simulation_data_%s.csv' % (args.data, str(args.iter), rand, args.query_strat)

if os.path.isfile(fn) or os.path.isfile(fn.replace('.csv','0.csv')):
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

def entropy(votes):
	C = len(votes)
	n_pos = sum(votes)
	n_neg = C - n_pos
	if n_pos == 0 or n_neg == 0:
		return 0
	pos = (n_pos/C) * np.log(n_pos/C)
	neg = (n_neg/C) * np.log(n_neg/C)
	return(-(pos + neg))


sgd_pipeline = Pipeline([
		('text', Pipeline([
			('selector', DtmSelector(fname=text_feature_sets[0])),
			('tfidf', TfidfTransformer())
		])),
		('clf', SGDClassifier(penalty='l2', max_iter=80, alpha=1e-06)),
])

if config['data_sets'][args.data]['sgd'] == True:
	svm = SGDClassifier(loss="hinge", random_state=1988)
	svm_parameters = {
		'clf__alpha': alpha(),
		'clf__class_weight': ['balanced', None],
		'text__selector__fname': text_feature_sets,
	}
else:
	svm = LinearSVC(penalty='l2', class_weight='balanced', random_state=1988)
	svm_parameters = {
		'clf__C': expon(50),
		'clf__class_weight': ['balanced', None],
		'text__selector__fname': text_feature_sets,
	}
if args.query_strat == 'committee':
	parameters = {
		'lr_l1' : {'text__selector__fname': text_feature_sets,
				'clf__C': expon(50)},
		'lr_l2' : {'text__selector__fname': text_feature_sets,
				'clf__C': expon(50)},
		'svm_l2_hinge' : {'text__selector__fname': text_feature_sets,
				'clf__C': expon(50),
				'clf__class_weight': ['balanced', None]},
		'svm_l2' : {'text__selector__fname': text_feature_sets,
				'clf__C': expon(50),
				'clf__class_weight': ['balanced', None]},
		'per_l1' : {'text__selector__fname': text_feature_sets},
		'per_l2' : {'text__selector__fname': text_feature_sets},
		'per_e' : {'text__selector__fname': text_feature_sets},
		'mnb' : {'text__selector__fname': text_feature_sets},
		'eln' : {'text__selector__fname': text_feature_sets}
	}
## Model pipeline
pipeline = Pipeline([
		('text', Pipeline([
			('selector', DtmSelector(fname=text_feature_sets[0])),
			('tfidf', TfidfTransformer())
		])),
		('clf', svm),

])

if args.query_strat == 'committee':
	estimators = {
		'lr_l1' : LogisticRegression(penalty='l1', random_state=1988),
		'lr_l2' : LogisticRegression(penalty='l2', random_state=1988),
		'mnb' : MultinomialNB(),
		'eln' : SGDClassifier(loss="log", penalty="elasticnet"),
		'svm_l2_hinge' : LinearSVC(penalty='l2', loss='hinge', class_weight='balanced', random_state=1988),
		'svm_l2' : LinearSVC(penalty='l2', class_weight='balanced', random_state=1988),
		'per_l1' : Perceptron(penalty='l1', tol=1e-3, random_state=1988),
		'per_l2' : Perceptron(penalty='l2', tol=1e-3, random_state=1988),
		'per_e' : Perceptron(penalty='elasticnet', tol=1e-3, random_state=1988)
	}
	pipelines = {
		name : Pipeline([
				('text', Pipeline([
					('selector', DtmSelector(fname=text_feature_sets[0])),
					('tfidf', TfidfTransformer())
				])),
				('clf', clf),
		])
		for name, clf in estimators.items()
	}

#############################################################################
# ACTIVE LEARNING SIMULATION
#############################################################################

## Batch size of 20
## Start with random sample and handle exception

runs = []
stepsize = 20
if args.pct_random is not None:
	random_stepsize = int(np.floor(stepsize * args.pct_random))
max_records = min(n_records // 5, 5000)
n_steps = max_records // stepsize
if n_steps < 200:
	n_steps = min(n_records // stepsize, 200)

## Get initial labels
labeled_ids = set(random.sample(list(train), 20))

print('beginning steps')
from datetime import datetime 

start_time = datetime.now() 
for i in range(n_steps):
	to_code = []

	labeled = data.ix[list(labeled_ids), :]
	if args.icr:
		y_true = labeled[y_col + "_icr"]
	else:
		y_true = labeled[y_col]

	unlabeled_ids = list(set(train) - set(labeled_ids))
	if len(unlabeled_ids) == 0:
		print("No more unlabeled ids.")
		break
	unlabeled = data.ix[unlabeled_ids, :]

	y_unlabeled_true = unlabeled[y_col]

	X_train, X_test, y_train, y_test = train_test_split(
		list(labeled_ids),
		y_true,
		test_size=0.20,
		random_state=1988
	)
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	X_pred = np.array(unlabeled_ids)
	X_dev = np.array(test)

	n_jobs = config['data_sets'][args.data]['n_jobs']
	## Randomized hyperparameter search
	grid = RandomizedSearchCV(
					pipeline,
					svm_parameters,
					n_iter=20,
					scoring=make_scorer(f1_score),
					n_jobs=n_jobs
				)
	try:
		grid.fit(X_train, y_train)
	except ValueError:
		run = {'support' : 0, 'batch' : i}
		runs.append(run)
		labeled_ids.update(set(random.sample(unlabeled_ids, 20)))
		continue
	if args.query_strat == 'committee':
		fit_models = {}
		for model in estimators.keys():
			fit_models[model] = RandomizedSearchCV(
				pipelines[model],
				parameters[model],
				n_iter=5,
				scoring=make_scorer(f1_score),
				n_jobs=n_jobs
			).fit(X_train, y_train)
		y_pred = {model : fit.predict(X_pred) for model, fit in fit_models.items()}
		y_preds = y_pred.values()
		entropy_matrix = np.column_stack(y_preds)
		entropies = np.apply_along_axis(entropy, 1, entropy_matrix)

	y_pred = grid.predict(X_dev)
	y_p = grid.predict(X_test)

	## Evaluate F1 on labeled data, save in dictionary
	f1_l = f1_score(y_test, y_p)
	p_l = precision_score(y_test, y_p)
	r_l = recall_score(y_test, y_p)

	## Evaluate F1 on development set, save in dictionary
	f1 = f1_score(test_y, y_pred)
	p = precision_score(test_y, y_pred)
	r = recall_score(test_y, y_pred)
	n_pos = np.sum(y_pred)

	support = len([y for y in y_true if y == 1])
	out = "%d/%d - F: %.2f, P: %.2f, R: %.2f, n_pos_pred: %d, pos_sup: %d, sup: %d" % (i+1, n_steps+1, f1, p, r, n_pos, support, (i+1) * stepsize)
	print(out)
	run = {
		'f1' : f1,
		'p' : p,
		'r' : r,
		'f1_l' : f1_l,
		'p_l' : p_l,
		'r_l' : r_l,
		'n_pos_pred' : n_pos,
		'support' : support,
		'batch' : i
	}
	run = merge(run, grid.best_params_)
	runs.append(run)

	if ((i+1) * stepsize) % 500 == 0:
		## Save runs from list of dictionaries to CSV
		simulation_data = pd.DataFrame(runs)
		simulation_data.to_csv(fn, index=False)

	if args.mode == 'active':
		if args.query_strat == 'margin':
			try:
				dist_to_hp = grid.decision_function(X_pred)
			except ValueError:
				print("All positive documents have been labeled. Breaking out of simulation")
				break
			dist_to_hp = abs(dist_to_hp)
			dist_uc = list(zip(unlabeled_ids, dist_to_hp))
			dist_uc = sorted(dist_uc, key=lambda x: x[1])
			sorted_ids = list(zip(*dist_uc))[0]
			if args.pct_random is not None:
				active_ids = sorted_ids[:(stepsize-random_stepsize)]
				to_code.extend(active_ids)
				to_code.extend(get_random(list(set(unlabeled_ids) - set(active_ids)), (stepsize - len(active_ids))))
			else:
				to_code.extend(sorted_ids[:stepsize])
		elif args.query_strat == 'committee':
			sorted_entropies = list(zip(unlabeled_ids, entropies))
			sorted_entropies = sorted(sorted_entropies, key=lambda x: x[1], reverse=True)
			sorted_ids = list(zip(*sorted_entropies))[0]
			if args.pct_random is not None:
				active_ids = sorted_ids[:(stepsize-random_stepsize)]
				to_code.extend(active_ids)
				to_code.extend(get_random(list(set(unlabeled_ids) - set(active_ids)), (stepsize - len(active_ids))))
			else:
				to_code.extend(sorted_ids[:stepsize])
		elif args.query_strat == 'emc':
			fit = sgd_pipeline.fit(X_train, y_train)
			y_pred_emc = fit.predict(X_pred)
			def get_scores(IN):
				try:
					i, x = IN
					X_cand = np.append(X_train, [x])
					y_cand = np.append(y_train,y_pred_emc[i])
					sgd = copy.deepcopy(fit)
					fit_cand = sgd.fit(X_cand, y_cand)
					y_pred_cand = fit_cand.predict(X_pred)
					label_change = np.abs(y_pred_cand - y_pred_emc)
					score = np.sum(label_change)
					return score
				except:
					return 0
			with Pool(16) as p:
				scores = p.map(get_scores, list(enumerate(X_pred)))
			print(np.max(scores))
			sorted_scores = list(zip(unlabeled_ids, scores))
			sorted_scores = sorted(sorted_scores, key=lambda x: x[1], reverse=True)
			sorted_ids = list(zip(*sorted_scores))[0]
			to_code.extend(sorted_ids[:stepsize])
	else:
		to_code = get_random(unlabeled_ids, stepsize)
	if len(to_code) > 0:
		labeled_ids.update(to_code)
	else:
		break

simulation_data = pd.DataFrame(runs)
simulation_data.to_csv(fn, index=False)
time_elapsed = datetime.now() - start_time 
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))