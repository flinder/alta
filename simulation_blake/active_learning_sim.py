import argparse
import itertools
import pandas as pd
import numpy as np
import random
from scipy.stats import expon

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
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

parser = argparse.ArgumentParser()
parser.add_argument("--random", help="Random sampling.", dest='random', action='store_true')
parser.add_argument("--pweight", help="Inverse probability weighted hyperplane sampling.", dest='pweight', action='store_true')
## sparsity
## dataset

args = parser.parse_args()

class ItemSelector(BaseEstimator, TransformerMixin):

	def __init__(self, key):
		self.key = key

	def fit(self, x, y=None):
		return self

	def transform(self, data_dict):
		return data_dict[self.key]

#############################################################################
# PREPROCESSING
#############################################################################

stemmer = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return None

def split_tokenizer(text):
	return text.split()

def token(text):
	return ' '.join(word_tokenize(text))

def stem(text):
	tokens = word_tokenize(text)
	tokens = [stemmer.stem(t) for t in tokens]
	return ' '.join(tokens)

def lemma(text):
	tokens = word_tokenize(text)
	lemmas = []
	for w, p in pos_tag(tokens):
		p = get_wordnet_pos(p)
		if p is None:
			lemmas.append(lemmatizer.lemmatize(w))
		else:
			lemmas.append(lemmatizer.lemmatize(w,p))
	return ' '.join(lemmas)

data = pd.read_csv('../data/wikipedia_hate_speech.csv')
if 'token' not in data.columns:
	with Pool(8) as p:
		data['token'] = p.map(token, data['comment_text'])
	data.to_csv('../data/wikipedia_hate_speech.csv', index=False)
if 'stem' not in data.columns:
	with Pool(8) as p:
		data['stem'] = p.map(stem, data['comment_text'])
	data.to_csv('../data/wikipedia_hate_speech.csv', index=False)
if 'lemma' not in data.columns:
	with Pool(8) as p:
		data['lemma'] = p.map(lemma, data['comment_text'])
	data.to_csv('../data/wikipedia_hate_speech.csv', index=False)

#############################################################################
# INITIAL SAMPLING / DEV SET
#############################################################################

BATCH_SIZE = 100
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

train, test = train_test_split(data, test_size=0.2, random_state=1988)
test_y = {l : test[l] for l in labels}

## Keep track of labeled data
## Start with class-stratified random sample (20 observations from each class)
labeled_ids = set()
class_ids = {l : train.loc[train[l]==1,].index for l in labels}
random_ids = []
for label in labels:
	current_ids = list(class_ids[label])
	random_ids += random.sample(current_ids, 20)
labeled_ids.update(random_ids)
print(len(labeled_ids))

## Update inverse probability weights for classes
def get_weight_dict(labeled_ids):
	labeled = train.ix[list(labeled_ids), :]
	counts = [len(labeled.loc[labeled[l]==1,]) for l in labels]
	num = (np.sum(counts) + 1)
	counts = [num/(c + 1) for c in counts]
	weights = [c/np.sum(counts) for c in counts]
	return dict(zip(labels, weights))

#############################################################################
# MODEL
#############################################################################

svm = LinearSVC(penalty='l2', class_weight='balanced', random_state=64)
l1 = LinearSVC(C=20, penalty='l1', dual=False, tol=1e-3)

## Hyperparameter distributions for randomized search
parameters = {
	'clf__C': expon(50), # expon(best_val_C[label]),
	'clf__class_weight': ['balanced', None],
	'clf__tol': expon(1e-4), # expon(best_val_tol[label]),
	'text__selector__key': ['stem', 'token', 'lemma'],
	'text__vect__ngram_range': [(1, 1), (1, 2)],
	'feature_selection__threshold': ['.5*mean', '.75*mean', 'mean']
}

## Model pipeline
pipeline = Pipeline([
		('text', Pipeline([
			('selector', ItemSelector(key='token')),
			('vect', CountVectorizer(tokenizer=split_tokenizer)),
			('tfidf', TfidfTransformer())
		])),
		('feature_selection', SelectFromModel(l1)), # L1-based feature selection
		('clf', svm),
])

#############################################################################
# ACTIVE LEARNING SIMULATION
#############################################################################

runs = []
for i in range(100):
	print("\nBatch %d" % (i+1,))
	to_code = []

	if args.random == False:
		if args.pweight == True:
			weight_dict = get_weight_dict(labeled_ids)
		else:
			p = float(1)/len(labels)
			weight_dict = {l : p for l in labels} 

	labeled = train.ix[list(labeled_ids), :]

	unlabeled_ids = set(train.index) - set(labeled_ids)
	unlabeled = data.ix[list(unlabeled_ids), :]

	for label in labels:
		y_true = labeled[label]
		y_unlabeled_true = unlabeled[label]

		vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=0.0001, max_df=0.95, ngram_range=(1, 2))
		
		X_train, X_test, y_train, y_test = train_test_split(labeled, y_true, test_size=0.20, random_state=64)
		X_pred = unlabeled
		X_dev = test

		## Randomized hyperparameter search
		grid = RandomizedSearchCV(pipeline, parameters, n_iter=5, scoring=make_scorer(f1_score), n_jobs=5, random_state=1988)
		grid.fit(X_train, y_train)

		y_pred = grid.predict(X_dev)

		## Evaluate F1 on development set, save in dictionary
		_, f1 = f1_score(test_y[label], y_pred, average=None)
		support = len([y for y in y_true if y == 1])
		print("%s - F_1: %.2f, Support: %d" % (label, f1, support))
		run = {'label' : label, 'f1' : f1, 'support' : support, 'batch' : i}
		runs.append(run)

		if args.random == False:
			try:
				n = int(round(BATCH_SIZE*weight_dict[label]))

				dist_to_hp = grid.decision_function(X_pred)

				dist_uc = list(zip(unlabeled_ids, dist_to_hp))
				dist_uc_pos = [d for d in dist_uc if d[1] >= 0]
				dist_uc_neg = [d for d in dist_uc if d[1] <= 0]

				pos_ids, _ = list(zip(*sorted(dist_uc_pos, key=lambda x: x[1])))
				neg_ids, _ = list(zip(*sorted(dist_uc_neg, key=lambda x: x[1])))
				
				to_code.extend(pos_ids[:int(n/2)])
				to_code.extend(neg_ids[:int(n/2)])
			except IndexError:
				print("INDEX ERROR: Not enough observations on each side of hyperplane.")
				dist_to_hp = abs(grid.decision_function(X_pred))
				dist_uc = list(zip(unlabeled_ids, dist_to_hp))
				dist_uc = sorted(dist_uc, key=lambda x: x[1])
				sorted_ids = list(zip(*dist_uc))[0]
				to_code.extend(sorted_ids[:n])

	if args.random == True:
		to_code = random.sample(unlabeled_ids, BATCH_SIZE)
	labeled_ids.update(to_code)

## Save runs from list of dictionaries to CSV
simulation_data = pd.DataFrame(runs)
if args.random:
	simulation_data.to_csv('../data/random_simulation_data.csv', index=False)
else:
	simulation_data.to_csv('../data/active_simulation_data.csv', index=False)