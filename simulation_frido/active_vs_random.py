import csv
import re
import logging
import sys
import itertools
import pandas as pd
import numpy as np
import Stemmer
import time

from gensim import corpora, matutils
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from multiprocessing import Pool

sys.path.append('../../dissdat/database/')
from db import make_session

def bi_from_uni(unis):
        bis = [] 
        for i, _ in enumerate(unis):
            try:
                bigram = '{}_{}'.format(unis[i], unis[i+1])
                bis.append(bigram)
            except IndexError:
                break
        return bis

def make_dictionaries(documents, parser, stemmer):
    
    unigrams = corpora.Dictionary()
    bigrams = corpora.Dictionary()

    for d in documents:
        # Get unigrams (lemmas)
        unis = [t for t in parser(d, stemmer)] 
        # Get bigrams
        bis = bi_from_uni(unis)
        # update dictionaries
        unigrams.doc2bow(unis, allow_update = True)
        bigrams.doc2bow(bis, allow_update = True)

    return unigrams, bigrams


def label_iter(input_file_name):
    with open(input_file_name, 'r', encoding='utf-8') as infile:
        header = next(infile)
        reader = csv.reader(infile, delimiter=',', quotechar='"')
        for row in reader:
            yield row[3]


def make_ngrams(tweet_text, parser, unigrams, bigrams, stemmer):
    # Tokenize and make bigrams
    unis = [t for t in parser(tweet_text, stemmer)] 
    bis = bi_from_uni(unis) # Get bow representation for dictionary terms
    uni_bow = unigrams.doc2bow(unis)
    bi_bow = bigrams.doc2bow(bis)

    return uni_bow, bi_bow 
    
def get_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return [prec, rec, f1]


def balance_data(X, dat, balance):
    
    # Get index of negative and postive rows
    p_max = len(dat.index[dat.annotation == 1])
    n_max = len(dat.index[dat.annotation == 0])
    
    # Get the balance in the data
    original_balance = p_max / (n_max + p_max) 
    
    # Pick the data
    if balance >= original_balance:
        # Use all positives and sample negatives
        n_neg = p_max * (1-balance) // balance
        neg_idx = dat[dat.annotation == 0].sample(n_neg).index
        pos_idx = dat[dat.annotation == 1].index
    else:
        # Use all negatives and sample positives
        n_pos = balance * n_max // (1 - balance)  
        neg_idx = dat[dat.annotation == 0].index
        pos_idx = dat[dat.annotation == 1].sample(n_pos).index
    
    # Generate dataset
    X_out = pd.concat([X_full.loc[neg_idx], X_full.loc[pos_idx]])
    y_out = pd.concat([dat.loc[neg_idx], dat.loc[pos_idx]]).annotation

    return y_out, X_out


def iteration(args):
    i = args[0]
    sparsity = args[1]
    logging.info(f'iteration {i}, sparsity {sparsity}: random')
    
    y, X = balance_data(X_full, df, sparsity) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                        random_state=221)

    y_train.index = X_train.index
    y_test.index = X_test.index

    out = []
    y_train_idx = pd.Series(shuffle(y_train.index))

    # Calculate the stepsize to produce 30 steps
    n_steps = 30
    max_samples = X_train.shape[0]
    stepsize = max_samples // n_steps
    ## Random annotation for increasing sample sizes
    for s in range(stepsize, max_samples, stepsize):
        annot_idx = y_train_idx.head(s)

        temp_y = y_train.loc[annot_idx]
        temp_X = X_train.loc[annot_idx]
        
        try:
            clf.fit(temp_X, temp_y)
            y_hat = clf.predict(X_test)
            res = ([s] + get_metrics(y_test, y_hat) + ['random_learner'] + [i] +
                   [max_samples] + [sparsity])
        except Exception as e:
            logging.error(f'RL {s}, {i}, {sparsity}: {e}')
            res = ([s] + [0]*3 + ['random_learner'] + [i] + [max_samples] + 
                   [sparsity])
     
        out.append(res)

    annot_idx = y_train_idx.head(stepsize)
    ## active learning annotation for increasing sample sizes
    logging.info(f'iteration {i}, sparsity {sparsity}: active')
    for c in itertools.count():
        # Get annotated data (as chosen in last iteration)
        temp_y = y_train.loc[annot_idx]
        temp_X = X_train.loc[annot_idx]

        try:
            clf.fit(temp_X, temp_y)

            # Get the scores for this iteration
            y_hat = clf.predict(X_test)

            res = ([len(annot_idx)] + get_metrics(y_test, y_hat) + 
                   ['active_learner'] + [i] + [max_samples] + [sparsity])

            # Get targets for next annotation step
            ## query least certain instances in the training set

            X_remain = X_train.drop(annot_idx, inplace=False, axis=0)
            
            y_hat_al = pd.Series([x[1] for x in clf.predict_proba(X_remain)])
            y_hat_al.index = X_remain.index
            dist_from_05 = (y_hat_al - 0.5)**2
            most_uncertain = dist_from_05.nsmallest(n=stepsize)
            annot_idx = annot_idx.append(pd.Series(most_uncertain.index))

     
        except Exception as e:
            # Model failed 0 on the metrics and add random data
            m = c * stepsize
            logging.error(f'AL {m}, {i}, {sparsity}: {e}')
            res = ([len(annot_idx)] + [0]*3 + ['active_learner'] + [i] + 
                   [max_samples] + [sparsity])
            annot_idx = y_train_idx.head((stepsize+m+stepsize))

        out.append(res)

        if len(annot_idx) >= max_samples:
            break
 
    return out


def tuning_iteration(clf):
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    res = get_metrics(y_test, y_hat) + [clf.__repr__()]
    return res


def process_text(text, stemmer):
    '''
    Preprocess text by:
        - Stemming
        - Removing special characters

    Args
    text: str

    Returns:
    Preprocessed tokens as list
    '''
    re_url = re.compile(('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%'
                         '[0-9a-fA-F][0-9a-fA-F]))+'))
    re_remove = re.compile('[^A-Za-z0-9\w ]')
    re_excs_space = re.compile('\s+')

    # Remove urls:
    text = re_url.sub(' ', text)
    # Remove special characters
    text = re_remove.sub(' ', text)
    # Remove excess space
    text = re_excs_space.sub(' ', text)

    text = text.lower()
    tokens = text.split()
    out = [stemmer(t) for t in tokens]
    return out

if __name__ == "__main__":

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

    # Load nlp pipeline
    logging.info('Loading nlp pipeline...')
    stemmer = Stemmer.Stemmer('german').stemWord 
     
    # Generate bag of words representation
          
    ## First pass to generate uni and bigram dictionaries
    logging.info('Generating dictionaries...')
    unigrams, bigrams = make_dictionaries(df.text, process_text, stemmer)

    ## Filter vocabularies
    unigrams.filter_extremes(no_below=2, no_above=0.2, keep_n=None)
    bigrams.filter_extremes(no_below=2, no_above=0.2, keep_n=None)

    ### Create document term matrices for uni and bigrams
    logging.info('Generating uni and bigram features...')
    grams = [make_ngrams(d, process_text, unigrams, bigrams, stemmer) 
                         for d in df.text]
    unigram_features = matutils.corpus2dense([x[0] for x 
                                            in grams],
                                            num_terms=len(unigrams)).transpose()
    bigram_features = matutils.corpus2dense([x[1] for x 
                                           in grams],
                                           num_terms=len(bigrams)).transpose()

    #X_full = pd.concat([pd.DataFrame(unigram_features),                      
    #                    pd.DataFrame(bigram_features)], axis=1)
    X_full = pd.DataFrame(unigram_features)
    X_full.index = df.index
    
    

    # Tuning
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # SGD classifiers
    #alphas = np.linspace(0.00001, 0.001, num=10)
    #losses = ['log']
    #l1_ratios = np.linspace(0, 1, num=10)
    #combos = itertools.product(alphas, losses, l1_ratios)
    #clfs = [SGDClassifier(penalty='elasticnet', loss=c[1], alpha=c[0], 
    #                      l1_ratio=c[2], random_state=26661) 
    #        for c in combos]
    #
    #logging.info('Tuning...')
    #pool = Pool(12)
    #tuning_results = pool.map(tuning_iteration, clfs)
    #pool.close()
    #
    #max_f1 = 0
    #for r in tuning_results:
    #    if r[2] > max_f1:
    #        print(r)
    #        max_f1 = r[2]

    #with open('../data/active_random_tuning_results.csv', 'w') as outfile:
    #    writer = csv.writer(outfile, delimiter=',')
    #    writer.writerow(['precision','recall','f1', 'clf'])
    #    for s in tuning_results:
    #        writer.writerow(s)

    # Experiment
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ### Initialize classifier
    clf = SGDClassifier(penalty='elasticnet', loss='log', alpha=0.0001, 
                        l1_ratio=0.77, random_state=26661)

    pool = Pool(10)
    params = list(itertools.product(np.arange(30), [0.005, 0.01, 0.05, 0.1, 
                                                     0.3, 0.5]))
    outputs = pool.map(iteration, params)
    pool.close()

    output = []
    for o in outputs:
        output.extend(o)


    with open('../data/active_random_w_sparsity_new_data.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['train_size','precision','recall','f1', 'clf', 
                         'iteration', 'max_samples', 'sparsity'])
        for s in output:
            writer.writerow(s)
