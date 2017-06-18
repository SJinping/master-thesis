import gc
import re
import sys
import csv
import nltk
import math
import pickle
import langid
import string
import pprint
import string
import platform
import operator
import numpy as np
from sklearn import svm
import multiprocessing as mp
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn import feature_extraction
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
from collections import defaultdict, Counter
from gensim import corpora, models, similarities
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize, sent_tokenize, WhitespaceTokenizer, TweetTokenizer

from progressbar import AnimatedMarker, Bar, BouncingBar, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

_MAIN_DIR_ = ''
_system_ = platform.system()
if _system_ == 'Darwin':
    _MAIN_DIR_ = '/Users/Pan/Idealab'
elif _system_ == 'Linux':
    _MAIN_DIR_ = '/home/pan/Idealab'

def del_url(line):
    return re.sub(r'(\S*(\.com).*)|(https?:\/\/.*)', "", line)

def checktag(line): 
    return re.sub(r'\@\S*', "@", line)

def checkhashtag(line): 
    return re.sub(r'\#\S*', "#", line)

def checkline(line):
    return del_url(checkhashtag(checktag(line)))

def getPatternCombination(pattern_word_list):
    pattern_list = []
    for i in range(3):
        temp_pattern_word_list = list(pattern_word_list)
        temp_pattern_word_list[i] = '<.>'
        pattern_list.append(checkline(' '.join(temp_pattern_word_list)))

    return pattern_list


def getPattern(user_tweet_list):
    return [pattern for token_list in user_tweet_list for pattern in slideWindows(token_list)]


def slideWindows(token_list, size = 3):
    if len(token_list) >= size:
        return getPatternCombination(token_list[:3]) + slideWindows(token_list[1:], size)
    else:
        return []


def get_pattern_counter(text_list):
    tknzr           = TweetTokenizer(reduce_len = True)
    pattern_counter = Counter()
    pbar            = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(text_list)).start()
    index           = 0
    for text in text_list:
        pattern_counter += Counter(slideWindows(tknzr.tokenize(text[3])))
        pbar.update(index+1)
        index += 1
    pbar.finish()
    return pattern_counter


"""Get Eric's patterns as features
infile: /home/pan/Idealab/Data/open data/text_emotion.csv
return X, y and id_list
"""
def get_pattern_features(infile):
    pool        = mp.Pool(processes = mp.cpu_count()-1)
    text_dict   = defaultdict(lambda: defaultdict(Counter))
    tknzr       = TweetTokenizer(reduce_len = True)
    pat_counter = Counter()
    pat_list    = set()
    text_list   = list()
    id_list     = list()
    y           = list()
    widgets     = [FormatLabel('Processed: %(value)d records (in: %(elapsed)s)')]
    pbar        = ProgressBar(widgets = widgets)
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header     = next(spamreader)
        for row in pbar((row for row in spamreader)):
            text_list.append(row)
            id_list.append(row[0])
            y.append(row[1])
        pbar.finish()

    cpus        = mp.cpu_count()-1
    unit        = int(len(text_list) / cpus)
    text_chunks = [text_list[i*unit:i*unit+unit] for i in range(cpus)]
    text_chunks[cpus-1].extend(text_list[unit*cpus:])

    res       = [pool.apply_async(get_pattern_counter, (text_chunks[i],)) for i in range(cpus)]
    pbar      = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(res)).start()
    index     = 0
    pool.close()
    pool.join()
    print("Extracting Eric's patterns...")
    for item in res:
        pat_counter += item.get()
        pbar.update(index+1)
        index += 1
    pbar.finish()

    pat_list = list(zip(*(pat_counter.most_common()[0:5000]))[0]) + list(zip(*(pat_counter.most_common()[-5000:-1]))[0])
    print len(pat_list)

    X        = list()
    pbar     = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(id_list)).start()
    index    = 0
    print("Generating Eric's pattern features...")
    for i, idx in enumerate(id_list):
        X.append([text_dict[idx][y[i]][pat] for pat in pat_list])
        pbar.update(index+1)
        index += 1
    pbar.finish()

    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)

    return X, y, id_list

def Pickled(X, y, id_list):
    out_X  = _MAIN_DIR_ + '/Data/open data/pattern_X.pkl'
    out_y  = _MAIN_DIR_ + '/Data/open data/pattern_y.pkl'
    out_id = _MAIN_DIR_ + '/Data/open data/pattern_id.pkl'
    with open(out_X, 'wb') as f:
        pickle.dump(X, f)
    with open(out_y, 'wb') as f:
        pickle.dump(y, f)
    with open(out_id, 'wb') as f:
         pickle.dump(id_list, f)


def svm_cls(X, y, id_list, pickled = False, is_pca = True, pca_components = 50):

    if pickled:
        with open(_MAIN_DIR_ + '/Data/open data/pattern_X.pkl', 'rb') as f:
            X_pca = pickle.load(f)
        with open(_MAIN_DIR_ + '/Data/open data/pattern_y.pkl', 'rb') as f:
            y = pickle.load(f)
        with open(_MAIN_DIR_ + '/Data/open data/pattern_id.pkl', 'rb') as f:
            id_list = pickle.load(f)
    else:
        pca        = decomposition.PCA(n_components = pca_components)
        scaler     = preprocessing.StandardScaler()
        X_scale    = scaler.fit_transform(X)
        X_pca      = pca.fit_transform(X_scale)

    # X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.3, random_state = 5)
    clf = svm.SVC(C = 1, kernel = 'rbf', gamma = 'auto', probability = True)
    kf = KFold(10)
    for train_index, test_index in kf.split(X_pca):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        scores = clf.score(X_test, y_test)
        print scores

def Random_Forest(X, y, id_list, pickled = True, is_pca = True, pca_components = 50):
    if pickled:
        with open(_MAIN_DIR_ + '/Data/open data/pattern_X.pkl', 'rb') as f:
            X_pca = pickle.load(f)
        with open(_MAIN_DIR_ + '/Data/open data/pattern_y.pkl', 'rb') as f:
            y = pickle.load(f)
        with open(_MAIN_DIR_ + '/Data/open data/pattern_id.pkl', 'rb') as f:
            id_list = pickle.load(f)
    else:
        pca        = decomposition.PCA(n_components = pca_components)
        scaler     = preprocessing.StandardScaler()
        X_scale    = scaler.fit_transform(X)
        X_pca      = pca.fit_transform(X_scale)

    clf = RandomForestClassifier(n_estimators = 15, n_jobs = 7)
    kf = KFold(10)
    for train_index, test_index in kf.split(X_pca):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        scores = clf.score(X_test, y_test)
        print scores

if __name__ == '__main__':

    infile        = _MAIN_DIR_ + '/Data/open data/text_emotion.csv'
    # X, y, id_list = get_pattern_features(infile)
    X       = None
    y       = None
    id_list = None
    Random_Forest(X, y, id_list, True)