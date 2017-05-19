# -*- coding: utf-8 -*- 
import gc
import re
import sys
import csv
import nltk
import math
# import langid
import string
import pprint
import string
import platform
import operator
import numpy as np
from sklearn import svm
from sklearn.svm import SVC, LinearSVC, NuSVC
from copy import deepcopy
import multiprocessing as mp
from nltk.util import ngrams
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn import feature_extraction
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from collections import defaultdict, Counter
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize, sent_tokenize, WhitespaceTokenizer, TweetTokenizer


from progressbar import AnimatedMarker, Bar, BouncingBar, ETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

_MAIN_DIR_ = ''
_system_ = platform.system()
if _system_ == 'Darwin':
    _MAIN_DIR_ = '/Users/Pan/Idealab'
elif _system_ == 'Linux':
    _MAIN_DIR_ = '/home/pan/Idealab'

sys.path.append(_MAIN_DIR_ + "/codes/WordProc")
# sys.path.append(_MAIN_DIR_ + "/codes/pythonUtil")
# sys.path.append(_MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/RegPattern")
import wordProcBase
# import Util

EMOTIONS = ['trust', 'fear', 'sadness', 'surprise', 'anger', 'disgust', 'anticipation', 'joy']


# infile: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/FB_valence_arousal/dataset-fb-valence-arousal-anon_combined.csv
def k_means_fb_post(infile, outfile = None, n_clusters = 7, ispca = True, n_components = 20):
    docs = []

    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            docs.append(row[1])

    # calculate tfidf
    X = wordProcBase.get_tf_idf(docs, tokenizer = wordProcBase.tokenize_tweet, use_idf = False)

    km = KMeans(n_clusters = n_clusters)
    if ispca:
        pca = decomposition.PCA(n_components = n_components)
        X = X.toarray()
        X = pca.fit_transform(X)

    # save file for Matlab
    # to select the best K
    # savefile = '/home/pan/Idealab/stability/dataset-fb-valence-arousal-anon_20PCA.csv'
    # np.savetxt(savefile, X, fmt = '%10.5f', delimiter = ',')
    # return

    km.fit(X)
    dist = km.transform(X)
    clusters = km.labels_.tolist()
    cluster_center = km.cluster_centers_

    print (dist.shape)

    
    # for x in xrange(0, dist.shape[0]):
    #   output = ''
    #   sum_K = sum(dist[x,:])
    #   for k in xrange(0, dist.shape[1]):
    #       output += '{:0.4f}'.format(dist[x][k]/sum_K) + '\t'
    #   print output

    if outfile:
        with open(outfile, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')
            for i in range(0, dist.shape[0]):
                prob = [1.0/math.pow(x, 3) for x in dist[i,:]]
                prob = [x/sum(prob) for x in prob]
                writestring = prob
                spamwriter.writerow(writestring)

# add sentiment polarity as features
# NOTICE: the text order in infile and sentiment file must be kept the same!
# sentiment file: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/FB_valence_arousal/dataset-fb-valence-arousal-anon_combined_sentiment.csv
def k_means_fb_post_with_sentiment(infile, sentiment_file, outfile = None, n_clusters = 15, ispca = True):
    docs = []

    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            docs.append(row[1])

    # get sentiment for each doc
    sentiment = []
    with open(sentiment_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            sentiment.append(float(row[2]))
    sentiment = np.array([[x] for x in sentiment])


    # calculate tfidf
    X = wordProcBase.get_tf_idf(docs, tokenizer = wordProcBase.tokenize_tweet, use_idf = False)
    X = X.toarray()

    # add sentiment to X
    X = np.c_[X, sentiment]

    km = KMeans(n_clusters = n_clusters)
    if ispca:
        pca = decomposition.PCA(n_components = 250)
        # X = X.toarray()
        X = pca.fit_transform(X)

    km.fit(X)
    dist = km.transform(X)
    clusters = km.labels_.tolist()
    cluster_center = km.cluster_centers_

    print (dist.shape)

    
    # for x in xrange(0, dist.shape[0]):
    #   output = ''
    #   sum_K = sum(dist[x,:])
    #   for k in xrange(0, dist.shape[1]):
    #       output += '{:0.4f}'.format(dist[x][k]/sum_K) + '\t'
    #   print output

    if outfile:
        with open(outfile, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')
            for i in xrange(0, dist.shape[0]):
                prob = [1/math.pow(x, 2) for x in dist[i,:]]
                prob = [x/sum(prob) for x in prob]
                writestring = prob
                spamwriter.writerow(writestring)


# cluster posts with VA norms as features
# the value of a certain feature (a word) is the square of distance to original point
def k_means_fb_post_with_VA(infile, outfile = None, n_clusters = 15, ispca = True, pca_component = 250):
    VA_dict = wordProcBase.get_VA_word_dict()
    docs    = []
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header     = next(spamreader)
        for row in spamreader:
            docs.append(row[1].decode('utf-8').lower())

    # get word set
    WORDS = set()
    for text in docs:
        for token in wordProcBase.tokenize3(text):
            WORDS.add(token)
    WORDS = list(WORDS)

    X     = []
    index = 0
    pbar  = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(docs)).start()
    for text in docs:
        NWORDS = dict.fromkeys(WORDS, 0.1)
        for token in wordProcBase.tokenize3(text):
            if not VA_dict.get(token, None):
                continue
            v = VA_dict[token]['V_value']
            a = VA_dict[token]['A_value']
            NWORDS[token] = (v*v + a*a) * math.atan(a/v)
        X.append([NWORDS[w] for w in WORDS])
        pbar.update(index+1)
        index += 1
    pbar.finish()
    X = np.array(X)

    if ispca:
        pca = decomposition.PCA(n_components = pca_component)
        X = pca.fit_transform(X)

    km = KMeans(n_clusters = n_clusters)
    km.fit(X)
    dist = km.transform(X)
    clusters = km.labels_.tolist()
    cluster_center = km.cluster_centers_

    if outfile:
        with open(outfile, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')
            for i in xrange(0, dist.shape[0]):
                prob = [1/math.pow(x, 2) for x in dist[i,:]]
                prob = [x/sum(prob) for x in prob]
                writestring = prob
                spamwriter.writerow(writestring)


def k_means_fb_post_word2vec(infile, outfile):
    docs = []
    words = Counter()
    print('Reading file...')
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            tokens = wordProcBase.tokenize_tweet(row[1].lower())
            tokens = wordProcBase.tokenize3(' '.join(tokens))
            docs.append(tokens)
            for token in tokens:
                words[token] += 1

    # size: the layers of NN, also representing the length of the word vector
    # window: n-gram
    layers = 100
    model = models.Word2Vec(docs, size = layers, window = 5, min_count = 3, workers = 4)

    print(len(docs))
    print(len(words.keys()))


    print('Generating matrix...')
    pbar  = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(docs)).start()
    X = np.full((len(docs), len(words.keys()), layers), math.exp(-20))
    word_list = words.keys()
    index = 0
    for i in range(len(docs)):
        for j in range(len(word_list)):
            if word_list[j] in docs[i]:
                vec = model[word_list[j]]
                X[i, j,] = vec
        pbar.update(index+1)
        index += 1
    pbar.finish()

    print(X.shape)


# pattern_file: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/FB_valence_arousal/emo_patt_combined/fb_post_patterns_pairs_combined
def k_means_fb_post_patterns(pattern_file, outfile, n_clusters = 8, ispca = True, pca_component = 50):
    patterns = defaultdict(list)
    patterns_count = Counter()
    with open(pattern_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = '\t')
        for row in spamreader:
            patterns[row[0]].append(row[1])
            patterns_count[row[1]] += 1

    pattern_list = patterns_count.keys()
    id_list = []
    X = []
    for key, vals in patterns.items():
        id_list.append(key)
        f = []
        for p in pattern_list:
            if p in vals:
                f.append(patterns_count[p])
            else:
                f.append(0)
        X.append(f)
    X = np.array(X)

    if ispca:
        pca = decomposition.PCA(n_components = pca_component)
        X = pca.fit_transform(X)

    print(X.shape)

    km = KMeans(n_clusters = n_clusters)
    dist = km.fit_transform(X)

    if outfile:
        with open(outfile, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')
            for i in range(0, dist.shape[0]):
                prob = [1/math.pow(x, 2) for x in dist[i,:]]
                prob = [x/sum(prob) for x in prob]
                writestring = [id_list[i]] + prob
                spamwriter.writerow(writestring)



def getPatternCombination(pattern_word_list):
    pattern_list = []
    for i in range(3):
        temp_pattern_word_list = list(pattern_word_list)
        temp_pattern_word_list[i] = '<.>'
        pattern_list.append(' '.join(temp_pattern_word_list))

    return pattern_list


def getPattern(user_tweet_list):
    return [pattern for token_list in user_tweet_list for pattern in slideWindows(token_list)]


def slideWindows(token_list, size = 3):
    if len(token_list) >= size:
        return getPatternCombination(token_list[:3]) + slideWindows(token_list[1:], size)
    else:
        return []


def get_pattern_counter(pattern_counter):
    counter = Counter()
    for pattern in pattern_counter:
        counter += pattern
    return counter


def get_pattern_features(infile):
    pool      = mp.Pool(processes = mp.cpu_count()-1)
    text_dict = defaultdict(Counter)
    tknzr     = TweetTokenizer(reduce_len = True)

    with open(infile, 'r') as file:
        header = next(file)
        for line in file:
            text_dict[line.split(',')[0]].update(slideWindows(tknzr.tokenize(line.split(',')[1])))

    '''
    Calculate the count of patterns
    Nothing particular
    '''
    # pattern_counter = Counter()
    # cpus            = mp.cpu_count()-1
    # unit            = int(len(text_dict.keys()) / cpus)
    # counters        = text_dict.values()
    # patterns        = [counters[i*unit:i*unit+unit] for i in range(cpus)]
    # patterns[cpus-1].extend(counters[unit*cpus:])
    # res   = [pool.apply_async(get_pattern_counter, (patterns[i],)) for i in range(cpus)]
    # pbar  = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(res)).start()
    # index = 0
    # print("Extracting Eric's patterns...")
    # for item in res:
    #     pattern_counter += item.get()
    #     pbar.update(index+1)
    #     index += 1
    # pbar.finish()

    # pbar  = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(text_dict.keys())).start()
    # index = 0
    # print("Counting Eric's patterns...")
    # for key, val in text_dict.items():
    #     for k, v in val.items():
    #         val[k] = pattern_counter[k]
    #     pbar.update(index+1)
    #     index += 1
    # pbar.finish()

    X               = list()
    id_list         = text_dict.keys()
    pat_list        = list(set([key for counters in text_dict.values() for key in counters.keys()]))
    pbar            = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(id_list)).start()
    index           = 0
    print("Generating Eric's pattern features...")
    for idx in id_list:
        X.append([text_dict[idx][pat] for pat in pat_list])
        pbar.update(index+1)
        index += 1
    pbar.finish()

    X = np.array(X)

    return X, id_list


# using slidewindwo pattern
def k_means_with_pattern(infile, n_clusters = 15, ispca = True, pca_component = 30):
    X, id_list = get_pattern_features(infile)
    pca        = decomposition.PCA(n_components = pca_component)
    scaler     = preprocessing.StandardScaler()
    X_scale    = scaler.fit_transform(X)
    X_pca      = pca.fit_transform(X_scale)
    X          = 1000*X_pca
    X          = X.astype(int)

    # save file for Matlab to select the best K
    savefile = '/home/pan/Idealab/stability/dataset-fb-valence-arousal-anon_pat_30PCA.csv'
    np.savetxt(savefile, X, fmt = '%10.5f', delimiter = ',')
    return


# inputfile: labeled open data
# test_file: 
def svm_gridsearch(inputfile, test_file):
    doc_train = []
    target_train = []
    with open(inputfile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            doc_train.append(row[3])
            target_train.append(row[1])

    doc_test = []
    target_test = []
    with open(test_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            doc_test.append(row[1])
            target_test.append(row[3])

    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf = False)),
                          ('cls', svm.SVC())])

    parameters = {'vect__ngram_range': [(1,1), (1,2), (1,3)],
                  'cls__kernel': ('linear', 'rbf'),
                  'cls__C': (0.01, 0.1, 1.0, 5, 10, 18)}

    print('Trainning data...')
    gs_cls = GridSearchCV(pipeline, parameters, cv = 10, n_jobs = mp.cpu_count()-1)
    gs_cls.fit(doc_train, target_train)

    n_candidates = len(gs_cls.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                 % (gs_cls.cv_results_['params'][i],
                    gs_cls.cv_results_['mean_test_score'][i],
                    gs_cls.cv_results_['std_test_score'][i]))
    print('Best Paras:', gs_cls.best_params_)

    print('Predicting...')
    y_predicted = gs_cls.predict(doc_test)

    print(metrics.classification_report(target_test, y_predicted, target_names = list(set(target_test))))


def svm_with_BOW(infile, outfile, ispca = True, pca_component = 30):
    docs = []
    id_list = []
    y = []
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            docs.append(row[1])
            id_list.append(row[0])
            y.append([row[3]])
    y = np.array(y)

    # calculate tfidf
    X = wordProcBase.get_tf_idf(docs, tokenizer = wordProcBase.tokenize_tweet, use_idf = False)
    if ispca:
        pca = decomposition.PCA(n_components = pca_component)
        X = X.toarray()
        X = pca.fit_transform(X)

    clf = svm.SVC(C = 1, kernel = 'rbf', gamma = 'auto', probability = True)
    kf = KFold(X.shape[0])

    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)
            labels = clf.classes_
            label_prob = defaultdict()
            for i in range(len(prob[0])):
                label_prob[labels[i]] = prob[0][i]

            pprint.pprint(label_prob)

            spamwriter.writerow([label_prob[x] for x in EMOTIONS])


# Best Paras: {'cls__n_estimators': 256, 'vect__ngram_range': (1, 1), 'cls__max_features': 'auto'}
def RandomForest_gridsearch(inputfile, test_file, outfile):
    doc_train = []
    target_train = []
    with open(inputfile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            doc_train.append(row[3])
            target_train.append(row[1])

    doc_test = []
    target_test = []
    with open(test_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            doc_test.append(row[1])
            target_test.append(row[3])

    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf = False, ngram_range = (1,1))),
                          ('cls', RandomForestClassifier(max_features = 'auto'))])

    parameters = {'cls__n_estimators': (256, 512)}

    print('Trainning data...')
    gs_cls = GridSearchCV(pipeline, parameters, cv = 10, n_jobs = mp.cpu_count())
    gs_cls.fit(doc_train, target_train)

    n_candidates = len(gs_cls.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                 % (gs_cls.cv_results_['params'][i],
                    gs_cls.cv_results_['mean_test_score'][i],
                    gs_cls.cv_results_['std_test_score'][i]))
    print('Best Paras:', gs_cls.best_params_)

    print('Predicting...')
    y_predicted = gs_cls.predict(doc_test)
    y_prob = gs_cls.predict_proba(doc_test)

    classes = gs_cls.best_estimator_.classes_
    print(metrics.classification_report(target_test, y_predicted, labels = classes, target_names = list(set(target_test))))

    np.savetxt(outfile, y_prob, delimiter = ',')


# Best Paras: {'cls__alpha': 0.5 / 0.3, 'vect__ngram_range': (1, 1), 'vect__use_idf': False, 'cls__fit_prior': False}
def Naive_Bayes_gridsearch(inputfile, test_file, outfile):
    doc_train = []
    target_train = []
    with open(inputfile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            doc_train.append(row[3])
            target_train.append(row[1])

    doc_test = []
    target_test = []
    with open(test_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            doc_test.append(row[1])
            target_test.append(row[3])

    pipeline = Pipeline([('vect', TfidfVectorizer(use_idf = False, ngram_range = (1,1))),
                          ('cls', MultinomialNB(fit_prior = False))])

    parameters = {'cls__alpha': (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}

    print('Trainning data...')
    gs_cls = GridSearchCV(pipeline, parameters, cv = 10, n_jobs = mp.cpu_count())
    gs_cls.fit(doc_train, target_train)

    n_candidates = len(gs_cls.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                 % (gs_cls.cv_results_['params'][i],
                    gs_cls.cv_results_['mean_test_score'][i],
                    gs_cls.cv_results_['std_test_score'][i]))
    print('Best Paras:', gs_cls.best_params_)

    print('Predicting...')
    y_predicted = gs_cls.predict(doc_test)
    y_prob = gs_cls.predict_proba(doc_test)

    classes = gs_cls.best_estimator_.classes_
    print(classes)
    print(metrics.classification_report(target_test, y_predicted, labels = classes, target_names = list(set(target_test))))

    np.savetxt(outfile, y_prob, delimiter = ',')


def GMM_fb_post(infile, outfile = None, n_components = 15, ispca = True, pca_component = 250):
    docs = []
    # f = open(infile, 'r')
    # line = f.readline()
    # while line:
    #   docs.append(line)
    #   line = f.readline()

    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            docs.append(row[1])

    # calculate tfidf
    X = wordProcBase.get_tf_idf(docs, tokenizer = wordProcBase.tokenize_tweet, use_idf = False)
    if ispca:
        pca = decomposition.PCA(n_components = pca_component)
        X   = X.toarray()
        X   = pca.fit_transform(X)

    GMM = GaussianMixture(n_components = n_components, covariance_type = 'full', init_params = 'kmeans')
    GMM.fit(X)
    labels = GMM.predict(X)
    resp = GMM.predict_proba(X)

    # for x in xrange(0,resp.shape[0]):
    #   output = ''
    #   for k in xrange(0,resp.shape[1]):
    #       output += '{:0.4f}'.format(resp[x][k]) + '\t'
    #   print output

    # pprint.pprint(labels)

    if outfile:
        with open(outfile, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')
            for i in xrange(0, resp.shape[0]):
                spamwriter.writerow(resp[i:])



def LDA_post(infile, outfile, topic = 14):
    docs = []
    # f = open(infile, 'r')
    # line = f.readline()
    # while line:
    #   docs.append(line.lower().split('\t')[1])
    #   line = f.readline()
    # f.close()

    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            docs.append(row[1])

    texts = []
    widgets = [FormatLabel('Processed: %(value)d records (in: %(elapsed)s)')]
    pbar = ProgressBar(widgets = widgets)
    for doc in pbar((doc for doc in docs)):
        texts.append([word for word in wordProcBase.tokenize_tweet(doc) if word not in stopwords.words('english')])
        # doc = wordProcBase.tokenize5(doc.decode('utf-8'))
        # texts.append([word for word in doc if word not in stopwords.words('english')])
    pbar.finish()

    pprint.pprint(texts)
    return

    # create a Gensim dictionary form the texts
    dictionary = corpora.Dictionary(texts)

    # remove extrems
    dictionary.filter_extremes(no_below = 1, no_above = 0.85)

    # convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(text) for text in texts]

    print ('Applying LDA...')
    lda = models.LdaModel(corpus, num_topics = topic, id2word = dictionary, update_every = 1, chunksize = 10000, passes = 100, minimum_probability = 0.001)

    topics = lda.show_topics(num_topics = topic, num_words = 5)

    # pprint.pprint(lda.print_topics(num_topics = topic)) 

    # pprint.pprint(topics)

    print ('Writing results into file...')
    # 結果寫入文件
    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        top_prob = lda.get_document_topics(corpus) #a list of (topic_id, topic_probability) 2-tuples
        index = 1
        for prob in top_prob:
            string = [0 for i in range(topic)]
            prob = sorted(prob, key = operator.itemgetter(0), reverse = False)
            for i, p in prob:
                string[i] = p
            spamwriter.writerow(string)
            index += 1

    return

    '''
    # reading unseen data
    '''
    print ('Reading unseen data...')
    unseen = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/google_survey_data.csv"
    docs = []
    with open(unseen, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        for row in spamreader:
            docs.append(row[1])
    texts = []
    for doc in docs:
        texts.append([word for word in wordProcBase.tokenize3(doc.decode('utf-8')) if word not in stopwords.words('english')])

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below = 1, no_above = 0.85)
    corpus = [dictionary.doc2bow(text) for text in texts]

    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        top_prob = lda.get_document_topics(corpus)
        index = 1
        for prob in top_prob:
            string = [index]
            for i in xrange(0, len(prob)):
                string.append(prob[i][1])
            spamwriter.writerow(string)
            index += 1



def LDA_fb_post_pat(patfile, outfile):
    print ('Reading training file...')

    with open(patfile, 'r') as data:
        patt_dict = json.load(data)
        patt_dict = dict(patt_dict)

    # Delete the item with id 71, because post 71 contains nothing
    patt_dict.pop('71', None)

    postid = patt_dict.keys()
    postid.sort()

    texts = []
    for id_ in postid:
        texts.append(patt_dict[id_])

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below = 3, no_above = 0.7)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = models.LdaModel(corpus, num_topics = 8, id2word = dictionary, update_every = 1, chunksize = 10000, passes = 100)

    topics = lda.show_topics(num_topics = 8, num_words = 10)

    pprint.pprint(topics)

    return

    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        top_prob = lda.get_document_topics(corpus)
        print ('The length of top_prob: {}'.format(len(top_prob)))
        print ('The lenght of postid: {}'.format(len(postid)))

        index = 0
        for prob in top_prob:
            string = [postid[index]]
            probs = []
            for i in xrange(0, len(prob)):
                probs.append(prob[i][1])

            cluster = probs.index(max(probs)) + 1
            string.append(cluster)
            string.extend(probs)
            spamwriter.writerow(string)
            index += 1


def LDA_fb_post_pat2(patfile, unseenpatfile, outfile):

    print ('Reading training file...')

    with open(patfile, 'r') as data:
        patt_dict = json.load(data)
        patt_dict = dict(patt_dict)

    postid = patt_dict.keys()
    postid.sort()

    texts = []
    for id_ in postid:
        texts.append(patt_dict[id_])

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below = 3, no_above = 0.8)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = models.LdaModel(corpus, num_topics = 8, id2word = dictionary, update_every = 1, chunksize = 10000, passes = 100)

    print ('Reading unseen data...')
    with open(unseenpatfile, 'r') as data:
        unseen_patt = json.load(data)
        unseen_patt = dict(unseen_patt)

    postid = unseen_patt.keys()
    postid = [int(i) for i in postid]
    postid.sort()

    texts = []
    for id_ in postid:
        print (id_)
        texts.append(unseen_patt[str(id_)].keys())

    print ('The length of unseen posts patterns list: {}'.format(len(texts)))

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below = 3, no_above = 0.8)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print ('Getting unseen data topics...')
    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        top_prob = lda.get_document_topics(corpus)
        index = 1
        for prob in top_prob:
            string = [index]
            for i in xrange(0, len(prob)):
                string.append(prob[i][1])
            spamwriter.writerow(string)
            index += 1



'''
The following functions are for SVD
'''
# get the emotion-docs
# return a dict, key is emotion and value is tweets
def get_PFIDF_emotion(patfile, postfile):

    emos = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'trust', 'surprise']

    # create a dict that defines the key as pattern and value as how many emotions contain this pattern
    with open(patfile, 'r') as data:
        post_pat_dict = json.load(data)
        post_pat_dict = dict(post_pat_dict)

    pat_post_dict = {}
    for key, vals in post_pat_dict.items(): # key is post id and value is the pattern and its frequency in this post
        for k in vals.keys():
            post_set = pat_post_dict.get(key, set())
            post_set.add(key)
            pat_post_dict[k] = post_set

    # get emotion docs
    # key: post id
    # value: emotion the post beongs to
    post_emo_dict = {}
    emo_post_dict = {}
    with open(postfile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = spamreader.next()
        for row in spamreader:
            post_emo_dict[row[0]] = row[2]
            post_id_list = emo_post_dict.get(row[2], list())
            post_id_list.append(row[0])
            emo_post_dict[row[2]] = post_id_list

    # Calculate the number of patterns that belongs to each emotion
    # key: emotion
    # value: the number of patterns that belongs to this emotion
    emo_pat_dict = {}
    for emo in emos:
        count = 0
        for postid in emo_post_dict[emo]:
            if str(postid) not in post_pat_dict.keys():
                freq = 0
            else:
                freq = sum(post_pat_dict[str(postid)].values())
            count += freq
        emo_pat_dict[emo] = count



    # compute the PF
    pattern_list = [] # remember the order of patterns because dict is not sortable
    PF_val = []
    for key, vals in pat_post_dict.items(): # key is pattern and value is post id set
        emo_pf = []
        emo_pat_count = 0 # the number of emotions that contain pattern key

        for emo in emos:
            pat_count = 0
            for postid in list(vals & set(emo_post_dict[emo])): # get the posts occuring in the emotion
                if key in post_pat_dict[str(postid)].keys(): # the pattern belongs to emo
                    pat_count += post_pat_dict[str(postid)][key]
            PF = float(pat_count) / float(emo_pat_dict[emo])
            # PF = float(pat_count)
            if pat_count > 0:
                emo_pat_count += 1

            # calculate IDF
            IDF = math.log(8.0 / (emo_pat_count + 1), 2)
            # print PF*IDF
            emo_pf.append(PF*IDF)

        PF_val.append(emo_pf)
    tfidf = np.array(PF_val)

    return tfidf

# each row is a doc
# each col is a pattern
def get_PFIDF_doc(patlist_file, post_pat_file):
    with open(patlist_file, 'r') as data:
        patlist_dict = json.load(data)
        patlist_dict = dict(patlist_dict)

    with open(post_pat_file, 'r') as data:
        post_pat_dict = json.load(data)
        post_pat_dict = dict(post_pat_dict)

    postid_list = post_pat_dict.keys()
    postid_list.sort()
    pat_list = patlist_dict.keys()
    patt_count = len(pat_list)
    pf_val = []
    for postid in postid_list:
        patt_freq = []
        for patt in pat_list:
            frequency = float(post_pat_dict[postid].get(patt, 0)) / float(patt_count)
            patt_freq.append(frequency)
        pf_val.append(patt_freq)

    tf = np.array(pf_val)
    return tf
    

def get_SVD(tfidf):
    # tfidf = np.transpose(tfidf)
    # print ('The size of transposed tfidf is {}'.format(tfidf.shape))
    print ('The size of tfidf matrix is {}: '.format(tfidf.shape))
    u, s, v = np.linalg.svd(tfidf, full_matrices = 0)
    print ('The singular values are {}: '.format(s))
    # np.savetxt('/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/processing/matrix_u', u, delimiter = ',', fmt = '%10.5f')
    # np.savetxt('/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/processing/matrix_v', v, delimiter = ',', fmt = '%10.3f')




if __name__ == '__main__':

    # infile = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/google_survey_data_classified_test.csv"
    # infile = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/FB_valence_arousal/dataset-fb-valence-arousal-anon"
    infile = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/turk_survey_LDA/turk_survey_data_classified_all_LDA.csv"
    # outfile = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/google_survey_results_pat_LDA.csv"
    # outfile = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/FB_valence_arousal/dataset-fb-valence-arousal-anon_pat_LDA_2"
    outfile        = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/turk_survey_LDA/All_LDA_result_15_topic"
    patfile        = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/FB_valence_arousal/emo_patt_extraction/fb_post_patterns_dict.json"
    unseenpatfile  = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/processing/survey_post_patterns_dict.json"
    
    infile         = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/FB_valence_arousal/dataset-fb-valence-arousal-anon_combined.csv"
    outfile        = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/Clusters/turk_survey_data_ABC_score_LDA14_combined.csv'
    sentiment_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/FB_valence_arousal/dataset-fb-valence-arousal-anon_combined_sentiment.csv'
    pattern_file   = '/home/pan/Idealab/Data/VA_Proc/emtion_tweets/FB_valence_arousal/emo_patt_combined/fb_post_patterns_pairs_combined'
    # k_means_fb_post_patterns(pattern_file, outfile)
    # k_means_with_pattern(infile)
    # k_means_fb_post(infile, outfile)
    # k_means_fb_post_with_sentiment(infile, sentiment_file, outfile)
    # k_means_fb_post_with_VA(infile, outfile)
    # k_means_fb_post_word2vec(infile, outfile)
    # GMM_fb_post(infile, outfile)
    LDA_post(infile, outfile)
    # LDA_fb_post_pat2(patfile, unseenpatfile, outfile)
    # LDA_fb_post_pat(patfile, outfile)

    infile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score.csv'
    outfile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/Clusters/turk_survey_data_ABC_score_SVM.csv'
    # svm_with_BOW(infile, outfile)

    # patfile = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/processing/survey_post_patterns_dict.json"
    # postfile = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/google_survey_data_classified_test_forSVD.csv"
    # tfidf = get_PFIDF_emotion(patfile, postfile)

    # patlist_file = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/processing/pattern_survey_posts.json"
    # post_pat_file = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/survey/processing/survey_post_patterns_dict.json"
    # tfidf = get_PFIDF_doc(patlist_file, post_pat_file)

    # get_SVD(tfidf)


    # Process patient's data
    infile  = _MAIN_DIR_ + '/Data/patient/processing/dump_bipolar_clean.txt'
    outfile = _MAIN_DIR_ + '/Data/patient/processing/dump_bipolar_clean_LDA_8.txt'
    # LDA_post(infile, outfile)

    infile = _MAIN_DIR_ + '/Data/open data/text_emotion_filtered_map2_8emos.csv'
    test_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score.csv'
    outfile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/Cluster/turk_survey_data_ABC_score_NaiveBayes.csv'
    # svm_gridsearch(infile, test_file)
    # RandomForest_gridsearch(infile, test_file, outfile)
    # Naive_Bayes_gridsearch(infile, test_file, outfile)

