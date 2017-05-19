# -*- coding: utf-8 -*- 
import re
import sys
import csv
import nltk
import math
import langid
import string
import pprint
import platform
import operator
import numpy as np
import pandas as pd
from sklearn import svm
from nltk.util import ngrams
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from collections import defaultdict, Counter
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize, WhitespaceTokenizer, TweetTokenizer

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils


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
import wordProcBase


emotions = ['anger', 'fear', 'joy', 'trust', 'sadness', 'surprise', 'disgust', 'anticipation']

def get_open_sentiment_data(infile):
	with open(infile, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
		header = next(spamreader)
		for row in spamreader:
			labels = [0 for _ in range(len(emotions))]
			labels[emotions.index(row[1])] = 1
			yield (row[3], labels) # return text and label

def get_fb_sentiment_data(infile):
	with open(infile, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
		header = next(spamreader)
		for row in spamreader:
			labels = [0 for _ in range(len(emotions))]
			labels[emotions.index(row[3])] = 1
			yield (row[1], labels)

# convert the sentence with words to vector representation with frequency of each word
def sent2freq(open_data_file, fb_file):
	open_data_docs = list(get_open_sentiment_data(open_data_file))
	fb_data_docs = list(get_fb_sentiment_data(fb_file))

	# for open data (training data)
	docs_train = list(zip(*open_data_docs))[0]
	y_train = list(zip(*open_data_docs))[1]

	docs_test = list(zip(*fb_data_docs))[0]
	y_test = list(zip(*fb_data_docs))[1]

	vect = CountVectorizer(tokenizer = wordProcBase.tokenize_tweet)
	vect = vect.fit(docs_train)
	freq_train = vect.transform(docs_train).toarray().sum(axis = 0)
	freq_test = vect.transform(docs_test).toarray().sum(axis = 0)

	vocabulary = vect.vocabulary_
	X_train = []
	for doc in docs_train:
		tokens = wordProcBase.tokenize_tweet(doc.lower())
		x = [freq_train[vocabulary[token]] for token in tokens]
		X_train.append(x)

	X_test = []
	for doc in docs_test:
		tokens = wordProcBase.tokenize_tweet(doc.lower())
		x = [freq_test[vocabulary[token]] if token in vocabulary.keys() else 0 for token in tokens]
		X_test.append(x)

	# encoder = LabelEncoder()
	# encoder = encoder.fit(y_train)
	# y_train = encoder.transform(y_train)
	# y_test = encoder.transform(y_test)
			
	return X_train, y_train, X_test, y_test

def LSTM_sentiment(X_train, y_train, X_test, y_test):
	max_len  = max([len(item) for item in X_train])
	max_len  = max([max_len] + [len(item) for item in X_test])
	max_word = max([max([item for item in X_train])])
	max_word = max([max_word] + [max([item for item in X_test])])
	max_word = 100
	X_train  = sequence.pad_sequences(X_train, maxlen = max_len)
	X_test   = sequence.pad_sequences(X_test, maxlen = max_len)
	y_test   = np.array(y_test)
	y_train  = np.array(y_train)

	# print(type(y_test))
	# print(y_train.shape)
	# return

	# create the model
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(max_word, embedding_vecor_length, input_length = max_len)) # the first layer
	model.add(LSTM(100)) # the second layer
	model.add(Dense(output_dim = 8, activation='relu')) # the last layer with relu as the activation
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 8, batch_size = 128)

	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == '__main__':

	open_data_file = _MAIN_DIR_ + '/Data/open data/text_emotion_filtered_map2_8emos.csv'
	fb_data_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score.csv'
	X_train, y_train, X_test, y_test = sent2freq(open_data_file, fb_data_file)
	# print(y_train[0])
	LSTM_sentiment(X_train, y_train, X_test, y_test)







