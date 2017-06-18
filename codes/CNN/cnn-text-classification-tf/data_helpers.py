import numpy as np
import re
import csv
import jieba
import pprint
import itertools
from collections import Counter
from nltk.tokenize import TweetTokenizer

stopword_list = ['什麼', '請問', '這裡', '不是', '意思', '這邊', '謝謝', '這句', '為何', '使用', '怎麼', \
                 '要加', '老師', '還是', '如何', '甚麼', '一下', '這個', '這樣', '問為', '因為', '何要', \
                 '用過', '是不是', '一個', '應該', '直接', '好像', '如果', '何不', '兩個', '這是', '何用', \
                 '需要', '時候', '所以', '您好', '起來', '還有', '加上', '寫成', '你好', '此句', '有點', \
                 '問此', '不好意思', '不到', '像是', '這裏', '為什麼']


def Chinese_tokenizer(string):
    segs = jieba.cut(string, cut_all = False)
    final = [seg for seg in segs if seg not in stopword_list]
    return ' '.join(final)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_tweet_str(string):
    string = re.sub(r'(\S*\.com\S*)|(https?:\/\/\S*)', "", string)
    string = re.sub(r"@\S+", "",string)
    string = clean_str(string)
    return string


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_emotion(emotion_text_file):
    emotions = ['trust', 'fear', 'sadness', 'surprise', 'anger', 'disgust', 'anticipation', 'joy']
    tknzr    = TweetTokenizer(reduce_len = True)
    x_text   = list()
    y        = list()
    with open(emotion_text_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            labels = [0 for _ in range(len(emotions))]
            string = clean_tweet_str(' '.join(tknzr.tokenize(row[3])))
            labels[emotions.index(row[1])] = 1
            x_text.append(string)
            y.append(labels)
    return [x_text, np.array(y)]

def load_data_and_labels_questions(question_file):
    question = ['others', 'perfect', 'continuous', 'past', 'future', 'relative pronouns', 'infinitive', \
               'clause', 'passive', 'prep', 'conj', 'subjective', 'V-ing', 'past participle', 'PT']

    x_text   = list()
    y        = list()
    with open(question_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            labels = [0 for _ in range(len(question))]
            string = Chinese_tokenizer(row[2])
            labels[int(row[3])] = 1
            x_text.append(string)
            y.append(labels)
    return [x_text, np.array(y)]

def load_data_and_labels_sentiment(sentiment_file):
    sentiment = ['neg', 'pos']
    tknzr     = TweetTokenizer(reduce_len = True)
    x_text    = list()
    y         = list()
    with open(sentiment_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            labels = [0 for _ in range(len(sentiment))]
            string = clean_tweet_str(' '.join(tknzr.tokenize(row[3])))
            labels[sentiment.index(row[1])] = 1
            x_text.append(string)
            y.append(labels)
    return [x_text, np.array(y)]


# for open data as training data and facebook data as testing data
def load_fb_data_and_labels(fbdata_file):
    emotions = ['trust', 'fear', 'sadness', 'surprise', 'anger', 'disgust', 'anticipation', 'joy']
    tknzr    = TweetTokenizer(reduce_len = True)
    x_text   = list()
    y        = list()
    with open(fbdata_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            labels = [0 for _ in range(len(emotions))]
            string = clean_tweet_str(' '.join(tknzr.tokenize(row[1])))
            labels[emotions.index(row[3])] = 1
            x_text.append(string)
            y.append(labels)
    return [x_text, np.array(y)]
            


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data                  = np.array(data)
    data_size             = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data   = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index   = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def kfold_iter(x, y, kfold = 10):
    pass

# positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"
# negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"
# x_text, y = load_data_and_labels(positive_data_file, negative_data_file)
# pprint.pprint(y[:5])