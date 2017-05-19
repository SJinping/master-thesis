# -*- coding: utf-8 -*- 
import sys
import csv
import copy
import math
import random
import pprint
# import MySQLdb
import operator
import platform
import multiprocessing
import numpy as np
import matplotlib as mpl
# import plotly.plotly as py
# import plotly.graph_objs as go
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from sklearn import mixture, linear_model
from sklearn.cluster import KMeans
from matplotlib.colors import colorConverter
from sklearn.cluster import AgglomerativeClustering
# py.sign_in('s_pan', 's3a3rz5j6k')

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

_MAIN_DIR_ = ''
_system_ = platform.system()
if _system_ == 'Darwin':
    _MAIN_DIR_ = '/Users/Pan/Idealab'
elif _system_ == 'Linux':
    _MAIN_DIR_ = '/home/pan/Idealab'


sys.path.append(_MAIN_DIR_ + "/codes/WordProc")
sys.path.append(_MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/codes")
import wordProcBase
import geometric_median


EMO_LIST = ['anger', 'joy', 'sadness', 'anticipation', 'fear', 'trust', 'disgust', 'surprise']
EMO_COLOR = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#D0BBFF']
EMO_MARKER = {'anger': 'o', 'joy': 'v', 'sadness': '^', 'anticipation': 's', 'fear': '*', 'trust': '+', 'disgust': 'D', 'surprise': '<'}
BAD_WORKERS = ['A19L2XNZKO9IUR', 'A3B3EJNO25CIS3', 'A2TXEZOOSNKRJL', 'A28QYIYZLQ66WV', 'ARR4XXROZEHL3', 'A22JTMBADARX55', \
                'A3CCXD63N92HZ5', 'A2Q0KCLHK04CFT', 'A3K47RAIRCI7X8', 'A14AKM0M9VIDCG', 'APSGMQ4NNC0O', 'A21QO39XLG47J6', 'A3P1T1TA64MTA3', \
                'AXYI5CFI1YA7C', 'A1L4A4X3GYJNNW', 'A1FVU0KN04WSYB', 'A161I6SGXNY96M', 'A27CYZX2SE5FMM', 'A395440UZY2D1T', 'A2OY9UZPTIBRWV', \
                'A3BBRXT3MV7CXB', 'A1UAI2ZQSDJULP', 'A1TBWEENZ908SZ', 'A3AGJSGD009WMM']



def reverse_big5(idx, rating):
    re_ids = [2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 31, 34, 35, 37, 41, 43]
    re_ratings = [0, 5, 4, 3, 2, 1]
    if idx in re_ids:
        return re_ratings[rating]
    return rating


# caculate the agreement for each record (assignment)
# the difference among the same texts can't be greater than 1 in both valence and arousal
# calculate the percentage of texts that match the agreement level
# return the record id, worker id and text id that doesn't match the agreement criteria
def get_agreement(th = 2):
    try:
        db = MySQLdb.connect('140.114.77.11', 'pan', 'PxQAC4MZrRF89EZ6', db = 'pan', charset='utf8')
        cursor = db.cursor()
    except Exception as e:
        print(e)

    '''
    dict
    {r_id: {repeating: total repeating count, count: turker repeating count, info: [(), ()...]}}
    '''
    bad_assign = {}

    sql_textid = ""
    sql_valence = ""
    sql_arousal = ""
    for i in xrange(1, 30+1):
        sql_textid += "`Input.textid{}`,".format(i)
        sql_valence += "`Answer.radio_{}_v`,".format(i)
        sql_arousal += "`Answer.radio_{}_a`,".format(i)
    sql_textid = sql_textid[0:len(sql_textid)-1] # cut the last comma
    sql_valence = sql_valence[0:len(sql_valence)-1] # cut the last comma
    sql_arousal = sql_arousal[0:len(sql_arousal)-1] # cut the last comma

    # 0-29: textid
    # 30-59: valence
    # 60-89: arousal
    sql = "SELECT {}, {}, {}, `WorkerId`, `r_id` FROM ratings_radio_big5 WHERE `AssignmentStatus` = 'Approved'".format(sql_textid, sql_valence, sql_arousal)
    cursor.execute(sql)
    results = cursor.fetchall()
    for row in results:
        text_set        = list(set(row[0:29])) # get the distinct text ids
        repeating_count = len(row[0:29]) - len(text_set) # the number of repeating records of each row
        count           = 0 
        worker_id       = row[90]
        r_id            = int(row[91])
        for textid in text_set:
            ratings = [item for item in zip(row[0:29], row[30:59], row[60:89]) if item[0] == textid]
            if len(ratings) < 2:
                continue # no need to add count
            elif len(ratings) == 2:
                diff_v = abs(float(ratings[0][1]) - float(ratings[1][1]))
                diff_a = abs(float(ratings[0][2]) - float(ratings[1][2]))
                if diff_v > th or diff_a > th:
                    count += 1
                    assign_vals = bad_assign.get(r_id, dict())
                    assign_list = assign_vals.get('info', list())
                    assign_list.append((textid, diff_v, diff_a))
                    assign_vals['info'] = assign_list
                    bad_assign[r_id] = assign_vals
                    # bad_assign.append((r_id, worker_id, textid, diff_v, diff_a))
            else:
                print('The length of repeated texts is greater than 2!')

        if r_id not in bad_assign.keys():
            continue
        bad_assign[r_id]['turk_repeating'] = count
        bad_assign[r_id]['total_repeating'] = repeating_count
        bad_assign[r_id]['ratio'] = float(count) / float(repeating_count)
        bad_assign[r_id]['workerid'] = worker_id

    cursor.close()
    db.close()

    # pprint.pprint(bad_assign)
    # print(len(bad_assign))

    return bad_assign


# get the worker reliability
# if a certain worker is not in the list, the worker's reliability is 100%
def get_worker_reliability(th = 2):
    reliable = {}
    bad_workers = get_agreement(th)
    for key, vals in bad_workers.items():
        reli_info = reliable.get(vals['workerid'], dict())
        reli_info['total_repeating'] = reli_info.get('total_repeating', 0.0) + vals['total_repeating']
        reli_info['turk_repeating'] = reli_info.get('turk_repeating', 0.0) + vals['turk_repeating']
        reliable[vals['workerid']] = reli_info

    for key in reliable.keys():
        reliable[key]['reverse_ratio'] = 1.0 - reliable[key]['turk_repeating'] / reliable[key]['total_repeating']

    return reliable
        

'''
get valence and arousal ratings with big5 rating
infile: _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results_new/Batch_2640330_batch_results.csv'
{record_id:
    {worker:{worker_id: id, reliability: ratio},
     big5: {big5_id: score},
     texts: {id: {'ratings':[{Valence': valence, 'Arousal': arousal}, ...],
                  'emo': emotion}}
    }
}
'''
def get_batch_result_big5_all(infile):

    def isValid(row, header):
        for i in xrange(1, 30):
            if row[header.index('Answer.radio_' + str(i) + '_v')] == '' or row[header.index('Answer.radio_' + str(i) + '_a')] == '':
                return False
        for i in xrange(1, 5):
            if row[header.index('Answer.radio_s_' + str(i))] == '':
                return False
        return True

    emos = []
    vals = []
    emo_post = get_emo_result(amb_filter = False)
    # reverse keys and values
    for key, val in emo_post.items():
        key = [key]
        key.extend([key[0] for i in xrange(0, len(val) - 1)])
        emos.extend(key)
        vals.extend(val)
    emo_post = dict(zip(vals, emos))

    post_info = {}

    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)

        index = 1
        for row in spamreader:
            if row[header.index('AssignmentStatus')] != 'Approved':
                continue

            if not isValid(row, header):
                continue

            big5_dict = {}
            for i in xrange(1, 5+1):
                big5_id            = int(row[header.index('Input.sid' + str(i))])
                big5_score         = int(row[header.index('Answer.radio_s_' + str(i))])
                big5_score         = reverse_big5(big5_id, big5_score)
                big5_dict[big5_id] = big5_score

            ratings_dict = {}
            for i in xrange(1, 30+1):
                textid_header  = 'Input.textid' + str(i)
                valence_header = 'Answer.radio_' + str(i) + '_v'
                arousal_header = 'Answer.radio_' + str(i) + '_a'

                text_id = row[header.index(textid_header)]
                valence = float(row[header.index(valence_header)])
                arousal = float(row[header.index(arousal_header)])

                # get ratings for each text
                testinfo = ratings_dict.get(text_id, dict())
                ratings  = testinfo.get('ratings', list()) # a text may have repeating ratings
                ratings.append({'Valence': valence, 'Arousal': arousal})
                testinfo['ratings']          = ratings
                ratings_dict[text_id]        = testinfo
                ratings_dict[text_id]['emo'] = emo_post[text_id]

            worker_id       = row[header.index('WorkerId')]
            worker_reliable = get_worker_reliability(1)
            reliability     = 1
            if worker_id in worker_reliable.keys():
                reliability = worker_reliable[worker_id]['reverse_ratio']

            info_dict = post_info.get(index, dict())
            info_dict['worker'] = {'worker_id': worker_id, 'reliability': reliability}
            info_dict['big5'] = big5_dict
            info_dict['texts'] = ratings_dict
            post_info[index] = info_dict

            index += 1

    return post_info


'''
get the same info with get_batch_result but return both posts info and workers info
THIS FUNCTION MAY BE WRONG!!!
'''
def get_batch_result_big5(infile):

    def isValid(header, row):
        for i in xrange(1, 30):
            if row[header.index('Answer.radio_' + str(i) + '_v')] == '' or row[header.index('Answer.radio_' + str(i) + '_a')] == '':
                return False
        return True

    worker_dict = {}
    postinfo_dict = {}

    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)

        widgets = [FormatLabel('Processed: %(value)d records (in: %(elapsed)s)')]
        pbar = ProgressBar(widgets = widgets)
        for row in pbar((row for row in spamreader)):
            if row[header.index('AssignmentStatus')] != 'Approved':
                continue

            # filter those whose valence and arousal are empty
            if not isValid(row, header):
                continue

            try:
                for i in xrange(1, 30):
                    textid_header = 'Input.textid' + str(i)
                    valence_header = 'Answer.radio_' + str(i) + '_v'
                    arousal_header = 'Answer.radio_' + str(i) + '_a'

                    VA_dict = postinfo_dict.get(row[header.index(textid_header)], {'Valence':[], 'Arousal':[]})
                    VA_dict['Valence'].append(float(row[header.index(valence_header)]))
                    VA_dict['Arousal'].append(float(row[header.index(arousal_header)]))
                    postinfo_dict[row[header.index(textid_header)]] = VA_dict

                    workerinfo = worker_dict.get(row[header.index('WorkerId')], {'post_info':[], 'time':0})
                    workerinfo['post_info'].append((row[header.index(textid_header)], float(row[header.index(valence_header)]), float(row[header.index(valence_header)])))
                    workerinfo['post_info'].append((row[header.index(textid_header)], float(row[header.index(arousal_header)]), float(row[header.index(arousal_header)])))
                    # workerinfo['time'] = int(row[header.index('WorkTimeInSeconds')])
                    worker_dict[row[header.index('WorkerId')]] = workerinfo
            except Exception as e:
                print (e)
            
        pbar.finish()
    return postinfo_dict, worker_dict


'''
get turk survey results and return post info dict and workers info dict
infile: _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/Batch_2549923_batch_results.csv'

postinfo_dict:
{post_id: {'Valence': [], 'Arousal':[]}}

worker_dict:
{workerid: {'post_info':[(post_id, valence, arousal)...], 'time': number}}
'''
def get_batch_result(infile):
    worker_dict = {}
    postinfo_dict = {}

    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)

        for row in spamreader:
            VA_dict = postinfo_dict.get(row[header.index('Input.textid1')], {'Valence':[], 'Arousal':[]})
            VA_dict['Valence'].append(float(row[header.index('Answer.range_1_v')]))
            VA_dict['Arousal'].append(float(row[header.index('Answer.range_1_a')]))
            postinfo_dict[row[header.index('Input.textid1')]] = VA_dict

            VA_dict = postinfo_dict.get(row[header.index('Input.textid2')], {'Valence':[], 'Arousal':[]})
            VA_dict['Valence'].append(float(row[header.index('Answer.range_2_v')]))
            VA_dict['Arousal'].append(float(row[header.index('Answer.range_2_a')]))
            postinfo_dict[row[header.index('Input.textid2')]] = VA_dict

            workerinfo = worker_dict.get(row[header.index('WorkerId')], {'post_info':[], 'time':0})
            workerinfo['post_info'].append((row[header.index('Input.textid1')], float(row[header.index('Answer.range_1_v')]), float(row[header.index('Answer.range_1_a')])))
            workerinfo['post_info'].append((row[header.index('Input.textid2')], float(row[header.index('Answer.range_2_v')]), float(row[header.index('Answer.range_2_a')])))
            # workerinfo['time'] = int(row[header.index('WorkTimeInSeconds')])
            worker_dict[row[header.index('WorkerId')]] = workerinfo

    return postinfo_dict, worker_dict


# get the emotion of each text
# infile: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/results/Batch_2549923_batch_results_for_python.csv
# emo_post: {emotion: [text ids]}
def get_emo_result(infile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/Batch_2549923_batch_results_for_python.csv', amb_filter = True):
    emo_post = {}
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)

        if amb_filter:
            for row in spamreader:
                emo = row[1] # emotion verified by me
                amb = row[2]

                # exclude ambiguous texts
                if amb != '':
                    continue

                post_id = row[0]
                id_list = emo_post.get(emo, list())
                id_list.append(post_id)
                emo_post[emo] = list(set(id_list))
        else:
            for row in spamreader:
                emo = row[1] # emotion verified by me
                amb = row[2]
                post_id = row[0]
                id_list = emo_post.get(emo, list())
                id_list.append(post_id)
                emo_post[emo] = list(set(id_list))

    return emo_post

def get_emo_result2(infile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/new_old_pattern_score/score_combined.csv'):
    emo_post = defaultdict(list)
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',')
        header = next(spamreader)

        for row in spamreader:
            # emo_post[row[1]].append(row[0]) # for score_combined.csv
            emo_post[row[3]].append(row[0]) # for turk_survey_data_ABC_score.csv
    return emo_post


def get_post_statistics(postinfo_dict, outfile = ''):
    post_stat_dict = {}
    for key, val in postinfo_dict.items():
        mean_v = np.mean(val['Valence'])
        std_v = np.std(val['Valence'])
        median_v = np.median(val['Valence'])

        mean_a = np.mean(val['Arousal'])
        std_a = np.std(val['Arousal'])
        median_a = np.median(val['Arousal'])

        post_stat_dict[key] = dict()
        post_stat_dict[key]['mean'] = (mean_v, mean_a)
        post_stat_dict[key]['std'] = (std_v, std_a)
        post_stat_dict[key]['median'] = (median_v, median_a)

    if outfile != '':
        with open(outfile, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')
            spamwriter.writerow(['Type', 'PostId', 'mean_v', 'mean_a', 'std_v', 'std_a', 'median_v', 'median_a'])
            for key, val in post_stat_dict.items():
                spamwriter.writerow([key[0], key[1:], val['mean'][0], val['mean'][1], val['std'][0], val['std'][1], val['median'][0], val['median'][1]])

    return post_stat_dict

# rearrange the annotations
# the result will be directly processed in matlab
# write all valence then write all arousal
def arrange_post(postinfo_dict, outfile):
    id_list = [] # post id

    # read post id to ensure the order
    with open(_MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score_matlab_ordered.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        # header = next(spamreader)
        for row in spamreader:
            id_list.append(row[0])

    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        for post_id in id_list:
            if post_id not in postinfo_dict.keys():
                continue
            writestring = [post_id]
            writestring.extend(postinfo_dict[post_id]['Valence'])
            writestring.extend(postinfo_dict[post_id]['Arousal'])
            spamwriter.writerow(writestring)

def arrange_post_big5(postinfo_dict_big5, outfile):
    id_list = []

    # read post id
    with open(_MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            id_list.append(row[0])

    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        reliability = []
        for post_id in id_list:
            valence     = []
            arousal     = []
            worker_reliability = []
            # get all the workers' ratings for specific post_id
            for key, vals in postinfo_dict_big5.items():
                if post_id not in vals['texts'].keys():
                    continue
                writestring = [post_id]
                valence.append(sum([item['Valence'] for item in vals['texts'][post_id]['ratings']]) / len(vals['texts'][post_id]['ratings']))
                arousal.append(sum([item['Arousal'] for item in vals['texts'][post_id]['ratings']]) / len(vals['texts'][post_id]['ratings']))
                worker_reliability.append(vals['worker']['reliability'])
            writestring.extend(valence + arousal)
            spamwriter.writerow(writestring)
            reliability.append(worker_reliability)

    with open('/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/results_new/reliability.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')
        for item in reliability:
            spamwriter.writerow(item)



# infile: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/manyemo/manyemo.csv
# 讀取manyemo中的情緒，對tweet進行重新歸類
def get_manyemo_dict(emo_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/manyemo/manyemo.csv'):
    manyemo_dict = {}
    f = open(emo_file, 'r')
    line = f.readline()
    while line:
        line = line.lower().strip().split(',')
        manyemo_dict[line[0]] = line[1:]
        line = f.readline()
    return manyemo_dict


# rearrange the emotion
# 除filter_emo意外的emotion都歸到各自的大類，filter_emo中的emotion保留子類
# infile: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/manyemo/turk_survey_data_ABC_score_299emos_proc.csv
# filter_emo: 需要保留的類
def rearrange_emo(infile, filter_emo = ['anger']):
    manyemo_dict = get_manyemo_dict()
    outfile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/manyemo/emo_recatogarized.csv'

    with open(outfile, 'w') as csvwrite:
        spamwriter = csv.writer(csvwrite, delimiter = ',', quotechar = '"')

        with open(infile, 'r') as csvread:
            spamreader = csv.reader(csvread, delimiter = ',', quotechar = '"')
            header = next(spamreader)

            index = 1
            for row in spamreader:
                for emo, val in manyemo_dict.items():
                    if row[1] in val:
                        if emo in filter_emo:
                            spamwriter.writerow([row[0], row[1]])
                            print ('index: {}, {}'.format(index, row[1]))
                        else:
                            spamwriter.writerow([row[0], emo])
                            print ('index: {}, {}'.format(index, row[1]))
                        break
                index += 1


# 讀取各text的score
# scorefile = '/Users/Pan/Idealab/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score_299emos_scorefile.csv'
def get_subemoscore(scorefile, outfile, filter_emo = ['anger']):

    keep_subemos = ['irritated', 'aggravated', 'aggressiveness', 'frustrated']
    
    # the key is subemotion
    # the value is main emotion
    def get_subemos_dict(emo_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/manyemo/manyemo.csv'):
        subemos_dict = {}
        f = open(emo_file, 'r')
        line = f.readline()
        while line:
            line = line.lower().strip().split(',')
            for emo in line[1:]:
                subemos_dict[emo] = line[0]
            line = f.readline()
        return subemos_dict


    subemos_dict = get_subemos_dict()

    # {id: {mainemo: {subemo: score}}}
    score_dict = {}
    with open(scorefile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        text_id = next(spamreader)[1:]
        classified_emo = next(spamreader)[1:]

        for row in spamreader:
            emo = row[0]
            scores = row[1:]
            if emo not in subemos_dict.keys():
                continue

            for i in xrange(0, len(text_id)):
                main_emo = score_dict.get(text_id[i], dict())
                emo_dict = main_emo.get(subemos_dict[emo], dict())
                emo_dict[emo] = scores[i]
                main_emo[subemos_dict[emo]] = emo_dict
                score_dict[text_id[i]] = main_emo

    # 取score_dict中每種main emotion的最小score
    sorted_dict = {}
    for key, emos in score_dict.items():
        sorted_dict[key] = dict()
        for mainemo, subemos in emos.items():
            if mainemo not in filter_emo:
                sorted_score = sorted(subemos.items(), key = operator.itemgetter(1), reverse = False)
                subemo_score = sorted_score[0][1]
                sorted_dict[key][mainemo] = subemo_score
                # pprint.pprint(sorted_dict)
            else:
                anger_list = []
                anger_list.append(subemos.pop('anger'))
                anger_list.append(subemos.pop('anger2'))
                anger_list.append(subemos.pop('angry'))

                for k in subemos.keys():
                    if k not in keep_subemos:
                        subemos.pop(k)

                subemos['anger'] = min(anger_list)
                for k, v in subemos.items():
                    sorted_dict[key][k] = v
                # sorted_dict[key] = subemos

    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')
        header = sorted_dict['C1'].keys()
        spamwriter.writerow(header)

        for key, val in sorted_dict.items():
            row = [key]
            for emo in header:
                row.append(val[emo])
            spamwriter.writerow(row)

                


def plot_scatter(postinfo_dict):
    id_list = ['A952', 'A1244', 'A1576', 'A1608', 'A1609', 'A1946', 'A1952', 'A2628', 'A2875', 'A1', 'A5', 'A244', 'A473', \
                'A872', 'A1536', 'A2403', 'A2803', 'A276', 'A283', 'A466', 'A680', 'A1901', 'A1737', 'A920', 'A1090', 'A1898', \
                'A1121', 'A1162', 'A502', 'A620', 'A1019', 'A1107', 'A1780', 'A1815', 'A2731', 'A8', 'A12', 'A74', 'A319', 'A354', \
                'A439', 'A295', 'A601', 'A514', 'A1725', 'A155', 'A255', 'A275', 'A363', 'A530', 'A632', 'A923', 'A1042', 'A1208', \
                'A1605', 'A1702', 'A1838', 'A2010', 'A38', 'A191', 'A215', 'A154', 'A134', 'A684', 'A1342', 'A1354', 'A1774', 'A2232', \
                'A2332', 'A1218', 'A1761', 'A1951', 'A2351', 'A341', 'A1136', 'A1256', 'A17', 'A9', 'A106', 'A180', 'A67', 'A497', \
                'A638', 'A815', 'A907', 'A1319']
    post_dict = wordProcBase.get_fb_post_dict()
    index = 1
    for key in id_list:

        if index <= 18:
            continue

        fig, ax = plt.subplots()
        x = postinfo_dict[key]['Valence']
        y = postinfo_dict[key]['Arousal']

        if key[0] != 'A':
            continue

        x_groud = (float(post_dict[key[1:]]['Valence1']) + float(post_dict[key[1:]]['Valence2'])) / 2.0
        y_groud = (float(post_dict[key[1:]]['Arousal1']) + float(post_dict[key[1:]]['Arousal2'])) / 2.0
        plt.scatter(x_groud, y_groud, c = 'r', marker = 'v')
        

        plt.scatter(x, y, alpha = 0.6)
        plot_url = py.plot_mpl(fig, filename = key)
        print('Porcessing post: {}'.format(index))
        index += 1


# plot valence and arousal by cluster
# this is for verifying the results of clustering or classification, such as kmeans
# cluster file is suposed to be organized as: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score_kmeans_15_combined_filtered.csv
def plot_scatter_by_cluster(postinfo_dict, cluster_file, outdir):

    Max_label = 10
    Min_label = 0

    cluster_dict = {}
    with open(cluster_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header     = next(spamreader)
        for row in spamreader:
            textid  = row[0]
            cluster = row[2]
            id_list = cluster_dict.get(cluster, list())
            id_list.append(textid)
            cluster_dict[cluster] = id_list

    index = 0
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(cluster_dict.keys())).start()
    for cluster, ids in cluster_dict.items():
        fig      = plt.figure(figsize=(10, 10))
        savefile = outdir + 'kmeans_' + cluster + '.png'
        valence  = []
        arousal  = []
        for key, vals in postinfo_dict.items():
            if key in ids:
                valence += [float(x) for x in vals['Valence']]
                arousal += [float(x) for x in vals['Arousal']]
        mean_v = np.mean(valence)
        mean_a = np.mean(arousal)
        plt.scatter(valence, arousal, alpha = 0.6)
        plt.scatter(mean_v, mean_a, alpha = 0.6, marker = '*', s = 40)
        plt.title('{}  {} texts'.format(cluster, len(ids)))
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        plt.ylabel('Arousal')  
        plt.xlabel('Valence')
        plt.plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
        plt.savefig(savefile)
        plt.close()
        pbar.update(index+1)
        index += 1
    pbar.finish()


def plot_scatter_by_emotion_as_whole(postinfo_dict, outdir):
    positive = ['joy', 'surprise', 'anticipation', 'trust']
    negative = ['sadness', 'anger', 'disgust', 'fear']
    emo_post = get_emo_result2('/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score.csv')
    f, axarr = plt.subplots(2, 4, figsize = (20, 10))
    for index, emo in enumerate(positive):
        id_list = emo_post[emo]
        X = []
        Y = []
        for i in id_list:
            x = postinfo_dict[i]['Valence']
            y = postinfo_dict[i]['Arousal']
            X.extend(x)
            Y.extend(y)
        axarr[0, index].scatter(X, Y, c = 'r', marker = 'o', s = 40, edgecolors = 'face', linewidths = 0.0, alpha = 0.4)
        axarr[0, index].set_title(emo, fontsize = 12)
        axarr[0, index].set_ylim([0, 10])
        axarr[0, index].set_xlim([0, 10])

    for index, emo in enumerate(negative):
        id_list = emo_post[emo]
        X = []
        Y = []
        for i in id_list:
            x = postinfo_dict[i]['Valence']
            y = postinfo_dict[i]['Arousal']
            X.extend(x)
            Y.extend(y)
        axarr[1, index].scatter(X, Y, c = 'r', marker = 'o', s = 40, edgecolors = 'face', linewidths = 0.0, alpha = 0.4)
        axarr[1, index].set_title(emo, fontsize = 12)
        axarr[1, index].set_ylim([0, 10])
        axarr[1, index].set_xlim([0, 10])
    plt.savefig(outdir + 'emotions_scatter.png', bbox_inches='tight')



def plot_scatter_by_emotion(postinfo_dict, amb_filter = False, worker_filter = False):
    emo_post = get_emo_result2()

    savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/emos/'
    plot_index = 0
    for emo in EMO_LIST:
        id_list = emo_post[emo]
        # colors = zip([i/1000. for i in random.sample(xrange(1000), len(id_list))], [i/1000. for i in random.sample(xrange(1000), len(id_list))], [i/1000. for i in random.sample(xrange(1000), len(id_list))])
        index = 0
        fig = plt.figure(figsize=(10, 10))
        X = []
        Y = []
        for i in id_list:
            x = postinfo_dict[i]['Valence']
            y = postinfo_dict[i]['Arousal']
            X.extend(x)
            Y.extend(y)
            plt.scatter(x, y, c = 'b', marker = 'o', s = 35, edgecolors = 'face', linewidths = 0.0, alpha = 0.5)

            index += 1

        # cov = np.array([X, Y])
        # cov = np.cov(cov)

        # v, w = np.linalg.eigh(cov)
        # u = w[0] / np.linalg.norm(w[0])
        # angle = np.arctan2(u[1], u[0])
        # angle = 180 * angle / np.pi  # convert to degrees
        # v = 2. * np.sqrt(2.) * np.sqrt(v)
        # ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color)

        '''
        Plot each emotion
        '''
        plt.title("{}  {} texts".format(emo, len(id_list)))
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        plt.ylabel('Arousal')  
        plt.xlabel('Valence')
        plot_index += 1
        plt.savefig(savefile + 'emo_classifier_{}.png'.format(emo))
        plt.close()


# infile: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/results/Batch_2549923_batch_results.xlsx
def plot_scatter_by_text(postinfo_dict, infile, outdir = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/ids_5/'):
    emo_post  = get_emo_result2('/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score.csv')
    Max_label = 10
    Min_label = 0
    savefile  = outdir
    # savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/ids/'
    plot_index = 0
    for emo in EMO_LIST:
        id_list = emo_post[emo]
        # colors = zip([i/1000. for i in random.sample(xrange(1000), len(id_list))], [i/1000. for i in random.sample(xrange(1000), len(id_list))], [i/1000. for i in random.sample(xrange(1000), len(id_list))])
        # axrr = plt.subplot(240+plot_index)
        index = 0
        # fig = plt.figure(figsize=(10, 10))
        for i in id_list:
            fig = plt.figure(figsize=(4, 4))

            x      = postinfo_dict[i]['Valence']
            y      = postinfo_dict[i]['Arousal']
            mean_v = np.mean(x)
            mean_a = np.mean(y)
            plt.scatter(x, y, c = 'r', marker = 'o', s = 32, edgecolors = 'face', linewidths = 0.0, alpha = 0.8)
            # plt.scatter(mean_v, mean_a, marker = '*', s = 45, edgecolors = 'face', linewidths = 0.0)

            '''
            Plot each text
            '''
            # plt.title("{}  {}".format(emo, i))
            plt.ylim([0, 10])
            plt.xlim([0, 10])
            # plt.ylabel('Arousal')  
            # plt.xlabel('Valence')
            # plt.plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
            plot_index += 1
            # plt.savefig(savefile + 'emo_manually_{}_{}.png'.format(emo, i))
            plt.savefig(outdir + i + '.png')
            plt.close()

            index += 1


def plot_workers_emotion(worker_dict, emo_post_file, amb = True):
    emo_post = get_emo_result()
    worker_num = len(worker_dict.keys())
    print ('The number of workers: {}'.format(worker_num))

    savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/worker_emo/'

    Max_label = 10
    Min_label = 0
    with open(savefile + 'workerID.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        index = 1
        for key, vals in worker_dict.items(): # each key is a worker

            print ('processing worker {}'.format(key))

            f, axarr = plt.subplots(2, 4, figsize = (25, 10))
            # plt.title("{}_{}  {} texts".format(key, index, len(vals['post_info'])))

            ids = zip(*vals['post_info'])[0] # get text id list
            for emo, texts in emo_post.items():
                x = []
                y = []
                sub_ids = list(set(ids) & set(texts))
                for i in sub_ids:
                    loc = ids.index(i)
                    x.append(vals['post_info'][loc][1])
                    y.append(vals['post_info'][loc][2])

                # plot scatter
                emo_index = EMO_LIST.index(emo)
                row = int(emo_index / 4)
                col = emo_index % 4
                axarr[row, col].set_title("{} workerID: {} {} texts".format(emo, index, len(vals['post_info'])))
                axarr[row, col].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
                axarr[row, col].set_xlabel('Valence')
                axarr[row, col].set_ylabel('Arousal')
                axarr[row, col].axis([Min_label, Max_label, Min_label, Max_label])
                axarr[row, col].scatter(x, y, s = 30, alpha = 0.6, edgecolors = 'face', linewidths = 0.0)

            plt.savefig(savefile + 'worker_{}_{}.png'.format(index, key))

            # save worker info to print
            spamwriter.writerow([index, key, len(vals['post_info'])])

            index += 1


###
# Plot each worker's annotation for all texts
###
def plot_workers(worker_dict):
    emo_post = get_emo_result()
    worker_num = len(worker_dict.keys())
    print ('The number of workers: {}'.format(worker_num))

    savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/worker/'

    with open(savefile + 'workerID.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        index = 1
        for key, vals in worker_dict.items():
            x = []
            y = []
            color = []
            fig = plt.figure(figsize=(10, 10))
            plt.ylim([0, 10])
            plt.xlim([0, 10])
            plt.ylabel('Arousal')  
            plt.xlabel('Valence')
            for post in vals['post_info']:
                x.append(post[1])
                y.append(post[2])
                post_id = post[0]

                for emo in emo_post.keys():
                    if post_id in emo_post[emo]:
                        color.append(EMO_COLOR[EMO_LIST.index(emo)])
                        break

                plt.scatter(x, y, marker = 'o', c = color, s = 35, edgecolors = 'face', linewidths = 0.0)

            plt.title("{}_{}  {} texts".format(key, index, len(vals['post_info'])))
            # plt.show()
            plt.savefig(savefile + 'worker_{}.png'.format(key))

            # save worker info to print
            spamwriter.writerow([index, key, len(vals['post_info'])])

            index += 1


###
# Plot scatter each worker's annotation for all texts
# comparing with the average arousal
###
def plot_worker_emotion_bar(postinfo_dict, worker_dict, amb = True):
    legend_labels = ['>Q3', 'mean<Arousal<Q3', 'Q1<Arousal<mean', '<Q1']

    emo_post = get_emo_result()
    worker_num = len(worker_dict.keys())
    print ('The number of workers: {}'.format(worker_num))

    savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/worker_emo_avg_bar/'

    Max_label = 10
    Min_label = 0
    with open(savefile + 'workerID.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        index = 1
        for key, vals in worker_dict.items():  # each key is a worker

            print ('processing worker {}'.format(key))

            f, axarr = plt.subplots(2, 4, figsize=(25, 10))
            # plt.title("{}_{}  {} texts".format(key, index, len(vals['post_info'])))

            ids = zip(*vals['post_info'])[0]  # get text id list
            labels = dict()
            for emo, texts in emo_post.items():
                height = []
                labels = []
                c = []
                sub_ids = list(set(ids) & set(texts))
                for i in sub_ids:
                    avg_a = np.mean(postinfo_dict[i]['Arousal'])
                    first_quantile = np.percentile(postinfo_dict[i]['Arousal'], 25)
                    third_quantile = np.percentile(postinfo_dict[i]['Arousal'], 75)
                    loc = ids.index(i)
                    labels.append(i)

                    y = vals['post_info'][loc][2]
                    height.append(y - avg_a)

                    if vals['post_info'][loc][2] < avg_a and vals['post_info'][loc][2] > first_quantile:
                        label = legend_labels[2]
                        # labels.append(label)
                        c.append(colorConverter.to_rgb('#ff0000'))
                    elif vals['post_info'][loc][2] <= first_quantile:
                        label = legend_labels[3]
                        c.append(colorConverter.to_rgb('#930000'))
                    elif vals['post_info'][loc][2] > avg_a and vals['post_info'][loc][2] < third_quantile:
                        label = legend_labels[1]
                        # labels.append(label)
                        c.append(colorConverter.to_rgb('#00abfe'))
                    else:
                        label = legend_labels[0]
                        # labels.append(label)
                        c.append(colorConverter.to_rgb('#0000ff'))

                # plot scatter
                emo_index = EMO_LIST.index(emo)
                row = int(emo_index / 4)
                col = emo_index % 4
                axarr[row, col].set_title("{} workerID: {} {} texts".format(emo, index, len(vals['post_info'])))
                axarr[row, col].set_xlabel('Text Index')
                axarr[row, col].set_ylabel('Arousal Diff')
                axarr[row, col].set_xticks(map(lambda x: x+1, np.arange(len(height))))
                # axarr[row, col].set_xticklabels(labels, rotation='vertical')
                axarr[row, col].bar(left = np.arange(len(height)), height = height, color = c, alpha = 0.6)

            plt.savefig(savefile + 'worker_{}_{}.png'.format(index, key))
            plt.clf()

            # save worker info to print
            spamwriter.writerow([index, key, len(vals['post_info'])])

            index += 1

###
# Plot scatter each worker's annotation for all texts
# comparing with the average arousal
###
def plot_workers_emotion_scatter(postinfo_dict, worker_dict, amb = True):

    legend_labels = ['>Q3', 'mean<Arousal<Q3', 'Q1<Arousal<mean', '<Q1']

    emo_post = get_emo_result()
    worker_num = len(worker_dict.keys())
    print ('The number of workers: {}'.format(worker_num))

    savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/worker_emo_avg_scatter/'

    Max_label = 10
    Min_label = 0
    with open(savefile + 'workerID.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',', quotechar = '"')

        index = 1
        for key, vals in worker_dict.items(): # each key is a worker

            print ('processing worker {}'.format(key))

            f, axarr = plt.subplots(2, 4, figsize = (25, 10))
            # plt.title("{}_{}  {} texts".format(key, index, len(vals['post_info'])))

            ids = zip(*vals['post_info'])[0] # get text id list
            labels = dict()
            for emo, texts in emo_post.items():
                # x = []
                # y = []
                # c = []
                sub_ids = list(set(ids) & set(texts))
                emo_index = EMO_LIST.index(emo)
                row = int(emo_index / 4)
                col = emo_index % 4
                for i in sub_ids:
                    avg_a = np.mean(postinfo_dict[i]['Arousal'])
                    first_quantile = np.percentile(postinfo_dict[i]['Arousal'], 25)
                    third_quantile = np.percentile(postinfo_dict[i]['Arousal'], 75)
                    loc = ids.index(i)

                    # x.append(vals['post_info'][loc][1])
                    # y.append(vals['post_info'][loc][2])
                    x = vals['post_info'][loc][1]
                    y = vals['post_info'][loc][2]

                    if vals['post_info'][loc][2] < avg_a and vals['post_info'][loc][2] > first_quantile:
                        label = legend_labels[2]
                        # labels.append(label)
                        c = '#ff0000'
                        # c.append('#ff0000')
                    elif vals['post_info'][loc][2] <= first_quantile:
                        label = legend_labels[3]
                        # labels.append(label)
                        c = '#930000'
                        # c.append('#930000')
                    elif vals['post_info'][loc][2] > avg_a and vals['post_info'][loc][2] < third_quantile:
                        label = legend_labels[1]
                        # labels.append(label)
                        c = '#00abfe'
                        # c.append('#00abfe')
                    else:
                        label = legend_labels[0]
                        # labels.append(label)
                        c = '#0000ff'
                        # c.append('#0000ff')

                    p = axarr[row, col].scatter(x, y, s=43, c=c, label = label, alpha=0.8, edgecolors='face', linewidths=0.0)
                    labels[label] = p

                # plot scatter
                # emo_index = EMO_LIST.index(emo)
                # row = int(emo_index / 4)
                # col = emo_index % 4
                axarr[row, col].set_title("{} workerID: {} {} texts".format(emo, index, len(vals['post_info'])))
                axarr[row, col].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
                axarr[row, col].set_xlabel('Valence')
                axarr[row, col].set_ylabel('Arousal')
                axarr[row, col].axis([Min_label, Max_label, Min_label, Max_label])
                # axarr[row, col].scatter(x, y, s = 43, c = c, alpha = 0.8, edgecolors = 'face', linewidths = 0.0)

            if len(labels.keys()):
                plt.legend(zip(*labels.items())[1], zip(*labels.items())[0], scatterpoints = 1, loc = 'lower right', prop = dict(size = 12))
                # plt.figlegend(zip(*labels.items())[1], zip(*labels.items())[0], loc = 'lower center')
                labels = dict()
            plt.savefig(savefile + 'worker_{}_{}.png'.format(index, key))
            plt.close()

            # save worker info to print
            spamwriter.writerow([index, key, len(vals['post_info'])])

            index += 1

# For each text, average the ratings of all workers
# Then plot the ratings by emotion
def plot_text_avg_worker_by_emotion(postinfo_dict):
    emo_post = get_emo_result()
    post_stat_dict = get_post_statistics(postinfo_dict)
    savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/emos/avg_worker/'
    plot_index = 0

    Max_label = 10
    Min_label = 0
    for emo in EMO_LIST:
        id_list = emo_post[emo]
        colors = zip([i/1000. for i in random.sample(xrange(1000), len(id_list))], [i/1000. for i in random.sample(xrange(1000), len(id_list))], [i/1000. for i in random.sample(xrange(1000), len(id_list))])
        index = 0
        fig = plt.figure(figsize=(10, 10))
        for i in id_list:
            x = post_stat_dict[i]['mean'][0]
            y = post_stat_dict[i]['mean'][1]
            plt.scatter(x, y, c = colors[index], marker = 'o', s = 35, edgecolors = 'face', linewidths = 0.0)

            index += 1

        '''
        Plot each emotion
        '''
        plt.title("{}  {} texts".format(emo, len(id_list)))
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        plt.ylabel('Arousal')  
        plt.xlabel('Valence')
        plt.plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
        plot_index += 1
        plt.savefig(savefile + 'emo_avg_{}.png'.format(emo))


# bar
def plot_text_avg_worker_by_text_2(postinfo_dict, worker_dict):

    savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/worker_text_bar/'

    # see anger first
    # see arousal first
    Max_label = 10
    Min_label = 0
    post_stat_dict = get_post_statistics(postinfo_dict)
    emo_post = get_emo_result()
    workers = worker_dict.keys()

    index = 0
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(emo_post['anger'])).start()
    for worker in workers:
        # get the post ids that the worker rated
        rate_ids = zip(*worker_dict[worker]['post_info'])[0]
        rate_arousal = zip(*worker_dict[worker]['post_info'])[2]
        height = []
        fig = plt.figure(figsize=(10, 10))
        for idx in rate_ids:
            arousal = postinfo_dict[idx]['Arousal']
            avg_a = np.mean(arousal)
            height.append(rate_arousal[rate_ids.index(idx)])

        plt.plot([0, len(height)], [0, 0])
        plt.bar(left = np.arange(len(height)), height = height)
        plt.xticks(np.arange(len(height)), rate_ids, rotation='vertical')
        plt.title("{}_{}_avg: {}  {}".format('anger', idx, avg_a, worker))
        plt.ylabel('Arousal')
        plt.xlabel('Text Index')
        plt.show()
        plt.clf()

    # for idx in emo_post['anger']:
    #   avg_a = post_stat_dict[idx]['mean'][1]
    #   for worker in workers:

    #       # get the post ids that the worker rated
    #       rate_ids = zip(*worker_dict[worker]['post_info'])[0]
    #       # get the index of current text id
    #       try:
    #           cur_id = rate_ids.index(idx)
    #       except ValueError:
    #           continue

    #       fig = plt.figure(figsize=(10, 10))
            
    #       # get valence and arousal that this worker rated
    #       x = worker_dict[worker]['post_info'][cur_id][1] # valence
    #       y = worker_dict[worker]['post_info'][cur_id][2] # arousal


    #       left = []
    #       height = []
    #       bottom = []
    #       width = 0.8
    #       for key, vals in worker_dict.items():
    #           ids = zip(*vals['post_info'])[0]
    #           if idx not in ids:
    #               continue
    #           if key != worker: # plot other workers
    #               valence = zip(*vals['post_info'])[1]
    #               arousal = zip(*vals['post_info'])[2] # arousal
    #               y = arousal[ids.index(idx)]
    #               x = valence[ids.index(idx)]
    #               height.append(x - avg_a)
    #               bottom.append(avg_a)

    #       '''
    #       Plot each worker for each text
    #       '''
    #       plt.plot([0, len(height)], [0, 0])
    #       plt.bar(left = np.arange(len(height)), height = height, width = width)
    #       plt.title("{}_{}_avg: {}  {}".format('anger', idx, avg_a, worker))
    #       # plt.ylim([0, 10])
    #       # plt.xlim([0, 10])
    #       plt.ylabel('Arousal')  
    #       # plt.xlabel('Valence')
    #       plt.xlabel('Text Index')
    #       # plt.plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
    #       # plot_index += 1
    #       # plt.savefig(savefile + 'emo_anger_{}_{}.png'.format(worker, idx))
    #       plt.show()
    #       plt.clf()

        pbar.update(index+1)
        index += 1
    pbar.finish()


# scatter
def plot_text_avg_worker_by_text(postinfo_dict, worker_dict):

    savefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/worker_text/'

    # see anger first
    # see arousal first
    Max_label = 10
    Min_label = 0
    post_stat_dict = get_post_statistics(postinfo_dict)
    emo_post = get_emo_result()
    workers = worker_dict.keys()

    index = 0
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(emo_post['anger'])).start()
    for idx in emo_post['anger']:
        avg_a = post_stat_dict[idx]['mean'][1]
        for worker in workers:

            # get the post ids that the worker rated
            rate_ids = zip(*worker_dict[worker]['post_info'])[0]
            # get the index of current text id
            try:
                cur_id = rate_ids.index(idx)
            except ValueError:
                continue

            fig = plt.figure(figsize=(10, 10))

            # plot diagnal
            plt.plot([Min_label, Max_label], [avg_a, avg_a])
            
            # get valence and arousal that this worker rated
            x = worker_dict[worker]['post_info'][cur_id][1]
            y = worker_dict[worker]['post_info'][cur_id][2]
            # Plot the worker to be compared
            plt.scatter(x, y, c = 'k', marker = '*', s = 60, edgecolors = 'face', linewidths = 0.0)

            for key, vals in worker_dict.items():
                ids = zip(*vals['post_info'])[0]
                if idx not in ids:
                    continue
                if key != worker: # plot other workers
                    valence = zip(*vals['post_info'])[1]
                    arousal = zip(*vals['post_info'])[2] # arousal
                    y = arousal[ids.index(idx)]
                    x = valence[ids.index(idx)]
                    # diff = a - avg_a

                    if y - avg_a < 0.:
                        plt.scatter(x, y, c = 'r', marker = 'o', s = 30, edgecolors = 'face', linewidths = 0.0, alpha = 0.4)
                    else:
                        plt.scatter(x, y, c = 'g', marker = 'o', s = 30, edgecolors = 'face', linewidths = 0.0, alpha = 0.4)

            '''
            Plot each worker for each text
            '''
            plt.title("{}_{}  {}".format('anger', idx, worker))
            plt.ylim([0, 10])
            plt.xlim([0, 10])
            plt.ylabel('Arousal')  
            plt.xlabel('Valence')
            plt.plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
            # plot_index += 1
            plt.savefig(savefile + 'emo_anger_{}_{}.png'.format(worker, idx))
            plt.clf()

        pbar.update(index+1)
        index += 1
    pbar.finish()


# 建立一个dict存储要绘制的散点图
# fig按照emotion绘制，plot按照big5 score绘制
'''
{emotion:
        {big5_score:
                    {'Valence': [], 'Arousal': []}
        }
}
'''
def plot_workers_scatter_by_big5_score(post_info, outdir, agreement_filter = None):

    # get repeating indexes, returning the indexes of repeating texts
    def get_repeating_indeces(record):
        text_set = list(set(record))
        repeating_index = []
        for textid in text_set:
            cur = 0
            indeces = []
            for item in record:
                if item == textid:
                    indeces.append(cur)
                cur += 1
            if len(indeces) >= 2:
                repeating_index.append(tuple(indeces))
        return repeating_index
    
    Min_label = 0
    Max_label = 10

    def get_emo_by_id(textid):
        emo = ''
        for key, vals in post_info.items():
            if textid in vals['texts'].keys():
                return vals['texts'][textid]['emo']
        return emo

    try:
        db = MySQLdb.connect('140.114.77.11', 'pan', 'PxQAC4MZrRF89EZ6', db = 'pan', charset='utf8')
        cursor = db.cursor()
    except Exception as e:
        print (e)
        return

    score_location = {1: (0,0), 2: (0,1), 3: (0,2), 4: (1,0), 5: (1,1)}

    sql_valence = ""
    sql_arousal = ""
    sql_textid = ""
    for i in xrange(1, 30+1):
        sql_valence += "`Answer.radio_{}_v`,".format(i)
        sql_arousal += "`Answer.radio_{}_a`,".format(i)
        sql_textid += "`Input.textid{}`,".format(i)
    sql_valence = sql_valence[0:len(sql_valence)-1] # remove the last comma
    sql_arousal = sql_arousal[0:len(sql_arousal)-1] # remove the last comma
    sql_textid = sql_textid[0:len(sql_textid)-1] # remove the last comma

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = 41).start()
    cur_index = 0

    if not agreement_filter:
        i = 1
        while i <= 41: # the number of records
            for j in xrange(0, 5): # big5 statements, 5 statements each record
                index = i + j # big5 statement index
                fig_dict = {'anger': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'joy': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'sadness': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'anticipation': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'fear': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'trust': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'disgust': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'surprise': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}}
                for k in xrange(1, 6): # big5 score
                    sql = "SELECT {},{},{},`WorkerId`, `Input.statement{}` FROM ratings_radio_big5 WHERE `Input.sid{}` = {} AND `Answer.radio_s_{}` = '{}' AND `AssignmentStatus` = 'Approved'".format(sql_valence, sql_arousal, sql_textid, j+1, j+1, index, j+1, k)
                    cursor.execute(sql)
                    results = cursor.fetchall()
                    
                    for row in results:
                        big5_statement = row[91]
                        for v, a, textid in zip(row[0:29], row[30:59], row[60:89]): # get valence, arousal and textid, respectively
                            emo = get_emo_by_id(textid)
                            V_list = fig_dict[emo][k].get('Valence', list())
                            A_list = fig_dict[emo][k].get('Arousal', list())
                            V_list.append(float(v))
                            A_list.append(float(a))
                            fig_dict[emo][k]['Valence'] = V_list
                            fig_dict[emo][k]['Arousal'] = A_list

                # plot scatters
                for key, vals in fig_dict.items(): # key is emotion
                    f, axarr = plt.subplots(2, 3, figsize = (25, 15))
                    annotators = []
                    for s, val in vals.items(): # s means score
                        if not val.has_key('Valence'):
                            annotators.append(0)
                            continue
                        annotators.append(len(val['Valence']))
                        r, c = score_location[s]
                        axarr[r][c].scatter(val['Valence'], val['Arousal'], marker = 'o', s = 70, c = 'r', alpha = 0.6, label = key)
                        axarr[r][c].scatter(np.mean(val['Valence']), np.mean(val['Arousal']), marker = '*', s = 85, c = 'b', alpha = 0.7)
                        axarr[r][c].set_xlabel('Valence')
                        axarr[r][c].set_ylabel('Arousal')
                        axarr[r][c].set_xticks(np.arange(0, 10), 1)
                        axarr[r][c].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
                        axarr[r][c].axis([Min_label, Max_label, Min_label, Max_label])
                        axarr[r][c].set_title("big5-{} avg. ({}, {})".format(s, round(np.mean(val['Valence']), 2), round(np.mean(val['Arousal']), 2)))
                    axarr[1][2].set_xlabel('index {}: {}'.format(index, big5_statement))
                    axarr[1][2].set_ylabel('count of annotators')
                    axarr[1][2].bar(left = [1, 2, 3, 4, 5], height = annotators, alpha = 0.6)
                    f.suptitle(key, fontsize = 20)
                    plt.savefig(outdir + 'big5_index_{}_{}.png'.format(index, key))
                    plt.close()
            i += 5
            pbar.update(cur_index+1)
            cur_index += 1
    else:
        '''
        篩選不符合條件的worker
        對符合條件的重複打分的text求平均
        bad worker: 打分差距在2分以上（不含）的tweet的數目佔所有重複tweet數目的比例大於30%的worker
        計算重複text平均值：
        遍歷textid list找到重複text的下標，根據找到的下標對valence/arousal數組求均值
        '''
        # get bad worker list
        bad_workers = []
        for key, vals in agreement_filter.items():
            if vals['ratio'] >= 0.3:
                bad_workers.append(vals['workerid'])
        i = 1
        while i <= 41: # the number of records
            for j in xrange(0, 5): # big5 statements, 5 statements each record
                index = i + j # big5 statement index
                fig_dict = {'anger': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'joy': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'sadness': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'anticipation': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'fear': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'trust': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'disgust': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'surprise': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}}
                for k in xrange(1, 6): # big5 score
                    sql = "SELECT {},{},{},`WorkerId`, `Input.statement{}` FROM ratings_radio_big5 WHERE `Input.sid{}` = {} AND `Answer.radio_s_{}` = '{}' AND `AssignmentStatus` = 'Approved'".format(sql_valence, sql_arousal, sql_textid, j+1, j+1, index, j+1, k)
                    cursor.execute(sql)
                    results = cursor.fetchall()
                    
                    for row in results:
                        worker = row[90] # worker id

                        if worker in bad_workers:
                            continue

                        # get repeating texts indices tuples
                        indeces = get_repeating_indeces(row[60:89]) # 60-89: text id

                        # cal the average valence and arousal, storing in new lists
                        filtered_tuple = []
                        v_a_id = zip(row[0:29], row[30:59], row[60:89]) # keep valence, arousal and textid at the same time
                        for cur in xrange(0, len(v_a_id)):
                            if cur in zip(*indeces)[0]: # combine
                                second_cur = zip(*indeces)[0].index(cur)
                                valence = (float(zip(*v_a_id)[0][cur]) + float(zip(*v_a_id)[0][zip(*indeces)[1][second_cur]])) / 2.0
                                arousal = (float(zip(*v_a_id)[1][cur]) + float(zip(*v_a_id)[1][zip(*indeces)[1][second_cur]])) / 2.0
                                filtered_tuple.append((valence, arousal, zip(*v_a_id)[2][cur]))
                            elif cur in zip(*indeces)[1]: # neglect the repeating one
                                continue
                            else: # keep
                                filtered_tuple.append(v_a_id[cur])
                            
                        big5_statement = row[91]
                        # for v, a, textid in zip(row[0:29], row[30:59], row[60:89]): # get valence, arousal and textid, respectively
                        for v, a, textid in filtered_tuple: # get valence, arousal and textid from distinct tuples
                            emo = get_emo_by_id(textid)
                            V_list = fig_dict[emo][k].get('Valence', list())
                            A_list = fig_dict[emo][k].get('Arousal', list())
                            V_list.append(float(v))
                            A_list.append(float(a))
                            fig_dict[emo][k]['Valence'] = V_list
                            fig_dict[emo][k]['Arousal'] = A_list

                # plot scatters
                for key, vals in fig_dict.items(): # key is emotion
                    f, axarr = plt.subplots(2, 3, figsize = (25, 15))
                    annotators = []
                    for s, val in vals.items(): # s means score
                        if not val.has_key('Valence'):
                            annotators.append(0)
                            continue
                        annotators.append(len(val['Valence']))
                        r, c = score_location[s]
                        axarr[r][c].scatter(val['Valence'], val['Arousal'], marker = 'o', s = 70, c = 'r', alpha = 0.6, label = key)
                        axarr[r][c].scatter(np.mean(val['Valence']), np.mean(val['Arousal']), marker = '*', s = 85, c = 'b', alpha = 0.7)
                        axarr[r][c].set_xlabel('Valence')
                        axarr[r][c].set_ylabel('Arousal')
                        axarr[r][c].set_xticks(np.arange(0, 10), 1)
                        axarr[r][c].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
                        axarr[r][c].axis([Min_label, Max_label, Min_label, Max_label])
                        axarr[r][c].set_title("big5-{} avg. ({}, {})".format(s, round(np.mean(val['Valence']), 2), round(np.mean(val['Arousal']), 2)))
                    axarr[1][2].set_xlabel('index {}: {}'.format(index, big5_statement))
                    axarr[1][2].set_ylabel('count of annotators')
                    axarr[1][2].bar(left = [1, 2, 3, 4, 5], height = annotators, alpha = 0.6)
                    f.suptitle(key, fontsize = 20)
                    plt.savefig(outdir + 'big5_index_{}_{}.png'.format(index, key))
                    plt.close()
            i += 5
            pbar.update(cur_index+1)
            cur_index += 1
            
    
    pbar.finish()
    cursor.close()
    db.close()




# similar to plot_workers_scatter_by_big5_score
# 橫軸：BFI的分數
# 縱軸：valence or arousal
# 一個figure包含八個emotion
def plot_scatter_big5_index_VA(post_info, outdir, agreement_filter = None):
    # get repeating indexes, returning the indexes of repeating texts
    def get_repeating_indeces(record):
        text_set = list(set(record))
        repeating_index = []
        for textid in text_set:
            cur = 0
            indeces = []
            for item in record:
                if item == textid:
                    indeces.append(cur)
                cur += 1
            if len(indeces) >= 2:
                repeating_index.append(tuple(indeces))
        return repeating_index
    
    Min_label = 0
    Max_label = 10

    def get_emo_by_id(textid):
        emo = ''
        for key, vals in post_info.items():
            if textid in vals['texts'].keys():
                return vals['texts'][textid]['emo']
        return emo

    try:
        db = MySQLdb.connect('140.114.77.11', 'pan', 'PxQAC4MZrRF89EZ6', db = 'pan', charset='utf8')
        cursor = db.cursor()
    except Exception as e:
        print (e)
        return

    score_location = {1: (0,0), 2: (0,1), 3: (0,2), 4: (1,0), 5: (1,1)}
    emo_location = {'anger': ((0,0), (0,1)), 'joy': ((0,2), (0,3)), \
                    'sadness': ((1,0), (1,1)), 'anticipation': ((1,2), (1,3)), \
                    'fear': ((2,0), (2,1)), 'trust': ((2,2), (2,3)), 'disgust': ((3,0), (3,1)), 'surprise': ((3,2), (3,3))}

    sql_valence = ""
    sql_arousal = ""
    sql_textid = ""
    for i in xrange(1, 30+1):
        sql_valence += "`Answer.radio_{}_v`,".format(i)
        sql_arousal += "`Answer.radio_{}_a`,".format(i)
        sql_textid += "`Input.textid{}`,".format(i)
    sql_valence = sql_valence[0:len(sql_valence)-1] # remove the last comma
    sql_arousal = sql_arousal[0:len(sql_arousal)-1] # remove the last comma
    sql_textid = sql_textid[0:len(sql_textid)-1] # remove the last comma

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = 41).start()
    cur_index = 0

    # plot parameters
    fig_width = 20
    fig_height = 20
    scatter_size = 50
    mean_size = 30


    if not agreement_filter:
        i = 1
        while i <= 41: # the number of records
            for j in xrange(0, 5): # big5 statements, 5 statements each record
                index = i + j # big5 statement index
                fig_dict = {'anger': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'joy': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'sadness': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'anticipation': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'fear': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'trust': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'disgust': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'surprise': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}}
                for k in xrange(1, 6): # big5 score
                    sql = "SELECT {},{},{},`WorkerId`, `Input.statement{}` FROM ratings_radio_big5 WHERE `Input.sid{}` = {} AND `Answer.radio_s_{}` = '{}' AND `AssignmentStatus` = 'Approved'".format(sql_valence, sql_arousal, sql_textid, j+1, j+1, index, j+1, k)
                    cursor.execute(sql)
                    results = cursor.fetchall()
                    
                    for row in results:
                        big5_statement = row[91]
                        for v, a, textid in zip(row[0:29], row[30:59], row[60:89]): # get valence, arousal and textid, respectively
                            emo = get_emo_by_id(textid)
                            V_list = fig_dict[emo][k].get('Valence', list())
                            A_list = fig_dict[emo][k].get('Arousal', list())
                            V_list.append(float(v))
                            A_list.append(float(a))
                            fig_dict[emo][k]['Valence'] = V_list
                            fig_dict[emo][k]['Arousal'] = A_list

                # plot scatters
                f, axarr = plt.subplots(4, 4, figsize = (fig_width, fig_height))
                for key, vals in fig_dict.items():
                    annotators    = []
                    x_index       = []
                    x_index_avg   = []
                    y_valence     = []
                    y_arousal     = []
                    y_valence_avg = []
                    y_arousal_avg = []

                    for s, val in vals.items():
                        if not val.has_key('Valence'):
                            annotators.append(0)
                            continue
                        annotators.append(len(val['Valence']))

                        x_index   += [s for x in range(len(val['Valence']))]
                        y_valence += val['Valence']
                        y_arousal += val['Arousal']
                        if len(val['Valence']):
                            x_index_avg.append(s)
                            y_valence_avg.append(np.mean(val['Valence']))
                            y_arousal_avg.append(np.mean(val['Arousal']))

                    # plot valence
                    r, c = emo_location[key][0]
                    r_score = 0.
                    if len(x_index) > 3:
                        regr   = linear_model.LinearRegression()
                        regr_x = np.array([[x] for x in x_index])
                        regr_y = np.array([[x] for x in y_valence])
                        regr.fit(regr_x, regr_y)
                        y_pre   = regr.predict(regr_x)
                        r_score = regr.score(regr_x, regr_y)
                        axarr[r][c].plot(x_index, y_pre, color='b', linewidth = 2) # plot regression

                    axarr[r][c].scatter(x_index, y_valence, marker = 'o', s = scatter_size, c = 'g', alpha = 0.6)
                    axarr[r][c].scatter(np.arange(1, len(y_valence_avg)+1), y_valence_avg, marker = 's', s = mean_size, c = 'k', alpha = 0.0) # plot average
                    axarr[r][c].plot(np.arange(1, len(y_valence_avg)+1), y_valence_avg, ls = 'dotted', c = 'k') # plot average line
                    axarr[r][c].plot(x_index, y_pre, color='b', ls = 'dotted', linewidth = 2) # plot regression
                    axarr[r][c].set_xlabel('BFI score')
                    axarr[r][c].set_ylabel('Valence')
                    axarr[r][c].axis([int(0), int(6), Min_label, Max_label])
                    axarr[r][c].set_title('{}: Valence  r_squre = {}'.format(key.capitalize(), round(r_score, 4)), fontdict = {'fontsize': 12})

                    # plot arousal
                    r, c = emo_location[key][1]
                    r_score = 0.
                    if len(x_index) > 3:
                        regr   = linear_model.LinearRegression()
                        regr_x = np.array([[x] for x in x_index])
                        regr_y = np.array([[x] for x in y_arousal])
                        regr.fit(regr_x, regr_y)
                        r_score = regr.score(regr_x, regr_y)
                        y_pre   = regr.predict(regr_x)
                        axarr[r][c].plot(x_index, y_pre, color='b', linewidth = 2) # plot regression

                    axarr[r][c].scatter(x_index, y_arousal, marker = 'o', s = scatter_size, c = 'r', alpha = 0.6)
                    axarr[r][c].scatter(np.arange(1, len(y_arousal_avg)+1), y_arousal_avg, marker = 's', s = mean_size, c = 'k', alpha = 0.0) # plot average
                    axarr[r][c].plot(np.arange(1, len(y_arousal_avg)+1), y_arousal_avg, ls = 'dotted', c = 'k') # plot average line
                    axarr[r][c].plot(x_index, y_pre, color='b', ls = 'dotted', linewidth = 2) # plot regression
                    axarr[r][c].set_xlabel('BFI score')
                    axarr[r][c].set_ylabel('Arousal')
                    axarr[r][c].axis([int(0), int(6), Min_label, Max_label])
                    axarr[r][c].set_title('{}: Arousal  r_squre = {}'.format(key.capitalize(), round(r_score, 4)), fontdict = {'fontsize': 12})

                f.suptitle('index {}: {}'.format(index, big5_statement), fontsize = 15)
                plt.savefig(outdir + 'BFI_index_{}'.format(index))
                plt.close()
            i += 5
            pbar.update(cur_index+1)
            cur_index += 1
    else:
        '''
        篩選不符合條件的worker
        對符合條件的重複打分的text求平均
        bad worker: 打分差距在2分以上（不含）的tweet的數目佔所有重複tweet數目的比例大於30%的worker
        計算重複text平均值：
        遍歷textid list找到重複text的下標，根據找到的下標對valence/arousal數組求均值
        '''
        # get bad worker list
        bad_workers = []
        for key, vals in agreement_filter.items():
            if vals['ratio'] >= 0.3:
                bad_workers.append(vals['workerid'])
        i = 1
        while i <= 41: # the number of records
            for j in xrange(0, 5): # big5 statements, 5 statements each record
                index = i + j # big5 statement index
                fig_dict = {'anger': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'joy': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'sadness': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'anticipation': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'fear': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'trust': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, \
                                'disgust': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, 'surprise': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}}
                for k in xrange(1, 6): # big5 score
                    sql = "SELECT {},{},{},`WorkerId`, `Input.statement{}` FROM ratings_radio_big5 WHERE `Input.sid{}` = {} AND `Answer.radio_s_{}` = '{}' AND `AssignmentStatus` = 'Approved'".format(sql_valence, sql_arousal, sql_textid, j+1, j+1, index, j+1, k)
                    cursor.execute(sql)
                    results = cursor.fetchall()
                    
                    for row in results:
                        worker = row[90] # worker id

                        # filter those records that don't meet the agreement
                        if worker in bad_workers:
                            continue

                        # get repeating texts indices tuples
                        indeces = get_repeating_indeces(row[60:89]) # 60-89: text id

                        # cal the average valence and arousal, storing in new lists
                        filtered_tuple = []
                        v_a_id = zip(row[0:29], row[30:59], row[60:89]) # keep valence, arousal and textid at the same time
                        for cur in xrange(0, len(v_a_id)):
                            if cur in zip(*indeces)[0]: # combine
                                second_cur = zip(*indeces)[0].index(cur)
                                valence = (float(zip(*v_a_id)[0][cur]) + float(zip(*v_a_id)[0][zip(*indeces)[1][second_cur]])) / 2.0
                                arousal = (float(zip(*v_a_id)[1][cur]) + float(zip(*v_a_id)[1][zip(*indeces)[1][second_cur]])) / 2.0
                                filtered_tuple.append((valence, arousal, zip(*v_a_id)[2][cur]))
                            elif cur in zip(*indeces)[1]: # neglect the repeating one
                                continue
                            else: # keep
                                filtered_tuple.append(v_a_id[cur])
                            
                        big5_statement = row[91]
                        # for v, a, textid in zip(row[0:29], row[30:59], row[60:89]): # get valence, arousal and textid, respectively
                        for v, a, textid in filtered_tuple: # get valence, arousal and textid from distinct tuples
                            emo = get_emo_by_id(textid)
                            V_list = fig_dict[emo][k].get('Valence', list())
                            A_list = fig_dict[emo][k].get('Arousal', list())
                            V_list.append(float(v))
                            A_list.append(float(a))
                            fig_dict[emo][k]['Valence'] = V_list
                            fig_dict[emo][k]['Arousal'] = A_list

                # plot scatters
                f, axarr = plt.subplots(4, 4, figsize = (fig_width, fig_height))
                for key, vals in fig_dict.items():
                    annotators    = []
                    x_index       = []
                    x_index_avg   = []
                    y_valence     = []
                    y_arousal     = []
                    y_valence_avg = []
                    y_arousal_avg = []

                    for s, val in vals.items():
                        if not val.has_key('Valence'):
                            annotators.append(0)
                            continue
                        annotators.append(len(val['Valence']))

                        x_index   += [s for x in range(len(val['Valence']))]
                        y_valence += val['Valence']
                        y_arousal += val['Arousal']
                        if len(val['Valence']):
                            x_index_avg.append(s)
                            y_valence_avg.append(np.mean(val['Valence']))
                            y_arousal_avg.append(np.mean(val['Arousal']))

                    # print('{} - {}'.format(index, y_valence_avg))

                    # plot valence
                    r, c = emo_location[key][0]
                    r_score = 0.
                    if len(x_index) > 3:
                        regr   = linear_model.LinearRegression()
                        regr_x = np.array([[x] for x in x_index])
                        regr_y = np.array([[x] for x in y_valence])
                        regr.fit(regr_x, regr_y)
                        y_pre   = regr.predict(regr_x)
                        r_score = regr.score(regr_x, regr_y)
                        axarr[r][c].plot(x_index, y_pre, color='b', ls = 'dotted', linewidth = 2) # plot regression

                    axarr[r][c].scatter(x_index, y_valence, marker = 'o', s = scatter_size, c = 'g', alpha = 0.6)
                    axarr[r][c].scatter(x_index_avg, y_valence_avg, marker = 's', s = mean_size, c = 'k', alpha = 0.0) # plot average
                    axarr[r][c].plot(x_index_avg, y_valence_avg, ls = 'dotted', c = 'k') # plot average line
                    axarr[r][c].set_xlabel('BFI score')
                    axarr[r][c].set_ylabel('Valence')
                    axarr[r][c].axis([int(0), int(6), Min_label, Max_label])
                    axarr[r][c].set_title('{}: Valence  r_squre = {}'.format(key.capitalize(), round(r_score, 4)), fontdict = {'fontsize': 12})


                    # plot arousal
                    r, c = emo_location[key][1]
                    r_score = 0.
                    if len(x_index) > 3:
                        regr   = linear_model.LinearRegression()
                        regr_x = np.array([[x] for x in x_index])
                        regr_y = np.array([[x] for x in y_arousal])
                        regr.fit(regr_x, regr_y)
                        r_score = regr.score(regr_x, regr_y)
                        y_pre   = regr.predict(regr_x)
                        axarr[r][c].plot(x_index, y_pre, color='b', ls = 'dotted', linewidth = 2) # plot regression
                    axarr[r][c].scatter(x_index, y_arousal, marker = 'o', s = scatter_size, c = 'r', alpha = 0.6)
                    axarr[r][c].scatter(x_index_avg, y_arousal_avg, marker = 's', s = mean_size, c = 'k', alpha = 0.0) # plot average
                    axarr[r][c].plot(x_index_avg, y_arousal_avg, ls = 'dotted', c = 'k') # plot average line
                    axarr[r][c].set_xlabel('BFI score')
                    axarr[r][c].set_ylabel('Arousal')
                    axarr[r][c].axis([int(0), int(6), Min_label, Max_label])
                    axarr[r][c].set_title('{}: Arousal  r_squre = {}'.format(key.capitalize(), round(r_score, 4)), fontdict = {'fontsize': 12})

                f.suptitle('index {}: {}'.format(index, big5_statement), fontsize = 15)
                plt.savefig(outdir + 'BFI_index_{}'.format(index))
                plt.close()
            i += 5
            pbar.update(cur_index+1)
            cur_index += 1
            
    pbar.finish()
    cursor.close()
    db.close()



# 九幅图，前八幅图为emotion，最后一幅图为big5分数的柱状图
# 根据big5 index的分数，相同分数的人为一类，每一个fig为这类人的emotion scatter
# 柱状图中的big5分数取这一类人的平均分
# post_info: used to get emotion
def plot_workers_scatter_by_big5_and_emo(post_info, outdir):

    Min_label = 0
    Max_label = 10

    def get_emo_by_id(textid):
        emo = ''
        for key, vals in post_info.items():
            if textid in vals['texts'].keys():
                return vals['texts'][textid]['emo']
        return emo

    try:
        db = MySQLdb.connect('140.114.77.11', 'pan', 'PxQAC4MZrRF89EZ6', db = 'pan', charset='utf8')
        cursor = db.cursor()
    except Exception as e:
        print(e)

    emo_location = {'anger': (0,0), 'joy': (0,1), 'sadness': (0,2), 'anticipation': (1,0), 'fear': (1,1), 'trust': (1,2), 'disgust': (2,0), 'surprise': (2,1)}

    sql_valence = ""
    sql_arousal = ""
    sql_textid = ""
    for i in xrange(1, 30+1):
        sql_valence += "`Answer.radio_{}_v`,".format(i)
        sql_arousal += "`Answer.radio_{}_a`,".format(i)
        sql_textid += "`Input.textid{}`,".format(i)
    sql_valence = sql_valence[0:len(sql_valence)-1] # remove the last comma
    sql_arousal = sql_arousal[0:len(sql_arousal)-1] # remove the last comma
    sql_textid = sql_textid[0:len(sql_textid)-1] # remove the last comma

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = 41).start()
    cur_index = 0
    i = 1
    while i <= 41:
        for j in xrange(0, 5): # big5 statements, 5 statements each record
            index = i + j # big5 statement index
            for k in xrange(1, 6): # big5 scores, ranging from 1 to 5
                sql = "SELECT {},{},{},`WorkerId`, `Input.statement{}` FROM ratings_radio_big5 WHERE `Input.sid{}` = {} AND `Answer.radio_s_{}` = '{}' AND `AssignmentStatus` = 'Approved'".format(sql_valence, sql_arousal, sql_textid, j+1, j+1, index, j+1, k)
                # print sql
                cursor.execute(sql)
                results = cursor.fetchall()
                f, axarr = plt.subplots(3, 3, figsize = (15, 15))
                rating_count = len(results)
                avg_ratings = {'anger': [0,0,0], 'joy': [0,0,0], 'sadness': [0,0,0], 'anticipation': [0,0,0], 'fear': [0,0,0], 'trust': [0,0,0], 'disgust': [0,0,0], 'surprise': [0,0,0]} # calculate the average rating of each emotion
                for row in results: # get the query results
                    big5_statement = row[91]
                    for v, a, textid in zip(row[0:29], row[30:59], row[60:89]): # get valence, arousal and textid, respectively
                        emo = get_emo_by_id(textid)
                        r, c = emo_location[emo]
                        axarr[r][c].scatter(float(v), float(a), marker = 'o', s = 70, c = 'r', alpha = 0.6, label = emo)
                        avg_ratings[emo][0] += float(v)
                        avg_ratings[emo][1] += float(a)
                        avg_ratings[emo][2] += 1.0
                # plot the average ratings
                for key, vals in avg_ratings.items():
                    r, c = emo_location[key]
                    axarr[r][c].set_xlabel('Valence')
                    axarr[r][c].set_ylabel('Arousal')
                    axarr[r][c].set_xticks(np.arange(0, 10), 1)
                    axarr[r][c].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
                    axarr[r][c].axis([Min_label, Max_label, Min_label, Max_label])
                    axarr[r][c].set_title(key)
                    if vals[2] > 1E-6:
                        axarr[r][c].scatter(vals[0]/vals[2], vals[1]/vals[2], marker = '*', s = 85, c = 'b', alpha = 0.7)
                        axarr[r][c].set_title('{} ({}, {})'.format(key, round(vals[0]/vals[2], 2), round(vals[1]/vals[2], 2)))
                # plot the bar chart with one bar
                axarr[2][2].set_xlabel('index {}: {}'.format(index, big5_statement))
                axarr[2][2].set_ylabel('big5 score')
                axarr[2][2].set_xticklabels([index], ha = 'center')
                axarr[2][2].set_ylim(0, 5)
                axarr[2][2].bar(left = [1], height = k, alpha = 0.6)
                f.suptitle('count: {}'.format(rating_count), fontsize = 20)

                plt.savefig(outdir + 'big5_index_{}_score_{}.png'.format(index, k))
                plt.clf()

        i += 5
        pbar.update(cur_index+1)
        cur_index += 1
    pbar.finish()
    cursor.close()
    db.close()
    

def plot_workers_scatter_bar_big5_by_emo(key, texts, big5, worker_id, outdir):

    emo_location = {'anger': (0,0), 'joy': (0,1), 'sadness': (0,2), 'anticipation': (1,0), 'fear': (1,1), 'trust': (1,2), 'disgust': (2,0), 'surprise': (2,1)}

    x = []
    y = []
    emo = []
    for k, v in texts.items():
        x.append(v['ratings']['Valence'])
        y.append(v['ratings']['Arousal'])
        emo.append(v['emo'])

    Min_label = 0
    Max_label = 10

    f, axarr = plt.subplots(3, 3, figsize = (15, 15))

    for i in xrange(0, len(x)):
        row, col = emo_location[emo[i]]

        # axarr[0, 0].set_title("{} workerID: {} {} texts".format(emo, index, len(vals['post_info'])))
        axarr[row][col].set_title(emo[i])
        axarr[row][col].set_xlabel('Valence')
        axarr[row][col].set_ylabel('Arousal')
        axarr[row][col].set_xticks(np.arange(0, 10), 1)
        axarr[row][col].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
        axarr[row][col].axis([Min_label, Max_label, Min_label, Max_label])
        p = axarr[row][col].scatter(x[i], y[i], marker = 'o', s = 70, c = 'r' ,alpha = 0.7, label = emo[i])
    
    axarr[2][2].set_xlabel('big5 index')
    axarr[2][2].set_ylabel('big5 score')
    axarr[2][2].set_xticklabels([str(k) for k in big5.keys()], ha = 'center')
    axarr[2][2].set_ylim(0, 5)
    p = axarr[2][2].bar(left = np.arange(len(big5.values())), height = big5.values(), alpha = 0.6)


    plt.savefig(outdir + '{}_worker_{}.png'.format(key, worker_id))
    plt.clf()

    return (multiprocessing.current_process().name)


# big5: {big5_id: score}
def plot_workers_scatter_bar_big5(key, texts, big5, worker_id, outdir):
    # print ('{}'.format(multiprocessing.current_process().name))

    import matplotlib.pyplot as plt

    x = []
    y = []
    emo = []
    for k, v in texts.items():
        x.append(v['ratings']['Valence'])
        y.append(v['ratings']['Arousal'])
        emo.append(v['emo'])

    Min_label = 0
    Max_label = 10

    f, axarr = plt.subplots(1, 2, figsize = (25, 10))

    # axarr[0, 0].set_title("{} workerID: {} {} texts".format(emo, index, len(vals['post_info'])))
    axarr[0].set_xlabel('Valence')
    axarr[0].set_ylabel('Arousal')
    axarr[0].set_xticks(np.arange(0, 10), 1)
    axarr[0].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
    axarr[0].axis([Min_label, Max_label, Min_label, Max_label])
    for i in xrange(0, len(x)):
        p = axarr[0].scatter(x[i], y[i], marker = EMO_MARKER[emo[i]], s = 70, c = 'r' ,alpha = 0.7, label = emo[i])

    handles, labels = axarr[0].get_legend_handles_labels()
    legend_ins = dict(zip(labels, handles))
    handles = legend_ins.values()
    labels = legend_ins.keys()
    axarr[0].legend(handles, labels, loc = 'best', scatterpoints = 1)
    
    axarr[1].set_xlabel('big5 index')
    axarr[1].set_ylabel('big5 score')
    axarr[1].set_xticklabels([str(k) for k in big5.keys()], ha = 'center')
    axarr[1].set_ylim(0, 5)
    
    p = axarr[1].bar(left = np.arange(len(big5.values())), height = big5.values(), alpha = 0.6)


    plt.savefig(outdir + '{}_worker_{}.png'.format(key, worker_id))
    plt.clf()

    return (multiprocessing.current_process().name)


def plot_scatter_big5_bar_by_assign(postinfo_dict_big5, outdir):
    Min_label = 0
    Max_label = 10
    cpu       = 8
    pool      = multiprocessing.Pool(processes = cpu)
    results   = []
    interval  = int(len(postinfo_dict_big5.keys()) / cpu)
    ids       = postinfo_dict_big5.keys()

    # for i in xrange(1, cpu):
    #   assign_ids = ids[(i*interval):(i*interval+interval)]
    #   results.append(pool.apply_async(plot_workers_scatter_bar_big5, (postinfo_dict_big5, assign_ids, outdir)))
    # assign_ids = ids[((cpu-1)*interval+interval):]
    # results.append(pool.apply_async(plot_workers_scatter_bar_big5, (postinfo_dict_big5, assign_ids, outdir)))

    # pool.close()
    # pool.join()

    for key, vals in postinfo_dict_big5.items():
        big5 = vals['big5']
        texts = vals['texts']
        worker_id = vals['worker']['worker_id']

        # x = []
        # y = []
        # for k, v in texts.items():
        #   x.append(v['ratings']['Valence'])
        #   y.append(v['ratings']['Arousal'])

        # results.append(pool.apply_async(plot_workers_scatter_bar_big5, (key, texts, big5, worker_id, outdir)))
        results.append(pool.apply_async(plot_workers_scatter_bar_big5_by_emo, (key, texts, big5, worker_id, outdir)))

        # f, axarr = plt.subplots(1, 2, figsize = (25, 10))

        # # axarr[0, 0].set_title("{} workerID: {} {} texts".format(emo, index, len(vals['post_info'])))
        # axarr[0].set_xlabel('Valence')
        # axarr[0].set_ylabel('Arousal')
        # axarr[0].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
        # axarr[0].axis([Min_label, Max_label, Min_label, Max_label])
        
        # axarr[1].set_xlabel('big5 index')
        # axarr[1].set_ylabel('big5 score')
        # axarr[1].set_xticks(map(lambda x: x+1, np.arange(len(big5.values()))))
        # p = axarr[0].scatter(x, y, marker = 'o', alpha = 0.8)
        # p = axarr[1].bar(left = np.arange(len(big5.values())), height = big5.values(), alpha = 0.6)

        # plt.savefig(outdir + '{}_worker_{}.png'.format(key, worker_id))
        # plt.clf()

    pool.close()
    pool.join()

    for res in results:
        print (res.get()[0])


def plot_GMM_ellipses(gmm, ax):
    colors = ['navy', 'turquoise', 'darkorange', '#E24A33', '#6d904f', '#00FFCC', '#b3de69', '#FF9F9A']
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def GMM_cluster(postinfo_dict, n_components = 8, init_params = 'kmeans', weights = False):

    if not weights:
        weights_init = None

    colors = ['navy', 'turquoise', 'darkorange', '#E24A33', '#6d904f', '#00FFCC', '#b3de69', '#FF9F9A']

    X = []
    for key, val in postinfo_dict.items():
        xy = zip(val['Valence'], val['Arousal'])
        for x in xy:
            X.append(list(x))
        
    X = np.array(X)

    estimators = dict((cov_type, mixture.GaussianMixture(n_components = n_components,
                   covariance_type = cov_type, weights_init = weights_init, init_params = init_params))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])


    n_estimators = len(estimators)
    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(bottom = .01, top = 0.95, hspace = .15, wspace = .05, left = .01, right = .99)

    for index, (name, estimator) in enumerate(estimators.items()):
        
        estimator.fit(X)

        h = plt.subplot(2, n_estimators // 2, index + 1)
        plot_GMM_ellipses(estimator, h)

        for n, color in enumerate(colors):
            plt.scatter(X[:, 0], X[:, 1], s = 0.8, color = color)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.show()
    

    # gmm = mixture.GaussianMixture(n_components = n_components, covariance_type='full', weights_init = weights_init, init_params = init_params)
    # gmm.fit(X)

                    

'''
Outliers Detection
'''

'''
return centroids of each cluster, sorted by ascend of the cluster number
'''
def cal_centroids(X, labels):
    cluster_num = len(set(labels))
    centroids = []
    for i in xrange(0, cluster_num):
        index = [j for j, k in enumerate(labels) if k == i] # get the index of specified value

        cluster_points = [X[m] for m in index]
        x = reduce(lambda x, y: x+y, map(lambda x: x[0], cluster_points)) / float(len(index))
        y = reduce(lambda x, y: x+y, map(lambda x: x[1], cluster_points)) / float(len(index))
        centroids.append((x, y))

    return centroids


'''
detect outliers
return outlier list and point list without outliers
'''
def get_dist_to_cent(X, labels, X_true_cluster = -1, cluster_centers = None, inertia = None):
    cluster_num = len(set(labels))
    dist = []

    # get cluster centroids
    if cluster_centers == None:
        centroids = cal_centroids(X, labels)
    else:
        centroids = cluster_centers

    if cluster_num <= 1: # if only have one cluster, calculate the distances between each point and centroid
        dist = map(lambda x: np.sqrt((x[0]-centroids[0][0])**2 + (x[1]-centroids[0][1])**2), labels)
    # find the cluter with less points (minor cluster) and delete k points that farest away from major cluster centtroid
    # if it has true values, set the cluster with trues values as standard
    elif cluster_num == 2:
        if X_true_cluster >= 0: # get the cluster number that contains true value
            major = X_true_cluster
        else:
            major = 0 if list(labels).count(0) > list(labels).count(1) else 1 # find major cluster

        # cal the distance between labels and major centroid
        minor = [i for i, j in enumerate(labels) if j != major] # get points index corresponding to minor cluster
        minor_X = [j for i, j in enumerate(X) if i in minor] # get points corresponding to minor cluster
        dist = map(lambda x: np.sqrt((x[0]-centroids[major][0])**2 + (x[1]-centroids[major][1])**2), minor_X)
    else:
        print ('There are more than 2 clusters!')
        # find the cluster with the most points
        if X_true_cluster >= 0: # get the cluster number that contains true value
            major = X_true_cluster
        else:
            major = 0
            for i in xrange(0, cluster_num):
                if major < list(labels).count(i):
                    major = i

        # cal the distance between labels and major centroid
        minor = [i for i, j in enumerate(labels) if j != major] # get points index corresponding to minor cluster
        minor_X = [j for i, j in enumerate(X) if i in minor] # get points corresponding to minor cluster
        dist = map(lambda x: np.sqrt((x[0]-centroids[major][0])**2 + (x[1]-centroids[major][1])**2), minor_X)

    return dist


'''
return labels of each point and cluster centers
'''
def k_means_cluster(X, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters).fit(X)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_


'''
agglomerative cluster
return labels of each point
'''
def agg_cluster(X, n_clusters, affinity = 'euclidean'):
    model = AgglomerativeClustering(n_clusters = n_clusters, affinity = affinity).fit(X)
    return model.labels_


'''
Detect outlier
return points list withou outliers and outlier list
'''
# CHANGE THE ALGO
# It should run clustering again after deleting a point!
def detect_outlier(postinfo_dict, method = 'kmeans', n_delete = 2):
    n_clusters = 2
    post_dict = wordProcBase.get_fb_post_dict()
    new_postinfo_dict = {} # the dict to be returned
    proc_postinfo_dict = {} # dict to be proceeded in the iterations

    # initialize the proc_postinfo_dict
    proc_postinfo_dict = copy.deepcopy(postinfo_dict)

    for i in xrange(0, n_delete):

        for key, val in proc_postinfo_dict.items():
            X = zip(val['Valence'], val['Arousal'])
            X_true_cluster = -1  # the cluster index to which the true value belongs
            cluster_centers = None
            deletes = []

            if key[0] == 'A':
                X_true = (float(post_dict[key[1:]]['Valence1']) + float(post_dict[key[1:]]['Valence2']) / 2.0, (float(post_dict[key[1:]]['Arousal1']) + float(post_dict[key[1:]]['Arousal2']) / 2.0))
                X.append(X_true) # add true valence and arousal to the last
     
            if method == 'kmeans':
                labels, cluster_centers, inertia = k_means_cluster(X, n_clusters)
            else:
                labels = agg_cluster(X, n_clusters)

            # if ture values are added, delete the true value
            if key[0] == 'A':
                X_true_cluster = labels[-1] # if X contians true value, the last label is the true value label
                X.pop()
                labels = np.delete(labels, -1)

            # delete the point that farest away from the centroid
            dist = get_dist_to_cent(X, labels, X_true_cluster, cluster_centers)
            index = dist.index(max(dist))
            deletes.append(dist.pop(index)) # delete the distance
            X.pop(index) # delete the points that with max distance to centroids

            # update proc_postinfo_dict
            proc_postinfo_dict[key]['Valence'] = [x for x in zip(*X)[0]]
            proc_postinfo_dict[key]['Arousal'] = [x for x in zip(*X)[1]]

    # generate new dict to be returned
    new_postinfo_dict = copy.deepcopy(proc_postinfo_dict)

    return new_postinfo_dict


def detect_outlier_by_worker(exclude_workers, worker_dict):
    new_postinfo_dict = {} # key is workerID
    for key, val in worker_dict.items():
        if key in exclude_workers:
            continue

        for v in val['post_info']:
            VA_dict = new_postinfo_dict.get(v[0], dict())
            valence = VA_dict.get('Valence', list())
            arousal = VA_dict.get('Arousal', list())
            valence.append(v[1])
            arousal.append(v[2])
            VA_dict['Valence'] = valence
            VA_dict['Arousal'] = arousal
            new_postinfo_dict[v[0]] = VA_dict

    # pprint.pprint(new_postinfo_dict)

    return new_postinfo_dict


'''
1. cal the center of each text
2. delete the point that farest away from the center
return the new postinfo_dict
'''
def detect_outlier_by_distance(postinfo_dict, n_delete):
    new_postinfo_dict = copy.deepcopy(postinfo_dict)
    for i in range(n_delete):
        for key, vals in new_postinfo_dict.items():
            # print('std Valence: {}, std Arousal: {}'.format(np.std(vals['Valence']), np.std(vals['Arousal'])))
            points = zip(vals['Valence'], vals['Arousal'])
            center = (np.mean(vals['Valence']), np.mean(vals['Arousal']))
            distance = [(x[0], math.sqrt((x[1][0]-center[0])**2+(x[1][1]-center[1])**2)) for x in enumerate(points)] # get distance with index for each point
            index = max(distance, key = operator.itemgetter(1))[0] # get the index of the max distance point
            del vals['Valence'][index] # delete the point
            del vals['Arousal'][index]
            # print('std Valence: {}, std Arousal: {}\n'.format(np.std(vals['Valence']), np.std(vals['Arousal'])))
    return new_postinfo_dict

# Chebyshev's inequality
def detect_outlier_by_std(postinfo_dict, nstd = 2):
    new_postinfo_dict = {}
    annotators = 30
    for key, vals in postinfo_dict.items():
        count = 0
        new_postinfo_dict[key] = {'Valence': [], 'Arousal': []}
        points = zip(vals['Valence'], vals['Arousal'])
        center = (np.mean(vals['Valence']), np.mean(vals['Arousal']))
        # center = geometric_median.geometric_median(np.array([point for point in zip(vals['Valence'], vals['Arousal'])]))
        std = (np.std(vals['Valence']), np.std(vals['Arousal']))
        for i in range(len(vals['Valence'])):
            v = vals['Valence'][i]
            a = vals['Arousal'][i]
            if v < center[0]+std[0]*nstd and a < center[1]+std[1]*nstd:
                new_postinfo_dict[key]['Valence'].append(v)
                new_postinfo_dict[key]['Arousal'].append(a)
                count += 1
        # print('{} deleted: {}'.format(key, annotators - count))
    return new_postinfo_dict

def detect_outlier_by_2D_std(postinfo_dict, nstd = 2):
    new_postinfo_dict = {}
    annotators = 30
    for key, vals in postinfo_dict.items():
        count = 0
        new_postinfo_dict[key] = {'Valence': [], 'Arousal': []}
        point = zip(vals['Valence'], vals['Arousal'])
        center = (np.mean(vals['Valence']), np.mean(vals['Arousal']))
        dist_std = [math.sqrt((p[0] - center[0])**2 + (p[1] - center[1])**2) for p in point]
        dist_std = np.mean(dist_std)
        for i in range(len(vals['Valence'])):
            v = vals['Valence'][i]
            a = vals['Arousal'][i]
            if math.sqrt((v - center[0])**2 + (a - center[1])**2) < nstd*dist_std:
                new_postinfo_dict[key]['Valence'].append(v)
                new_postinfo_dict[key]['Arousal'].append(a)
                count += 1
        # print('{} deleted: {}'.format(key, annotators - count))
    return new_postinfo_dict



'''Cal the geometric median

Cal each text's geometric median and mean
'''
def plot_geo_median_scatter(postinfo_dict, outdir):
    Max_label = 10
    Min_label = 0
    savefile  = outdir

    for idx, vals in postinfo_dict.items():
        valence    = vals['Valence']
        arousal    = vals['Arousal']
        mean_v     = np.mean(valence)
        mean_a     = np.mean(arousal)
        
        geo_median = geometric_median.geometric_median(np.array([point for point in zip(valence, arousal)]))

        fig        = plt.figure(figsize=(10, 10))

        plt.scatter(valence, arousal, c = 'b', marker = 'o', s = 35, edgecolors = 'face', linewidths = 0.0, alpha = 0.7)
        plt.scatter(mean_v, mean_a, marker = '*', s = 45, edgecolors = 'face', linewidths = 0.0)
        plt.scatter(geo_median[0], geo_median[1], marker = 'x', s = 45, edgecolors = 'face', linewidths = 0.0)


        plt.ylim([0, 10])
        plt.xlim([0, 10])
        plt.ylabel('Arousal')  
        plt.xlabel('Valence')
        plt.plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
        plt.savefig(savefile + '{}.png'.format(idx))
        plt.close()


'''
calculate each sample's distance weight
infile: /home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/results/turk_label_ABC_ordered.csv
'''
def cal_annotator_prior(infile, outfile, method = 'median'):
    id_list = []
    with open(outfile, 'w') as csvout:
        spamwriter = csv.writer(csvout, delimiter = ',')

        with open(infile, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter = ',')
            for row in spamreader:
                user_num = int((len(row) - 1) / 2)
                id_list.append(row[0])
                valence = [float(x) for x in row[1:user_num+1]]
                arousal = [float(x) for x in row[user_num+1:]]

                avg = geometric_median.geometric_median(np.array([point for point in zip(valence, arousal)]))
                # avg = (np.mean(valence), np.mean(arousal))

                prior = []
                for x, y in zip(valence, arousal):
                    # prior.append(1.0 / math.sqrt((x-avg[0])**2 + (y-avg[1])**2))
                    prior.append(math.exp(-math.sqrt((x-avg[0])**2 + (y-avg[1])**2)))

                user_prior = [x / sum(prior) for x in prior]

                for x in user_prior:
                    spamwriter.writerow([x])


def avg_dist(x, y):
    dist = []
    count = 0
    for p1, p2 in combinations(zip(x, y), 2):
        dist.append(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
    dist_mean = np.mean(dist)
    dist_std = np.std(dist)
    return dist_mean, dist_std


def cal_avg_dist(postinfo_dict):

    all_dist = []
    for key, vals in postinfo_dict.items():
        valence = vals['Valence']
        arousal = vals['Arousal']
        dist = 0.
        count = 0
        for p1, p2 in combinations(zip(valence, arousal), 2):
            count += 1
            dist += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        all_dist.append(dist / count)

    return (np.mean(all_dist), np.std(all_dist))

def cal_avg_dist_by_emo(postinfo_dict):
    emo_post = get_emo_result2()

    for emo in EMO_LIST:
        id_list = emo_post[emo]
        # colors = zip([i/1000. for i in random.sample(xrange(1000), len(id_list))], [i/1000. for i in random.sample(xrange(1000), len(id_list))], [i/1000. for i in random.sample(xrange(1000), len(id_list))])
        index = 0
        fig = plt.figure(figsize=(10, 10))
        x = []
        y = []
        for i in id_list:
            x.extend(postinfo_dict[i]['Valence'])
            y.extend(postinfo_dict[i]['Arousal'])

        dist = 0.
        count = 0
        for p1, p2 in combinations(zip(x, y), 2):
            count += 1
            dist += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        print((emo, np.mean(dist/count)))


def cal_std(infile, postinfo_dict):
    emo_post = []
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',')
        header = next(spamreader)
        for row in spamreader:
            emo_post.append(row)

    for i, post in enumerate(emo_post):
        idx = post[1]
        valence = postinfo_dict[idx]['Valence']
        arousal = postinfo_dict[idx]['Arousal']
        dist_mean, dist_std = avg_dist(valence, arousal)
        emo_post[i].extend([dist_mean, dist_std])

    with open(infile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',')
        header.extend(['dist_mean', 'dist_std'])
        spamwriter.writerow(header)
        for row in emo_post:
            spamwriter.writerow(row)


def download_file_for_multivariate_test(postinfo_dict, outdir):
    for key, vals in postinfo_dict.items():
        with open(outdir + key + '.csv', 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = ',')
            spamwriter.writerow(['valence', 'arousal'])
            for v, a in zip(vals['Valence'], vals['Arousal']):
                spamwriter.writerow([v, a])




if __name__ == '__main__':

    '''
    Get basic data and infomation
    '''
    '''
    files without big5 processing
    '''
    batch_result_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/Batch_2549923_batch_results.csv'
    stat_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/stat.csv'
    postinfo_dict, worker_dict = get_batch_result(batch_result_file)
    # post_stat_dict = get_post_statistics(postinfo_dict, stat_file)

    # plot_scatter(postinfo_dict)

    # new_postinfo_dict = detect_outlier(postinfo_dict, '', 5)
    # new_postinfo_dict = detect_outlier_by_worker(BAD_WORKERS, worker_dict)
    # new_postinfo_dict = detect_outlier_by_distance(postinfo_dict, 5)
    # new_postinfo_dict = detect_outlier_by_std(postinfo_dict)
    new_postinfo_dict = detect_outlier_by_2D_std(postinfo_dict)

    outfile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/turk_label_ABC_ordered_outlier_by_dist_std.csv'
    # arrange_post(new_postinfo_dict, outfile)

    outdir = '/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/multivariate test/'
    # download_file_for_multivariate_test(new_postinfo_dict, outdir)

    outdir = '/home/pan/Idealab/NTHU/Master thesis/'
    # plot_scatter_by_emotion_as_whole(new_postinfo_dict, outdir)

    # for key, val in postinfo_dict.items():
    #   org_std_v = np.std(val['Valence'])
    #   org_std_a = np.std(val['Arousal'])
    #   new_std_v = np.std(new_postinfo_dict[key]['Valence'])
    #   new_std_a = np.std(new_postinfo_dict[key]['Arousal'])
    #   print ('Valence: {}, Arousal: {}'.format((round(org_std_v, 3), round(new_std_v, 3)), (round(org_std_a, 3), round(new_std_a, 3))))

    '''
    Plotting
    '''
    infile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/Batch_2549923_batch_results_for_python.csv'
    # plot_scatter_by_emotion(postinfo_dict)
    # plot_workers(worker_dict, infile, True)
    # plot_workers_emotion(worker_dict, infile, True)

    # infile = '/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score_for_ploteachtext.csv'
    # plot_scatter_by_text(new_postinfo_dict, infile, outdir = '/home/pan/Idealab/NTHU/Master thesis/scatter text/')

    # infile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/manyemo/turk_survey_data_ABC_score_299emos_proc.csv'
    # rearrange_emo(infile)

    # scorefile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score_299emos_scorefile.csv'
    # outfile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/manyemo/turk_survey_data_ABC_score_12emos.csv'
    # get_subemoscore(scorefile, outfile)

    # plot_text_avg_worker_by_emotion(postinfo_dict)

    # plot_text_avg_worker_by_text_2(postinfo_dict, worker_dict)
    # plot_workers_emotion_scatter(postinfo_dict, worker_dict)
    # plot_worker_emotion_bar(postinfo_dict, worker_dict)

    # GMM_cluster(postinfo_dict)

    cluster_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score_kmeans_with_VA_15_combined_filtered.csv'
    outdir       = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/clusters_kmeans_VA/'
    # plot_scatter_by_cluster(postinfo_dict, cluster_file, outdir)


    '''
    files with big5 processing
    '''
    batch_result_file_big5 = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results_new/Batch_2640330_batch_results.csv'
    postinfo_dict_big5, worker_dict_big5 = get_batch_result_big5(batch_result_file_big5)
    
    # infile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score_for_ploteachtext.csv'
    # outdir = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results_new/figs/ids/'
    # plot_scatter_by_text(postinfo_dict_big5, infile, outdir)

    outdir = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results_new/figs/worker_big5_emo/'
    # postinfo_dict_big5 = get_batch_result_big5_all(batch_result_file_big5)
    # plot_scatter_big5_bar_by_assign(postinfo_dict_big5, outdir)

    # pprint.pprint(postinfo_dict_big5)

    # write big5 turkers' ratings
    outfile = '/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/results_new/scores.csv'
    # arrange_post_big5(postinfo_dict_big5, outfile)

    outdir = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results_new/figs/worker_big5_emo_by_score/'
    # plot_workers_scatter_by_big5_and_emo(postinfo_dict_big5, outdir)

    # pprint.pprint(get_worker_reliability(2))

    # bad_workers = get_agreement()
    # pprint.pprint(bad_workers)
    outdir = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results_new/figs/worker_big5_by_emo_avg/'
    # plot_workers_scatter_by_big5_score(postinfo_dict_big5, outdir, bad_workers)

    outdir = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results_new/figs/worker_big5_by_index_VA/'
    plot_scatter_big5_index_VA(postinfo_dict_big5, outdir, bad_workers)

    outdir = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/figs/id_geo_median/'
    # plot_geo_median_scatter(postinfo_dict, outdir)

    infile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/turk_label_ABC_ordered.csv'
    outfile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/exp_dist_geo_prior_ordered.csv'
    # cal_annotator_prior(infile, outfile)

    # print(cal_avg_dist(new_postinfo_dict))
    # cal_avg_dist_by_emo(postinfo_dict)

    infile = '/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/results/PWKL_emo.csv'
    # cal_std(infile, postinfo_dict)


    # emo_post = get_emo_result2()
    # for emo, id_list in emo_post.items():
    #     with open('/home/pan/Idealab/stability/{}_VA.csv'.format(emo), 'w') as csvfile:
    #         spamwriter = csv.writer(csvfile, delimiter = ',')
    #         for idx in id_list:
    #             for vals in zip(postinfo_dict[idx]['Valence'], postinfo_dict[idx]['Arousal']):
    #                 spamwriter.writerow([vals[0], vals[1]])


