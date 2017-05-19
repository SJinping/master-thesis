import csv
import sys
import platform
import pprint
import math
import numpy as np
from collections import defaultdict
from itertools import combinations

_MAIN_DIR_ = ''
_system_ = platform.system()
if _system_ == 'Darwin':
    _MAIN_DIR_ = '/Users/Pan/Idealab'
elif _system_ == 'Linux':
    _MAIN_DIR_ = '/home/pan/Idealab'

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



# Calculate the KL divergence of two single Gaussians
def GaussianKL(mu1, mu2, Sigma1, Sigma2):
    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)
    return 0.5 * (np.trace(np.dot(Sigma2.I, Sigma1)) + \
        np.dot(np.dot((mu1 - mu2), Sigma2.I), (mu1-mu2).T) + \
        math.log(det_Sigma2/det_Sigma1) - 2)

# f: predicted distribution
# g: groud truth distribution
def KL_diver(f, g):
    mu1 = np.mat([np.mean(f[0]), np.mean(f[1])])
    mu2 = np.mat([np.mean(g[0]), np.mean(g[1])])
    Sigma1 = np.mat(np.cov(f))
    Sigma2 = np.mat(np.cov(g))
    return GaussianKL(mu1, mu2, Sigma1, Sigma2)

# KL for each rated valence and arousal pair
# aims to indicate the divergence of the ratings
def get_KL(infile, post_info):
    id_cat = defaultdict(list)
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',')
        header = next(spamreader)
        for row in spamreader:
            id_cat[row[1]].append(row[0])

    PWKL = {}
    for key, vals in id_cat.items():
        count = 0
        KL = 0
        for idx1, idx2 in combinations(vals, 2):
            count += 1
            valence_f = post_info[idx1]['Valence']
            arousal_f = post_info[idx1]['Arousal']
            valence_g = post_info[idx2]['Valence']
            arousal_g = post_info[idx2]['Arousal']
            f = np.array([valence_f, arousal_f])
            g = np.array([valence_g, arousal_g])
            KL += KL_diver(f, g)

        PWKL[key] = KL / count if count > 0 else 99999

    return PWKL

def get_KL_by_emotion(infile, emotion_file):
    emotions = defaultdict(list)
    KL = defaultdict()
    ED = defaultdict()
    Likelihood = defaultdict()
    with open(emotion_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            emotions[row[3]].append(row[0])

    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',')
        header = next(spamreader)
        for row in spamreader:
            KL[row[0]] = float(row[2])
            ED[row[0]] = float(row[3])
            Likelihood[row[0]] = float(row[4])

    for key, vals in emotions.items():
        AKL = [KL[i] for i in vals]
        AED = [ED[i] for i in vals]
        ALikelihood = [Likelihood[i] for i in vals]
        print('{}  AKL: {}, AED: {}, ALikelihood: {}'.format(key, round(np.mean(AKL), 4), round(np.mean(AED),4), round(np.mean(ALikelihood), 4)))
 

if __name__ == '__main__':
    # batch_result_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/results/Batch_2549923_batch_results.csv'
    # infile = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/Clusters/turk_survey_data_ABC_score_LDA_8_combined_filtered.csv'
    # postinfo_dict, worker_dict = get_batch_result(batch_result_file)
    # PWKL = get_KL(infile, postinfo_dict)
    # pprint.pprint(PWKL)

    infile = _MAIN_DIR_ + '/AEGTools/Text/validation_cnn_outlier/individual_output_CNN_outlier.csv'
    emotion_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score.csv'
    print('CNN')
    get_KL_by_emotion(infile, emotion_file)

