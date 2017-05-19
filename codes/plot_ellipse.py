import csv
import numpy as np
import sklearn
import math
import sys
import platform
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
from numpy.linalg import cholesky
from collections import defaultdict

_MAIN_DIR_ = ''
_system_ = platform.system()
if _system_ == 'Darwin':
    _MAIN_DIR_ = '/Users/Pan/Idealab'
elif _system_ == 'Linux':
    _MAIN_DIR_ = '/home/pan/Idealab'

sys.path.append(_MAIN_DIR_ + "/codes/plot_ellipse")
import gaussian_ellipse

def get_ellipse_data(infile):
    ellipse_data = defaultdict(defaultdict)
    with open(infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            ellipse_data[row[0]]['mean_x'] = float(row[5])
            ellipse_data[row[0]]['mean_y'] = float(row[6])
            ellipse_data[row[0]]['cov_x']  = float(row[7])
            ellipse_data[row[0]]['cov_y']  = float(row[8])
            ellipse_data[row[0]]['cov_xy'] = float(row[9])
    return ellipse_data


def plot_predict_ellipse_by_emotion(emo_file, data_file, outdir):
    emo_ids = defaultdict(list)
    with open(emo_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        header = next(spamreader)
        for row in spamreader:
            emo_ids[row[3]].append(row[0])

    ellipse_data = get_ellipse_data(data_file)
    
    kwrg         = {'edgecolor':'k', 'linewidth':0.6}
    Max_label    = 9
    Min_label    = 1

    for key, vals in emo_ids.items():
        fig = plt.figure(figsize=(5, 5))
        plt.plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k', linewidth = 0.5)
        for idx in vals:
            cov = np.array([[ellipse_data[idx]['cov_x'], ellipse_data[idx]['cov_xy']], [ellipse_data[idx]['cov_xy'], ellipse_data[idx]['cov_y']]])
            pos = np.array([ellipse_data[idx]['mean_x'], ellipse_data[idx]['mean_y']])
            plt.scatter(pos[0], pos[1], marker = 'o', c = 'r', s = 30, alpha = 0.5)
            gaussian_ellipse.plot_cov_ellipse(cov, pos, nstd = 1, alpha = 0.7, fill = False, **kwrg)
        print(key)
        # plt.show()
        plt.savefig(outdir +'EC_' + key +'.png', bbox_inches='tight')
        plt.close()

# covarance: (x, y, xy)
def plot_GMM_modeling_ellipse(outdir):
    GMM_sp = {'EC':{'mu': [(6.0993, 3.8252, 2.4539, 7.2055, 3.4726, 3.0073, 6.9598, 7.6250), (4.9331, 6.2736,2.9492,5.7133,4.4421,6.0294,6.606,7.0708)], 
                 'cov': [(2.1187, 2.3512, -0.9091), (2.8826, 2.6792, -2.1801), (0.9455, 1.0961, 0.7841), (1.2555, 2.7837, 0.2325), 
                         (2.4188, 3.7102, 0.2950), (1.1579, 2.9428, -0.7810), (1.7181, 1.4158, 1.1282), (1.0728, 1.4858, 1.0549)]}, 
           'RF': {'mu': [(7.4681,5.9486,2.9569,4.6614,2.9865,2.6225,6.1511,7.7665), (5.9636,4.7564,3.34241,5.5569,5.894,5.2682,6.0494,7.209)], 
                  'cov': [(1.0682, 2.5022, -0.0249), (1.4930, 3.6211, 0.1181), (1.6482, 1.5392, 1.3074), (3.0779, 2.9406, -2.6252), 
                          (1.4308, 3.4491, -1), (0.8665, 4.3765, 0.1760), (0.3915, 0.3527, 0.3149), (0.8346, 1.2884, 0.8596)]}, 
            'CNN': {'mu': [(6.5542,4.083,4.2346,6.1864,2.4919,3.0764,6.9426,6.957), (5.74,5.2411,4.9197,5.9201,5.8114,3.1751,6.6684,6.2517)], 
                  'cov': [(3.6909, 3.4743, 1.0402), (3.7196, 3.6936, -0.4777), (4.4531, 3.9842, 1.1656), (3.9214, 2.8423, 2.0440), 
                          (0.8551, 4.4414, -0.1670), (3.1574, 2.0337, 2.0585), (1.9011, 1.7661, 1.6486), (2.3234, 2.6112, 1.1888)]}, 
            'NB': {'mu': [(7.6456487955,4.2618113992,3.5641895464,6.1101990398,2.8822112401,2.6001490296,6.2034454239,7.8687929036), (5.9856827793,5.7334504022,3.8039572393,5.2083477383,5.8671703559,4.5266140617,5.9976072374,7.3787615215)], 
                   'cov': [(0.9199, 2.5857, 0.2949), (3.0033, 3.1403, -2.9625), (2.4248, 2.0883, 2.0700), (1.6566, 2.9549, -0.6965), 
                          (1.4131, 4.1345, -1.4231), (0.8408, 4.2789, 1.3229), (0.4063, 0.3172, 0.2703), (0.7229, 1.1905, 0.8166)]}}
    GMM = {'K8': {'mu':[(2.12410820378871, 5.07044858548973, 8.03800390218534, 5.11661350181792, 5.07926260711300, 4.81514439133074, 6.63696432136261, 3.84788802297019), 
                             (2.49759650169733, 4.95263262708440, 7.62639849758654, 5.14606733510362, 5.40512580457496, 5.56088591924007, 6.14402001213801, 6.29076115857237)],
                       'cov': [(0.6769, 0.9473, 0.7266), (4.628, 3.2857, 1.0705), (0.6671, 1.1339, 0.7845), (4.8957, 3.7483, 1.187), 
                               (5.2444, 3.7483, 1.187), (6.2234, 3.784, 1.0298), (2.0313, 1.1912, 1.307), (3.9244, 3.8233, -3.5352)]},
                'LDA8': {'mu': [(5.43173938353078, 5.50614556727446, 6.21469960503113, 5.35610661813136, 5.52285474763824, 4.84981512253809, 4.39973634723678, 3.44786332505724), 
                                (5.93789597445698, 5.61156341243733, 5.97325098285176, 5.28408125928526, 5.36541163277992, 5.44464792274123, 5.18231601832514, 4.49694709169587)],
                         'cov':[(6.8311, 4.0253, 2.2595), (5.742, 3.9762, -0.5179), (4.6572, 3.8977, 3.8941), (5.63, 4.0595, 2.1965),
                                (5.1282, 3.7611, 2.2061), (5.5653, 4.2623, 1.2194), (5.9415, 4.9916, 1.3437), (2.7025, 4.6774, 0.6271)]},
                'K14': {'mu': [(8.11696014714374, 4.90605080346294, 5.27748969875437, 4.89847068229426, 4.93400973140185, 4.95499234348978, 3.73767228197475, 4.66566631995082, 7.24180018529214, 1.86165620668923, 6.71055585121735, 4.09050085987528, 4.56728448626120, 8.32927593355318), 
                               (6.85507863482370, 5.10458652822909, 5.15150290804370, 4.92630846460123, 5.80793039171448, 5.38316262034369, 3.85639670822853, 5.50605377635504, 6.53801087098441, 2.31072576361973, 6.16950548244498, 6.11621427494904, 5.39665093995488, 8.11961758868942)],
                        'cov': [(0.4825, 0.6346, 0.3116), (4.901, 3.6753, 0.5098), (4.6735, 3.3277, 3.7481), (4.5256, 3.3787, 0.5036), (4.7345, 2.6839, -1.9419), (5.3549, 3.986, 1.268), (2.5126, 2.1897, 2.3002),
                                (6.2077, 3.9337, 1.2576), (0.7966, 0.4373, 0.4699), (0.4712, 0.9560, 0.6154), (1.9730, 1.3648, -0.2001), (5.955, 5.3647, -5.5329), (3.2779, 3.5396, -0.6807), (0.4742, 0.8113, 0.5934)]}, 
                'LDA14': {'mu': [(4.97148902382583, 7.94366562192532, 6.41290543855008, 5.59166897658700, 5.12550537396169, 5.61488329361499, 6.32506278151012, 5.39440688494448, 4.64429383494097, 4.50637928956749, 4.32928201168642, 3.56486075391397, 3.96837221086050, 4.36158984626990), 
                                 (5.72347748389569, 7.38352725734803, 5.70550568986279, 5.58674488833216, 5.02221965397944, 5.54757321109283, 5.62533159145318, 5.67424516931651, 5.23688639559033, 5.13121748017506, 5.64252934169444, 3.93846043945437, 5.51800934177125, 5.12398383540333)],
                          'cov': [(5.5155, 3.7229, 0.3841), (0.7614, 1.2285, 0.7936), (3.6881, 2.9854, 1.4453), (5.2963, 3.5646, 1.1432), (4.8478, 3.3793, 2.5968), (4.9391, 3.0790, 2.5868), (3.3054, 3.5045, 1.4776), 
                                  (5.5898, 3.7957, 1.7101), (4.6391, 3.5581, 1.8529), (5.8595, 4.8818, 2.2050), (4.4889, 3.3669, -0.2802), (3.4633, 2.9142, 2.2118), (5.2831, 4.5213, -0.5929), (4.739, 3.6978, 1.2085)]}}

    Max_label     = 9
    Min_label     = 1
    kwrg          = {'edgecolor':'b', 'linewidth':0.6}
    emotion_order = ['trust', 'fear', 'sadness', 'surprise', 'anger', 'disgust', 'anticipation', 'joy']
    # order         = ['EC', 'RF', 'NB', 'CNN']
    order         = ['K8', 'LDA8', 'K14', 'LDA14']
    f, axarr      = plt.subplots(2, 2, figsize = (13, 10))
    for index, key in enumerate(order):
        K = len(GMM[key]['mu'][0])
        mu  = list(zip(GMM[key]['mu'][0], GMM[key]['mu'][1]))
        col = index % 2
        row = int(index / 2)
        axarr[row, col].plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k', linewidth = 0.5)
        axarr[row, col].set_title(key, fontsize = 14)
        axarr[row, col].set_xlabel('Valence')
        axarr[row, col].set_ylabel('Arousal')
        for k in range(K):
            cov = np.array([[GMM[key]['cov'][k][0], GMM[key]['cov'][k][2]], [GMM[key]['cov'][k][2], GMM[key]['cov'][k][1]]])
            pos = np.array(mu[k])
            axarr[row, col].scatter(pos[0], pos[1], c = 'r', marker = 'x', s = 30, linewidth = 0.8)
            # axarr[row, col].annotate(emotion_order[k], pos, xytext = (pos[0] + 0.1, pos[1] + 0.1), fontsize = 10)
            gaussian_ellipse.plot_cov_ellipse(cov, pos, nstd = 1, ax = axarr[row, col], alpha = 0.7, fill = False, **kwrg)
    plt.savefig(outdir + 'GMM_modeling_ellipse_unsp.png', bbox_inches='tight')


def plot_predict_ellipse_id(data_file, idx):
    ellipse_data = get_ellipse_data(data_file)
    
    kwrg         = {'edgecolor':'k', 'linewidth':0.5}
    Max_label    = 9
    Min_label    = 1

    cov = np.array([[ellipse_data[idx]['cov_x'], ellipse_data[idx]['cov_xy']], [ellipse_data[idx]['cov_xy'], ellipse_data[idx]['cov_y']]])
    pos = np.array([ellipse_data[idx]['mean_x'], ellipse_data[idx]['mean_y']])
    plt.plot([Min_label, Max_label], [Min_label, Max_label], [Min_label, Max_label], [5, 5], [5, 5], [Min_label, Max_label], [Min_label, Max_label], [Max_label, Min_label], ls = 'dotted', c = 'k')
    gaussian_ellipse.plot_cov_ellipse(cov, pos, nstd = 1, alpha = 0.7, fill = False, **kwrg)
    plt.show()


if __name__ == '__main__':
    emo_file = _MAIN_DIR_ + '/Data/VA_Proc/emtion_tweets/survey/turk_survey_data_ABC_score.csv'
    data_file = _MAIN_DIR_ + '/AEGTools/Text/validation_patterns_outlier/individual_output_pattern_outlier.csv'
    outdir = _MAIN_DIR_ + '/NTHU/Master thesis/GMM_ellipse/'

    # plot_predict_ellipse_by_emotion(emo_file, data_file, outdir)
    # plot_predict_ellipse_id(data_file, 'A514')

    outdir = '/home/pan/Idealab/NTHU/Master thesis/'
    plot_GMM_modeling_ellipse(outdir)