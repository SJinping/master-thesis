import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from itertools import combinations
from scipy.stats import multivariate_normal

GMM_sp = {'EC':{'mu': [(6.0993, 3.8252, 2.4539, 7.2055, 3.4726, 3.0073, 6.9598, 7.6250), (4.9331, 6.2736,2.9492,5.7133,4.4421,6.0294,6.606,7.0708)], 
                 'cov': [(2.1187, 2.3512, -0.9091), (2.8826, 2.6792, -2.1801), (0.9455, 1.0961, 0.7841), (1.2555, 2.7837, 0.2325), 
                         (2.4188, 3.7102, 0.2950), (1.1579, 2.9428, -0.7810), (1.7181, 1.4158, 1.1282), (1.0728, 1.4858, 1.0549)],
                 'pi': [0.100588266331637, 0.0624652286632110, 0.111284468635840, 0.138583354938072, 0.166888994478542, 0.117264331300730, 0.131029234611390, 0.171896121040577]}, 
           'RF': {'mu': [(7.4681,5.9486,2.9569,4.6614,2.9865,2.6225,6.1511,7.7665), (5.9636,4.7564,3.34241,5.5569,5.894,5.2682,6.0494,7.209)], 
                  'cov': [(1.0682, 2.5022, -0.0249), (1.4930, 3.6211, 0.1181), (1.6482, 1.5392, 1.3074), (3.0779, 2.9406, -2.6252), 
                          (1.4308, 3.4491, -1), (0.8665, 4.3765, 0.1760), (0.3915, 0.3527, 0.3149), (0.8346, 1.2884, 0.8596)],
                  'pi': [0.149105803555876, 0.0551730101213059, 0.194670878050009, 0.161486164327635, 0.0877838800572364, 0.0758589395973405, 0.0589668850341389, 0.216954439256459]}, 
            'CNN': {'mu': [(6.5542,4.083,4.2346,6.1864,2.4919,3.0764,6.9426,6.957), (5.74,5.2411,4.9197,5.9201,5.8114,3.1751,6.6684,6.2517)], 
                  'cov': [(3.6909, 3.4743, 1.0402), (3.7196, 3.6936, -0.4777), (4.4531, 3.9842, 1.1656), (3.9214, 2.8423, 2.0440), 
                          (0.8551, 4.4414, -0.1670), (3.1574, 2.0337, 2.0585), (1.9011, 1.7661, 1.6486), (2.3234, 2.6112, 1.1888)],
                  'pi': [0.155350113331041, 0.0858528646389973, 0.219417186701612, 0.123166359866743, 0.0934595177787123, 0.0665188737877388, 0.0741447909689702, 0.182090292926186]}, 
            'NB': {'mu': [(7.6456487955,4.2618113992,3.5641895464,6.1101990398,2.8822112401,2.6001490296,6.2034454239,7.8687929036), (5.9856827793,5.7334504022,3.8039572393,5.2083477383,5.8671703559,4.5266140617,5.9976072374,7.3787615215)], 
                   'cov': [(0.9199, 2.5857, 0.2949), (3.0033, 3.1403, -2.9625), (2.4248, 2.0883, 2.0700), (1.6566, 2.9549, -0.6965), 
                          (1.4131, 4.1345, -1.4231), (0.8408, 4.2789, 1.3229), (0.4063, 0.3172, 0.2703), (0.7229, 1.1905, 0.8166)],
                   'pi': [0.147730017363936, 0.0654752119401377, 0.180292651319162, 0.137203335290035, 0.123771807491324, 0.112466604833396, 0.0562126124122784, 0.176847759349730]}}

GMM_unsp = {'K8': {'mu':[(2.12410820378871, 5.07044858548973, 8.03800390218534, 5.11661350181792, 5.07926260711300, 4.81514439133074, 6.63696432136261, 3.84788802297019), 
                             (2.49759650169733, 4.95263262708440, 7.62639849758654, 5.14606733510362, 5.40512580457496, 5.56088591924007, 6.14402001213801, 6.29076115857237)],
                       'cov': [(0.6769, 0.9473, 0.7266), (4.628, 3.2857, 1.0705), (0.6671, 1.1339, 0.7845), (4.8957, 3.7483, 1.187), 
                               (5.2444, 3.7483, 1.187), (6.2234, 3.784, 1.0298), (2.0313, 1.1912, 1.307), (3.9244, 3.8233, -3.5352)],
                       'pi': [0.0680085337922435, 0.181199281141288, 0.127070021024149, 0.207921884243477, 0.100604691773400, 0.159467633283019, 0.0900158493724118, 0.0657121053700116]},
                'LDA8': {'mu': [(5.43173938353078, 5.50614556727446, 6.21469960503113, 5.35610661813136, 5.52285474763824, 4.84981512253809, 4.39973634723678, 3.44786332505724), 
                                (5.93789597445698, 5.61156341243733, 5.97325098285176, 5.28408125928526, 5.36541163277992, 5.44464792274123, 5.18231601832514, 4.49694709169587)],
                         'cov':[(6.8311, 4.0253, 2.2595), (5.742, 3.9762, -0.5179), (4.6572, 3.8977, 3.8941), (5.63, 4.0595, 2.1965),
                                (5.1282, 3.7611, 2.2061), (5.5653, 4.2623, 1.2194), (5.9415, 4.9916, 1.3437), (2.7025, 4.6774, 0.6271)],
                         'pi': [0.238355583913759, 0.0866647330997778, 0.0720512578186694, 0.148329638946718, 0.236764425894851, 0.0812645141970111, 0.0877157322691879, 0.0488541138600262]},
                'K14': {'mu': [(8.11696014714374, 4.90605080346294, 5.27748969875437, 4.89847068229426, 4.93400973140185, 4.95499234348978, 3.73767228197475, 4.66566631995082, 7.24180018529214, 1.86165620668923, 6.71055585121735, 4.09050085987528, 4.56728448626120, 8.32927593355318), 
                               (6.85507863482370, 5.10458652822909, 5.15150290804370, 4.92630846460123, 5.80793039171448, 5.38316262034369, 3.85639670822853, 5.50605377635504, 6.53801087098441, 2.31072576361973, 6.16950548244498, 6.11621427494904, 5.39665093995488, 8.11961758868942)],
                        'cov': [(0.4825, 0.6346, 0.3116), (4.901, 3.6753, 0.5098), (4.6735, 3.3277, 3.7481), (4.5256, 3.3787, 0.5036), (4.7345, 2.6839, -1.9419), (5.3549, 3.986, 1.268), (2.5126, 2.1897, 2.3002),
                                (6.2077, 3.9337, 1.2576), (0.7966, 0.4373, 0.4699), (0.4712, 0.9560, 0.6154), (1.9730, 1.3648, -0.2001), (5.955, 5.3647, -5.5329), (3.2779, 3.5396, -0.6807), (0.4742, 0.8113, 0.5934)],
                        'pi': [0.0280452524332837, 0.171984110760173, 0.0724127692873343, 0.140405651901258, 0.0545481693909842, 0.0766136336435476, 0.0273685730438849, 0.133818154103220, 0.0439203922643704, 0.0432763812023775, 0.0456651098157943, 0.0293052178436857, 0.0468541475251079, 0.0857824367849792]}, 
                'LDA14': {'mu': [(4.97148902382583, 7.94366562192532, 6.41290543855008, 5.59166897658700, 5.12550537396169, 5.61488329361499, 6.32506278151012, 5.39440688494448, 4.64429383494097, 4.50637928956749, 4.32928201168642, 3.56486075391397, 3.96837221086050, 4.36158984626990), 
                                 (5.72347748389569, 7.38352725734803, 5.70550568986279, 5.58674488833216, 5.02221965397944, 5.54757321109283, 5.62533159145318, 5.67424516931651, 5.23688639559033, 5.13121748017506, 5.64252934169444, 3.93846043945437, 5.51800934177125, 5.12398383540333)],
                          'cov': [(5.5155, 3.7229, 0.3841), (0.7614, 1.2285, 0.7936), (3.6881, 2.9854, 1.4453), (5.2963, 3.5646, 1.1432), (4.8478, 3.3793, 2.5968), (4.9391, 3.0790, 2.5868), (3.3054, 3.5045, 1.4776), 
                                  (5.5898, 3.7957, 1.7101), (4.6391, 3.5581, 1.8529), (5.8595, 4.8818, 2.2050), (4.4889, 3.3669, -0.2802), (3.4633, 2.9142, 2.2118), (5.2831, 4.5213, -0.5929), (4.739, 3.6978, 1.2085)],
                          'pi':[0.0844373644545292, 0.0777526299762519, 0.0744078915818343, 0.0597389893015536, 0.0788435224621517, 0.0541217612974307, 0.0778534485945309, 0.100680266316155, 0.0772811214873126, 0.0936949611559137, 0.0374690948146188, 0.0403043159894312, 0.0655934972098239, 0.0778211353584632]}}

class BiGauss(object):
    """docstring for BiGauss"""
    def __init__(self, mu1, mu2, Sigma1, Sigma2, pi1, pi2, steps = 1000):
        super(BiGauss, self).__init__()
        self.mu1      = mu1
        self.mu2      = mu2
        self.Sigma1   = Sigma1
        self.Sigma2   = Sigma2
        self.pi1      = pi1
        self.pi2      = pi2
        self.biGauss1 = multivariate_normal(mean = self.mu1, cov = self.Sigma1, allow_singular = True)
        self.biGauss2 = multivariate_normal(mean = self.mu2, cov = self.Sigma2, allow_singular = True)
        self.steps    = steps
        self.inv_Sig1 = -inv(self.Sigma1)
        self.inv_Sig2 = -inv(self.Sigma2)

        self.A_1 = self.inv_Sig1[0][0]
        self.B_1 = self.inv_Sig1[0][1]
        self.C_1 = self.inv_Sig1[1][0]
        self.D_1 = self.inv_Sig1[1][1]
        self.A_2 = self.inv_Sig2[0][0]
        self.B_2 = self.inv_Sig2[0][1]
        self.C_2 = self.inv_Sig2[1][0]
        self.D_2 = self.inv_Sig2[1][1]

    def pdf(self, x):
        return self.pi1 * self.biGauss1.pdf(x) + self.pi2 * self.biGauss2.pdf(x)

    # Overlap rate
    def OLR(self):
        e      = math.sqrt((self.mu1[0] - self.mu2[0])**2 + (self.mu1[1] - self.mu2[1])**2) / float(self.steps)
        x_step = e*(self.mu1[0]-self.mu2[0])
        y_step = e*(self.mu1[1]-self.mu2[1])
        p      = [self.mu1[0] - x_step, self.mu1[1] - y_step]
        p_pre  = self.mu1
        p_min = min(self.pdf(p), self.pdf(p_pre))
        index  = 0
        while index < self.steps:
            p_next = [p[0] - x_step, p[1] - y_step]
            if self.pdf(p) > self.pdf(p_pre) and self.pdf(p) > self.pdf(p_next):
                p_max = self.pdf(p)
            if self.pdf(p) < self.pdf(p_pre) and self.pdf(p) < self.pdf(p_next):
                p_min = self.pdf(p)
            p_pre = p
            p     = p_next
            index += 1

        pdf_mu1 = self.pdf(self.mu1)
        pdf_mu2 = self.pdf(self.mu2)
        return p_min / min(pdf_mu1, pdf_mu2) if p_min < min(pdf_mu1, pdf_mu2) else 1.0

    def OLR2(self):
        e      = math.sqrt((self.mu1[0] - self.mu2[0])**2 + (self.mu1[1] - self.mu2[1])**2) / float(self.steps)
        x_step = e*(self.mu1[0]-self.mu2[0])
        y_step = e*(self.mu1[1]-self.mu2[1])
        p_x = self.mu1[0] - x_step
        while self.RC(p_x) == None:
            p_x = p_x - x_step
        p_y = self.RC(p_x)
        p = [p_x, p_y]
        p_pre  = self.mu1
        p_min = min(self.pdf(p), self.pdf(p_pre))
        p_max = max(self.pdf(p), self.pdf(p_pre))
        index  = 0
        while index < self.steps:
            if self.RC(p[0] - x_step) != None:
                p_next = [p[0] - x_step, self.RC(p[0] - x_step)]
                if self.pdf(p) > self.pdf(p_pre) and self.pdf(p) > self.pdf(p_next):
                    p_max = self.pdf(p)
                if self.pdf(p) < self.pdf(p_pre) and self.pdf(p) < self.pdf(p_next):
                    p_min = self.pdf(p)
            p_pre = p
            p     = p_next
            index += 1

        pdf_mu1 = self.pdf(self.mu1)
        pdf_mu2 = self.pdf(self.mu2)
        return p_min / min(pdf_mu1, pdf_mu2) if p_min < min(pdf_mu1, pdf_mu2) else 1.0

    # get y given x, satisfying (x,y) is on the RC
    def RC(self, x):
        E = self.A_1 * (x - self.mu1[0])
        F = self.C_1 * (x - self.mu1[0])
        G = self.A_2 * (x - self.mu2[0])
        H = self.C_2 * (x - self.mu2[0])

        I = E * self.D_2 - F * self.B_2
        J = H * self.B_1 - G * self.D_1
        K = self.B_1 * self.D_2 - self.B_2 * self.D_1
        M = F * G - E * H

        P = K
        Q = I + J - K * (self.mu2[1] + self.mu1[1])
        S = -(M + I * self.mu2[1] + J * self.mu1[1])

        if Q**2 - 4*P*S < 0:
            return None

        y = max((-Q + math.sqrt(Q**2 - 4*P*S)) / (2*P), (-Q - math.sqrt(Q**2 - 4*P*S)) / (2*P))

        return y



if __name__ == '__main__':
    
    # GMM = GMM_sp
    # for key, vals in GMM.items():
    #     # if key != 'LDA8':
    #     #     continue
    #     mu  = list(zip(GMM[key]['mu'][0], GMM[key]['mu'][1]))
    #     covs = [np.array([[cov[0], cov[2]], [cov[2], cov[1]]]) for cov in GMM[key]['cov']]
    #     pi = GMM[key]['pi']
    #     OLRs = []
    #     for GMM1, GMM2 in combinations(zip(mu, covs, pi), 2):
    #         biGauss = BiGauss(GMM1[0], GMM2[0], GMM1[1], GMM2[1], GMM1[2], GMM2[2])
    #         OLR = biGauss.OLR2()
    #         OLRs.append(OLR)
    #     OLR = np.mean(OLRs)
    #     print(key, OLR)
    #     print(OLRs)

    # Single sample
    pi1 = 0.47
    pi2 = 1.0 - pi1
    mu1 = [0,0]
    mu2 = [3,0]
    Sigma1 = [[1,0],[0,1]]
    Sigma2 = [[2.17,1.82], [1.82,2.17]]
    Bi = BiGauss(mu1, mu2, Sigma1, Sigma2, pi1, pi2)
    print(Bi.OLR2())


    # step = 0.1
    # pi1 = 0.5
    # OLRs = []
    # x = []
    # mu1 = [0,0]
    # mu2 = [0,0]
    # Sigma1 = [[1,0],[0,1]]
    # Sigma2 = [[2.17,1.82], [1.82,2.17]]
    # while mu2[0] <= 8.0:
    #     pi2 = 1.0-pi1
    #     Bi = BiGauss(mu1, mu2, Sigma1, Sigma2, pi1, pi2)
    #     OLRs.append(Bi.OLR2())
    #     x.append(mu2[0])
    #     mu2[0] += step
    # print(OLRs)
    # plt.plot(x, OLRs)
    # plt.show()