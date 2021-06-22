import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import time
import pickle
import zipfile
import os
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('GTK3Cairo')

path = ""
T = int(3e4)
legend = {2: 'origin', 3: 'dot', 4: 'ONS', 6: 'lazy', 11: 'dot-lazy', 12: 'trainONS'}

fig , ax = plt.subplots()
for i in [2, 3, 4, 6, 11, 12]:
    # print out the average cumulative regret all methods
    with open(path + 'ID=' + str(i) + '_linbandits_seed_' + str(46) + '.pickle', 'rb') as handle:
        dictResults = pickle.load(handle)
    regret = dictResults["lin_rbmle"]["meanCumRegrets"]

    plt.plot(range(1, T+1), regret, label=legend[i])
ax.axis('auto')
plt.xlim(0, 30000)
plt.ylim(0, 13000)
leg = ax.legend(loc='upper left', shadow=True) 
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.show()

# legend = ['origin', 'ONS', 'lazy update', 'dot']
# plt.bar(legend,
#         [0.235270643234252, 0.017754952669143675, 0.025433093786239622, 0.21336042046546935], 
#         width=0.5, 
#         bottom=None, 
#         align='center', 
#         color=['lightsteelblue', 
#                'cornflowerblue', 
#                'royalblue', 
#                'midnightblue'])
# # plt.xticks(rotation='vertical')
# plt.show()

# legend = ['origin', 'ONS', 'lazy update', 'dot']
# plt.bar(legend,
#         [0.26410406589508056, 0.04661219382286072, 0.048460785865783694, 0.2380339741706848], 
#         width=0.5, 
#         bottom=None, 
#         align='center', 
#         color=['lightsteelblue', 
#                'cornflowerblue', 
#                'royalblue', 
#                'midnightblue'])
# # plt.xticks(rotation='vertical')
# plt.show()

# legend = ['origin', 'ONS', 'lazy update', 'dot']
# plt.bar(legend,
#         [0.2305649745464325, 0.01749592411518097, 0.019796232130527495, 0.21539191484451295], 
#         width=0.5, 
#         bottom=None, 
#         align='center', 
#         color=['lightsteelblue', 
#                'cornflowerblue', 
#                'royalblue', 
#                'midnightblue'])
# # plt.xticks(rotation='vertical')
# plt.show()

legend = ['origin', 'ONS', 'lazy update', 'dot']
plt.bar(legend,
        [0.351669899225235, 0.15053131568431855, 0.06604543685913086, 0.2406567680835724], 
        width=0.5, 
        bottom=None, 
        align='center', 
        color=['lightsteelblue', 
               'cornflowerblue', 
               'royalblue', 
               'midnightblue'])
# plt.xticks(rotation='vertical')
plt.show()
