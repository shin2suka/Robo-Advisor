# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:48:31 2020

@author: zhong
"""

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec
colors = plt.rcParams["axes.prop_cycle"]()

path = "C:\\Users\\zhong\\Documents\\MMF\\Risk Management Lab\\Robo-Advisor\\data\\factor data.xlsx"
factor_df = pd.read_excel(path, header=0)
factor_df.index = factor_df["DATE"]
del factor_df["DATE"]

fig = plt.figure(figsize=[25, 16])
# set height ratios for sublots
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 

# # the fisrt subplot
# ax0 = plt.subplot(gs[0])
# # log scale for axis Y of the first subplot
# line0, = ax0.plot(x, f1, color='r')

# #the second subplot
# # shared axis X
# ax1 = plt.subplot(gs[1], sharex = ax0)
# line1, = ax1.plot(x, f2, color='b', linestyle='--')
# plt.setp(ax0.get_xticklabels(), visible=False)
# # remove last tick label for the second subplot
# yticks = ax1.yaxis.get_major_ticks()
# yticks[-1].label1.set_visible(False)

# # put lened on first subplot
# ax0.legend((line0, line1), ('BAA10Y', 'WTI'), loc='lower left')

# # remove vertical gap between subplots
# plt.subplots_adjust(hspace=.0)
# plt.show()
gs = gridspec.GridSpec(len(factor_df.columns), 1, height_ratios=[1] * len(factor_df.columns))

ax_i = plt.subplot(gs[-1])
c = next(colors)["color"]
linei, = ax_i.plot(factor_df.index, factor_df[factor_df.columns[0]], color=c, label=factor_df.columns[0])
# plt.axvline(x='2007-10-01')
# plt.axvline(x='2009-04-01')
plt.axvspan('2007-10-01', '2009-04-01', facecolor='#EED5D2')
plt.axvspan('2001-07-01', '2001-10-01', facecolor='#EED5D2')
plt.axvspan('1997-01-01', '1999-01-01', facecolor='#EED5D2')
plt.axvspan('2020-01-01', '2020-04-01', facecolor='#EED5D2')
ax_i.legend(loc="upper left", handlelength = 2)
line_list = [linei]   

for i, f in enumerate(list(factor_df.columns)):
    if i != 0: 
        c = next(colors)["color"]
        print(f)
        ax_i = plt.subplot(gs[-1-i], sharex=ax_i)
        linei, = ax_i.plot(factor_df.index, factor_df[f], color=c, label=f)
        # plt.axvline(x='2007-10-01')
        # plt.axvline(x='2009-04-01')
        plt.axvspan('2007-10-01', '2009-04-01', facecolor='#EED5D2')
        plt.axvspan('2001-07-01', '2001-10-01', facecolor='#EED5D2')
        plt.axvspan('1997-01-01', '1999-01-01', facecolor='#EED5D2')
        plt.axvspan('2020-01-01', '2020-04-01', facecolor='#EED5D2')
        # plt.axvspan('2007-10-01', '2009-04-01', facecolor='#BCEE68')
        # plt.axvspan('2001-07-01', '2001-10-01', facecolor='#66CDAA')
        # plt.axvspan('1997-01-01', '1999-01-01', facecolor='#00BFFF')
        # plt.axvspan('2020-01-01', '2020-04-01', facecolor='#FFAEB9')
        ax_i.legend(loc="upper left", handlelength=2)
        line_list.append(linei)
        plt.setp(ax_i.get_xticklabels(), visible=False)
        yticks = ax_i.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)


plt.subplots_adjust(hspace=.0)
plt.show()
    
    












