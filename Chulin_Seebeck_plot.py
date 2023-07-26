# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:30:51 2019

@author: CRYOGENIC
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
from IPython import get_ipython
mgc = get_ipython().magic
mgc(u'%matplotlib qt')

font = {'size'   : 7}

matplotlib.rc('font', **font)

#fnames = ['results/GaGdN-300K-190809_001.dat',
#          'results/GaGdN-300K-190809_002.dat']
fname = "20200727_11323(4a)_thermocouple_001.txt"
colors = ['b', 'g']
interval = 1000

#x_column = 'B_digital'
#y_column = 'V_real_12'

style.use('fivethirtyeight')

def animate(i, ax, fname, x_column, y_column):
    graph_data = pd.read_csv(fname, skipinitialspace=True)
    x = graph_data[x_column]
    y = graph_data[y_column]
    ax.clear()
    ax.plot(x,y)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    
def animate_multi(i, axs, fname, color=None):
    axs = axs.flatten()
    for ax in axs:
        ax.clear()
    graph_data = pd.read_csv(fname, sep = ' ', header = 0, )
#    graph_data = graph_data[::10]
    graph_data['RTDl'] = graph_data['RTDl'] -graph_data['RTDl'][0]+graph_data['RTDr'][0]
    
    graph_data['Tl'] = 9.91684E-6*graph_data['RTDl']**2+0.23605*graph_data['RTDl']-245.96823
    graph_data['Tr'] = 9.91684E-6*graph_data['RTDr']**2+0.23605*graph_data['RTDr']-245.96823
    
    graph_data['TimeDur'] = (pd.to_timedelta(graph_data.Time)-pd.to_timedelta(graph_data.Time[0]))/1e9;
    graph_data['T_average'] = (graph_data.Tl+graph_data.Tr)/2
    graph_data['DeltaT'] = (graph_data.Tl-graph_data.Tr)
  
    axs[0].set_xlabel('times(s)')
    axs[0].set_ylabel('T(C)')
    axs[0].plot(graph_data.TimeDur, graph_data.Tl, label = 'Tl')
    axs[0].plot(graph_data.TimeDur, graph_data.Tr, label = 'Tr')
    axs[0].plot(graph_data.TimeDur, (graph_data.Tr + graph_data.Tl) / 2, '-', label = 'Tavg')
    axs[0].legend(loc = 'upper left')
    
    axs[1].set_xlabel('times(s)')
    axs[1].set_ylabel('U(uV)')
#    axs[1].set_xlim([2347, 2347+600])
#    axs[1].set_ylim([-15e-6, 10e-6])
    axs[1].plot(graph_data.TimeDur, graph_data.Vsamp*1e6, label = 'Vsamp')
    #axs[1].legend(loc = 'middle left')
    
    axs[2].set_ylabel('U(uV)')
#    axs[2].set_ylim([-15e-6, 10e-6])
    axs[2].set_xlabel('DeltaT(K)')
    axs[2].plot(graph_data.DeltaT, graph_data.Vsamp*1e6, label = 'Vsamp')
    
#    axs[2].set_xlabel('T_Ave(C)')
#    axs[2].plot(graph_data.T_average, graph_data.Vsamp, label = 'Vsamp')
    #axs[2].legend(loc = 'lower left')
    axs[3].set_ylabel('U(uV)')
    axs[3].set_xlabel('T_Ave(C)')
    axs[3].plot(graph_data.T_average, graph_data.Vsamp*1e6, label = 'Vsamp')
 
    
    
    plt.tight_layout()
  
    '''
def animate_multi_files(i, axs, fnames, x_columns, y_columns, colors=None):
    for ax in axs.flatten():
        ax.clear()
    for fname, color in zip(fnames, colors):
        graph_data = pd.read_csv(fname, skipinitialspace=True)
        for x_column, y_column, ax in zip(x_columns, y_columns, axs[0]):
            x = graph_data[x_column]
            y = graph_data[y_column]
#            ax.clear()
            ax.plot(x,y, color=color)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
        graph_data = graph_data[graph_data['flag'] != 0]
        for x_column, y_column, ax in zip(x_columns, y_columns, axs[1]):
            x = graph_data[x_column]
            y = graph_data[y_column]
#            ax.clear()
            ax.plot(x,y,'x', color=color)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
    plt.tight_layout()
    '''
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ani = animation.FuncAnimation(fig, animate, fargs=[ax, fname, x_column, y_column], interval=1000)
fig, axs = plt.subplots(2,2, figsize=(8,6))
fig.set_dpi(300)
ani = animation.FuncAnimation(fig, animate_multi, fargs=[axs, fname, colors], interval=interval)
plt.show()