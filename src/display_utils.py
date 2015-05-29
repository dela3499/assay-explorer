import pandas as pd
from pandas import DataFrame as DF

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn.apionly as sns

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

from toolz import pipe,thread_first, curry, frequencies

from IPython.html.widgets import interact, interactive, fixed
from IPython.display import display
import IPython.html.widgets as widgets
from IPython.utils.traitlets import link as traitlink

def filter_and_drop(df,col,val):
    """ Return DataFrame with rows that match filter. Filter column is dropped. """
    return df[df[col] == val].drop([col],axis=1)

def get_means(df):
    """ Get means from data. """
    return filter_and_drop(df,'Function','mean')

def normalize_columns(df):
    """ Return new DataFrame, where the norm of each column is the unit value. """
    return df.apply(lambda x: x.values/np.linalg.norm(x.values))

# Need to improve this chunk of code. It's important, but difficult to read. 

def plot_matrix_tree(matrix,xlink,ylink,xlabels,ylabels,color):
    plt.figure(figsize=(20,10))
    plt.subplots_adjust(hspace=0,wspace=0)
    gs = gridspec.GridSpec(2, 1,height_ratios=[1,5]) 
    
    def subplot_dendrogram(subplot,link,orientation):
        plt.subplot(subplot)
        dendrogram(link, orientation = orientation, link_color_func = lambda x: 'k')
        plt.axis('off')

    subplot_dendrogram(gs[0],xlink,'top')
    
    plt.subplot(gs[1])
    plt.imshow(matrix,interpolation='nearest',cmap=color,aspect='auto');
    [plt.gca().spines[loc].set_visible(False) for loc in ['top','bottom','left','right']]
    
    plt.xticks(range(len(xlabels)),xlabels,rotation=30,ha='right')
    plt.yticks(range(len(ylabels)),ylabels)

def matrix_tree(data,color):
    normed_data = data.values

    condition_link = linkage(normed_data)
    feature_link = linkage(normed_data.T)

    condition_order = leaves_list(condition_link)
    feature_order = leaves_list(feature_link)

    conditions = data.index.values[condition_order]
    features = data.columns.values[feature_order]

    color_matrix = normed_data.T[feature_order,:][:,condition_order]
    
    plot_matrix_tree(color_matrix,
                     condition_link,
                     feature_link,
                     conditions,
                     features,
                     color)

def parse_condition(x):
    words = x.split(' ')
    if len(words) == 2: 
        return dict(Base = words[0],
                    Dose = 0,
                    Unit = '',
                    Drug = words[-1])
    elif len(words) == 3:
        return dict(Base = words[0],
                    Dose = float(words[1].rstrip('%')),
                    Unit = '%',
                    Drug = words[2])
    elif len(words) == 4: 
        return dict(Base = words[0],
                    Dose = float(words[1]),
                    Unit = words[2],
                    Drug = words[3])
    elif len(words) == 5:
        return dict(Base = words[0],
                    Dose = float(words[1]),
                    Unit = words[2],
                    Drug = " ".join(words[3:]))
    else:
        raise Exception("don't know how to parse condition: {}".format(x))

def add_condition_cols(df):
    """ Return new DataFrame with new columns containing parsed information
        from condition. """
    new_df = df.copy()
    for field in ['Base','Dose','Unit','Drug']:
        new_df[field] = df['Condition'].map(lambda c: parse_condition(c)[field])
    return move_columns(new_df,[['Base',2],['Dose',3],['Unit',4],['Drug',5]])

def filter_rows(df,col,val):
    """ Return new DataFrame where the values in col match val. 
        col may be a single value or a list of values. """
    if type(val) == list:
        return df[df[col].isin(val)]
    else:
        return df[df[col] == val]

def get_drugs_with_multiple_doses(df):
    """ Return list of drugs that have multiple entries. """
    """ TODO: add check for unique doses, rather than just multiple entries. """
    return pipe(df.groupby('Drug').count()['Dose'],
                lambda x: x[x>1].index.values,
                list)

def label(x,y,t,fontsize=12):
    plt.xlabel(x,fontsize=fontsize)
    plt.ylabel(y,fontsize=fontsize)
    plt.title(t,fontsize=fontsize)

def dose_plot(df,err,cols,scale='linear'):
    n_rows = int(np.ceil(len(cols)/3.0))
    plt.figure(figsize=(20,4 * n_rows))
    subs = gridspec.GridSpec(n_rows, 3) 
    plt.subplots_adjust(hspace=0.54,wspace=0.27)

    for col,sub in zip(cols,subs):
        plt.subplot(sub)
        for base in df['Base'].unique():
            for drug in get_drugs_with_multiple_doses(filter_rows(df,'Base',base)):
                data = thread_first(df,
                                    (filter_rows,'Drug',drug),
                                    (filter_rows,'Base',base),
                                    (DF.sort, 'Dose'))
                error = thread_first(err,
                                     (filter_rows,'Drug',drug),
                                     (filter_rows,'Base',base),
                                     (DF.sort, 'Dose'))
                if scale == 'linear':
                    plt.errorbar(data['Dose'],data[col],yerr=error[col])
                    title = "{} vs. Dose".format(col)
                else: 
                    plt.errorbar(data['Dose'],data[col],yerr=error[col])
                    plt.xscale('log')
                    title = "{} vs. Dose (Log Scale)".format(col)
                    plt.xticks(data['Dose'].values,data['Dose'].values)
                    plt.xlim(0.06,15)
                label('Dose ({})'.format(data.Unit.values[0]), col,title,fontsize = 15)

                plt.legend(df['Base'].unique(), loc = 0)

def colorize(c):
    if 'control' in c: 
        return (1,1,1)
    elif 'drug A' in c:
        return (0.3,1,1)
    elif 'Telomestatin' in c:
        return (1,0.3,1)
    elif 'HU' in c:
        return (1,1,0.3)
    elif 'DMSO' in c:
        return (0.3,1,0.3)
    else:
        raise Exception('Do not know how to color {}'.format(c))              