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

import re

# TODO: many of these functions are quite general and may be move to utils.
# TODO: some of these function can be removed, or added to some other smaller, more specific utils thing. Or maybe just moved to the bottom of this file. 
# TODO: many of these functions lack type signatures and docstrings. Add them. 

def filter_and_drop(df,col,val):
    """ Return DataFrame with rows that match filter. Filter column is dropped. """
    return df[df[col] == val].drop([col],axis=1)

def get_means(df):
    """ Get means from data. """
    return filter_and_drop(df,'Function','avg')

def normalize_columns(df):
    """ Return new DataFrame, where the norm of each column is the unit value. """
    return df.apply(lambda x: x.values/np.linalg.norm(x.values))

# Need to improve this chunk of code. It's important, but difficult to read. 

def plot_matrix_tree(matrix,xlink,ylink,xlabels,ylabels,color):
    plt.figure(figsize=(20,10))
    plt.subplots_adjust(hspace=0,wspace=0)
    gs = gridspec.GridSpec(2, 2,height_ratios=[1,5],width_ratios=[5,1]) 
    
    def subplot_dendrogram(subplot,link,orientation):
        plt.subplot(subplot)
        dendrogram(link, orientation = orientation, link_color_func = lambda x: 'k')
        plt.axis('off')
        if orientation == 'left':
            plt.gca().invert_yaxis() # Need to flip dendrogram, since the default seems incorrect. 

    subplot_dendrogram(gs[0],xlink,'top')
    subplot_dendrogram(gs[3],ylink,'left')
    
    plt.subplot(gs[2])
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

def to_plate_layout_matrix(data,col,function='avg'):
    """ Return a matrix, given a DataFrame, a function to select, and a column to choose data from."""

    assert function in data['Function'].unique(), "{} function not present in dataframe.".format(function)

    def get_row(well_name):
        raw_coords = re.match('([a-z]*)([0-9]*)',well_name,flags=re.IGNORECASE).groups()
        return 'abcdefghijklmnopqrstuvwxyz'.index(raw_coords[0].lower())

    def get_col(well_name):
        raw_coords = re.match('([a-z]*)([0-9]*)',well_name,flags=re.IGNORECASE).groups()
        return int(raw_coords[1]) - 1
    
    df = filter_rows(data,'Function',function).copy()
    df['row'] = df['Well Name'].map(get_row)
    df['column'] = df['Well Name'].map(get_col)
    df['xy'] = df['row'].map(str) + df['row'].map(lambda x: ' - ') + df['column'].map(str)

    df = df.sort(['row','column'])
    
    n_rows = df['row'].unique().size
    n_cols = df['column'].unique().size

    return np.reshape(df[col].values,[n_rows,n_cols])

def plot_plate_old(data,parameter,function='avg',color = 'Blues',show = 'None'):
    matrix = to_plate_layout_matrix(data,parameter,function)
    xlabels = np.arange(matrix.shape[1]) + 1
    ylabels = 'abcdefghijklmnopqrstuvwxyz'.upper()[:len(matrix)]
    plt.figure(figsize=(17,5))
    plt.imshow(matrix,interpolation='nearest',cmap=color,aspect='auto');
    [plt.gca().spines[loc].set_visible(False) for loc in ['top','bottom','left','right']]
    
    plt.xticks(range(len(xlabels)),xlabels)
    plt.yticks(range(len(ylabels)),ylabels)
    plt.title("{} ({})".format(parameter,function))

    if show == 'Values':
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                plt.text(x, y, '%.1f' % matrix[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         ) 
    elif show == 'Conditions':
        labels = to_plate_layout_matrix(data,'Condition',function)
        for y in range(labels.shape[0]):
            for x in range(labels.shape[1]):
                plt.text(x, y, format_long_line(labels[y, x],12),
                         horizontalalignment='center',
                         verticalalignment='center',
                         )

def findall(s,pattern):
    """ Return list of indices where pattern occurs in s. """
    return [m.start() for m in re.finditer(pattern, s)]

def get_split_location(s):
    """ Return location with space nearest middle of string."""
    n = float(len(s))
    spaces = findall(s,' ')
    return spaces[np.argmin([np.abs((x/n) - 0.5) for x in spaces])]

def split_line(s):
    """ Return new string, where space nearest center has been replaced by newline. """
    x = get_split_location(s)
    return s[:x] + '\n' + s[x+1:]

def format_long_line(s,n):
    """ If s is longer than n, try to break line in two, """
    if len(s) > n:
        return split_line(s)
    else:
        return s
    
@curry
def map(f,x):
    """ Apply f to every element in x and return the result. """
    return [f(xi) for xi in x]   

# Num -> Num
def inc(x):
    """ Increment the value of x. """
    return x + 1

# TODO: change this function - add new variable, and maybe have a look at the summary code. 
def get_params(df):
    cols = df.columns.tolist()
    return sorted([col for col in cols if col not in ['Plate ID','Plate Name','Date','Well Name', 'Condition', 'Base', 'Dose', 'Unit', 'Drug','Function']])

# [a] -> a
def snd(x):
    """ Return second element of list. """
    return x[1]

# [(a,b)] -> [[a],[b]]
def unzip(x):
    """ Undo the zip operation. """
    return [[xi[i] for xi in x] for i in range(len(x[0]))]

# [Num] -> Num
def vrange(x):
    """ Return range of values in x. """
    return max(x) - min(x)

# (a -> b -> c) -> (a -> b -> c)
class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)
    
# [a] -> Boolean
def is_empty(x):
    return len(x) == 0

# [a] -> [a] -> Boolean
""" Return True if every element of a is in list b. """
are_all_in = Infix(lambda a,b: is_empty(set(a).difference(set(b))))

# [(Int,Int)] -> (Int,Int)
def get_shape_from_coords(coords):
    """ Return shape of smallest well plate that can contain provided indices."""  
    well_plate_shapes = [(2,4),(8,12)] # 8 well slide, 96 well plate
    rows,cols = unzip(coords)
    for shape in well_plate_shapes:
        if rows |are_all_in| range(shape[0]) and cols |are_all_in| range(shape[1]):
            return shape
    raise Exception("Given well coordinates do not fit in plate shapes:{}".format(well_plate_shapes))
    
# a -> [Int,Int] -> [[a]]
def init_matrix(val,shape):
    """ Return matrix of given shape with every element set to val. """
    rows,cols = shape
    return np.array([[val for _ in range(cols)] for _ in range(rows)])    

# [(Int,Int)] -> [a] -> a -> [[a]]
def ij_to_matrix(coords,vals,missing):
    """ Place vals at corresponding (row,column) coordinates.
        'missing' value is present wherever no val exists."""
    matrix = init_matrix(missing,
                         get_shape_from_coords(coords))
    for coord,val in zip(coords,vals):
        matrix[coord[0],coord[1]] = val
    return matrix

# String -> (Int,Int)
def well_to_ij(well):
    """ Return (row,column) matrix coordinate from alphanumeric well name. 
        For instance, well_to_ij(A03) == (0,2)"""
    return ('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(well[0]),
            int(well[1:]) -1)

# [String] -> [a] -> [[a]]
def to_plate_layout(well_names,vals):
    """ Return a 2D matrix of vals, where each val's position corresponds to well position.
        Any wells missing data are set to mean(vals). """
    return ij_to_matrix(map(well_to_ij,well_names),
                        vals,
                        np.mean(vals))

# [[a]] -> SideEffects
def plot_plate_ticks(matrix):
    xlabels = np.arange(matrix.shape[1]) + 1
    ylabels = 'abcdefghijklmnopqrstuvwxyz'.upper()[:len(matrix)]
    plt.xticks(range(len(xlabels)),xlabels)
    plt.yticks(range(len(ylabels)),ylabels)
    
# DataFrame -> String -> String -> {color:String, show:String} -> SideEffects
def plot_plate(dataframe, parameter, function, config):
    data = filter_rows(dataframe,'Function',function)
    matrix = to_plate_layout(data['Well Name'].values,
                             data[parameter].values)
    plt.imshow(matrix,interpolation='nearest',cmap=config['color'],aspect='auto');
    plot_plate_ticks(matrix)
    [plt.gca().spines[loc].set_visible(False) for loc in ['top','bottom','left','right']]
    
# DataFrame -> String -> String -> {color:String, show:String} -> SideEffects
def plot_plates(dataframe, parameter, function, color, show):
    """ Plot each plate in given dataframe."""
    plates = map(snd,dataframe.groupby('Plate ID'))
    plt.figure(figsize=(17,7))
    subplots = gridspec.GridSpec(len(plates),1)
    plt.subplots_adjust(hspace=0.54)
    for plate,sub in zip(plates,subplots):
        plt.subplot(sub) 
        plot_plate(plate,parameter,function,dict(color = color,
                                                 show = show))    