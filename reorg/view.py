import pandas as pd
from pandas import DataFrame as DF
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display
import IPython.html.widgets as widgets

from scipy.cluster.hierarchy import \
    linkage,\
    leaves_list,\
    dendrogram
    
from IPython.html.widgets import \
    interact,\
    interactive,\
    fixed
    
from toolz import \
    pipe,\
    thread_first,\
    curry,\
    frequencies
    
from utils import \
    concatenate,\
    snd,\
    filter_rows,\
    unzip,\
    are_all_in,\
    format_long_line,\
    checker


# TODO: many of these functions are quite general and may be move to utils.
# TODO: some of these function can be removed, or added to some other smaller, more specific utils thing. Or maybe just moved to the bottom of this file. 
# TODO: many of these functions lack type signatures and docstrings. Add them. 

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

# [a] -> [b] -> [(a,b)]
def cross(a,b):
    """ Return list containing all combinations of elements of a and b."""
    return concatenate([[(ai,bi) for bi in b] for ai in a])

# (Int,Int) -> [(Int,Int)] -> [(Int,Int)]
def find_missing_values(shape,coords):
    """ Given shape of 2D matrix and list of ij (row,column) coordinates, return coordinates not in list. """
    all_coords = cross(range(shape[0]),range(shape[1]))
    return list(set(all_coords).difference(set(coords)))

# [(Int,Int)] -> [a] -> a -> ([[a]], [(Int,Int)])
def ij_to_matrix(coords,vals,missing):
    """ Place vals at corresponding (row,column) coordinates in smallest-possible well plate.
        'missing' value is present wherever no val exists. Returns both a 2D matrix and a list of wells missing data. """
    shape = get_shape_from_coords(coords)
    matrix = init_matrix(missing,shape)
    for coord,val in zip(coords,vals):
        matrix[coord[0],coord[1]] = val
    return (matrix,
            find_missing_values(shape,coords))

# String -> (Int,Int)
def well_to_ij(well):
    """ Return (row,column) matrix coordinate from alphanumeric well name. 
        For instance, well_to_ij(A03) == (0,2)"""
    return ('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(well[0]),
            int(well[1:]) -1)

# [(Int,Int)] -> [a] -> ([[a]], [(Int,Int)])
def to_plate_layout(coords,vals):
    """ Return a smallest-possible, 2D well-plate matrix of vals, where each val's position corresponds to its ij-coordinate.
        Any wells missing data are set to mean(vals). Also returns list of coords in plate that are missing data."""
    return ij_to_matrix(coords,
                        vals,
                        np.nan)
                        
# [[a]] -> SideEffects
def plot_plate_ticks(matrix,xticks=True):
    xlabels = np.arange(matrix.shape[1]) + 1
    ylabels = 'abcdefghijklmnopqrstuvwxyz'.upper()[:len(matrix)]
    plt.xticks(range(len(xlabels)),xlabels) if xticks else plt.xticks([])
    plt.yticks(range(len(ylabels)),ylabels)
    
# Int -> Int -> String -> SideEffects                
def plot_plate_text(i,j,text,format_string='%s'):
    plt.text(j,i,format_string % text,
             horizontalalignment='center',
             verticalalignment='center',
             fontsize = 8)    
    
# DataFrame -> String -> {color:String, show:String, xticks?:Boolean, vmin: Float, vmax: Float} -> SideEffects
def plot_plate(dataframe, parameter, config):
    data = dataframe
    coords = map(well_to_ij, data['Well Name'].values)
    values = data[parameter].values
    conditions = data['Condition'].values
    matrix, missing_coords = to_plate_layout(coords, values)
    
    if config['color'] == None:
        plt.imshow(
            checker(*matrix.shape) & ~np.isnan(matrix),
            interpolation = 'nearest',
            cmap = 'Greys',
            aspect = 'auto',
            vmin = 0, 
            vmax = 6);
    else:
        plt.imshow(
            matrix,
            interpolation = 'nearest',
            cmap = config['color'],
            aspect = 'auto',
            vmin = config['vmin'], 
            vmax = config['vmax']);
    
    plot_plate_ticks(matrix,xticks=config['xticks?'])
    
    # Plot 'no data' for wells that don't have any data
    [plot_plate_text(i,j,'No data') for i,j in missing_coords]
 
    # Label wells with values, conditions, or nothing (empty strings)
    show = config['show']
    vs = {'Conditions': [format_long_line(c, 12) for c in conditions],
          'Values': values,
          'None': ['' for _ in values]}
    formats = {'Conditions': '%s',
               'Values': '%.2f',
               'None': '%s'}
    [plot_plate_text(i, j, v, format_string = formats[show]) \
         for (i, j), v in zip(coords, vs[show])]
    
    #[plt.gca().spines[loc].set_visible(False) for loc in ['top','bottom','left','right']]
    
# DataFrame -> String -> String -> String -> String -> SideEffects
def plot_plates(dataframe, parameter, color, show):
    """ Plot each plate in given dataframe."""
    plates = map(snd, dataframe.groupby('Plate ID'))
    plt.figure(figsize=(17, 7))
    subplots = gridspec.GridSpec(len(plates), 1)
    plt.subplots_adjust(hspace=0.0)
    for i, (plate, sub) in enumerate(zip(plates, subplots)):
        plt.subplot(sub) 
        plot_plate(
            plate,
            parameter,
            {'color': color,
             'show': show,
             'xticks?': i == len(plates)-1,
             'vmin': dataframe[parameter].min(),
             'vmax': dataframe[parameter].max()})