from pandas import DataFrame as df
import pandas as pd
import numpy as np
from toolz import thread_first
from utils import curry_funcs,\
                  drop_matching_columns,\
                  add_normalized_columns,\
                  headers_to_column,\
                  summarize_groups

curry_funcs(['pd.read_csv',
             'df.dropna',
             'df.rename'])

###################################################
### Cell Data Config ##############################
###################################################

# String -> String
def rename_column(col):
    """ Rename column col to remove whitespace, backslashes, prefixes, 
        and suffixes (esp. large parenthetic suffix). """
    if col.startswith('Cell:'):
        return col.split('(')[0].lstrip("Cell:").rstrip('/').strip(' ')
    else: 
        return col.split('(')[0].rstrip('/').strip(' ')

def check_cell_data(dataframe):
    return dataframe

# type CellConfig = {
#        path      :: String,
#        skiprows  :: Int | [Int],
#        dropcols  :: [RegexString],
#        normcols  :: [[String,[String],[String]]],
#        colrename :: (String -> String),
#        check     :: (DataFrame -> DataFrame | Exception) }

cell_config = dict(
    path = '../input/moldev_data.csv',
    skiprows = 4,
    dropcols = ['Cell ID',
                'Site ID',
                'MEASUREMENT SET ID',
                '.*ObjectID.*',
                '\.[0-9]*\Z'],
    normcols = [['Normalized APB spots',  
                  ['# of APBs'],
                  ['# of FITC spots', '# of TxRed spots']],
                ['Normalized Coloc area', 
                  ['Area_Coloc_Avg'],
                  ['Area_FITC','Area_TxRed']],
                ['Normalized Coloc spots',
                  ['# Coloc Spots'],   
                  ['# of FITC spots', '# of TxRed spots']]],
    colrename = rename_column,
    check = check_cell_data
    )

###################################################
### Lookup Config #################################
###################################################

def check_lookup_data(dataframe):
    return dataframe

# type LookupConfig = {
#        path      :: String,
#        skiprows  :: Int | [Int],
#        check     :: (DataFrame -> DataFrame | Exception) }

lookup_config = dict(
    path = '../input/conditions_and_wells.csv',
    skiprows = [1],
    check = check_lookup_data
    )

###################################################
### Analysis ######################################
###################################################

# CellConfig -> DataFrame
def get_cell_data(c):
    return thread_first(c['path'],
                        pd.read_csv(skiprows=c['skiprows']),
                        df.dropna(axis=1,how='all'),
                        (drop_matching_columns,c['dropcols']),
                        df.rename(columns=c['colrename']),
                        (add_normalized_columns,c['normcols']),
                        c['check'])

# LookupConfig -> DataFrame
def get_lookup_data(c):
    return thread_first(c['path'],
                        pd.read_csv(skiprows=c['skiprows']),
                        df.dropna(axis=1,how='all'),
                        headers_to_column,
                        df.rename(columns=dict(
                                    values = 'Well Name',
                                    label = 'Condition')),
                        c['check'])

def summarize_conditions(data,c):
    return thread_first(data,
                        (df.groupby,c['groupby']),
                        (summarize_groups,c['funcs'],c['fnames']))

funcs = [df.mean,df.std,df.sem,df.count,df.min,df.max]
fnames = ['avg','std','sem','count','min','max']

condition_config = dict(groupby = 'Condition', funcs = funcs, fnames = fnames)
well_config = dict(groupby = 'Well Name', funcs = funcs, fnames = fnames)

data = pd.merge(get_cell_data(cell_config),
                get_lookup_data(lookup_config),
                on = 'Well Name')

condition_summary = summarize_conditions(data.drop('Well Name',axis=1),
                                         condition_config)

well_summary = summarize_conditions(data,
                                    well_config)

# Write to files
data.to_csv('../output/moldev_cleaned.csv',index=False)
well_summary.to_csv('../output/well_summary.csv',index=False)
condition_summary.to_csv('../output/condition_summary.csv',index=False)
# condition_summary.T.to_csv('../output/condition_summary_transpose.csv',header=False)
