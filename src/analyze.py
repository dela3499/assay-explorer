from pandas import DataFrame as df
import pandas as pd
import numpy as np
from toolz import thread_first
from utils import curry_funcs,\
                  drop_matching_columns,\
                  add_normalized_columns

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

cell = dict(
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
                ['Normalized Coloc spots',
                  ['# Coloc Spots'],   
                  ['# of FITC spots', '# of TxRed spots']],
                ['Normalized Coloc area', 
                  ['Area_Coloc_Avg'],
                  ['Area_FITC','Area_TxRed']]],
    colrename = rename_column,
    check = check_cell_data
    )

###################################################
### Lookup Config #################################
###################################################

def check_lookup_data(dataframe):
    return dataframe

lookup = dict(
    path = '../input/conditions_and_wells.csv',
    skiprows = [1],
    check = check_lookup_data
    )

###################################################
### Analysis ######################################
###################################################

# type CellConfig = {
#        path      :: String,
#        skiprows  :: Int | [Int],
#        dropcols  :: [RegexString],
#        normcols  :: [[String,[String],[String]]],
#        colrename :: (String -> String),
#        check     :: (DataFrame -> DataFrame | Exception) }
#
# CellConfig -> DataFrame
def get_cell_data(c):
    return thread_first(c['path'],
                        pd.read_csv(skiprows=c['skiprows']),
                        df.dropna(axis=1,how='all'),
                        df.rename(columns=c['colrename']),
                        (drop_matching_columns,c['dropcols']),
                        (add_normalized_columns,c['normcols']),
                        c['check'])

# type LookupConfig = {
#        path      :: String,
#        skiprows  :: Int | [Int],
#        check     :: (DataFrame -> DataFrame | Exception) }
#
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

def get_data(cell_data,lookup_data):
    return 

data = pd.merge(cell_data,
                lookup_data,
                on='Well Name')

