from pandas import DataFrame as df
import pandas as pd
import numpy as np
from toolz import thread_first,\
                  thread_last,\
                  juxt
from utils import curry_funcs,\
                  drop_matching_columns,\
                  add_normalized_columns,\
                  headers_to_column,\
                  groupby_and_summarize,\
                  identity

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
### Summary Config ################################
###################################################

funcs = [df.mean,df.std,df.sem,df.count,df.min,df.max]
fnames = ['avg','std','sem','count','min','max']

condition_config = dict(groupby = 'Condition', funcs = funcs, fnames = fnames)
well_config = dict(groupby = 'Well Name', funcs = funcs, fnames = fnames)


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

def get_well_cell_counts(dataframe):
    return thread_last(dataframe.groupby('Well Name'),
                      (map,lambda x: {"Well Name": x[0],
                                      "Cell Count": len(x[1]),
                                      "Condition": x[1]['Condition'].iloc[0]}),
                      df)

# DataFrame -> WellSummaryConfig -> DataFrame
def summarize_wells(dataframe,c):
    parameters = groupby_and_summarize(dataframe,c['groupby'],c['funcs'],c['fnames'])
    cell_counts = get_well_cell_counts(dataframe)
    cell_counts['Function'] = 'avg'
    cell_counts = cell_counts.drop('Condition',axis=1)
    return pd.merge(parameters,
                    cell_counts,
                    on=['Well Name','Function'],
                    how='left')

# DataFrame -> ConditionSummaryConfig -> DataFrame
def summarize_conditions(dataframe,c):
    return thread_last(dataframe,
                       juxt(identity,get_well_cell_counts),
                       (map,lambda x: groupby_and_summarize(x,
                                                            c['groupby'],
                                                            c['funcs'],
                                                            c['fnames'])),
                       lambda x: pd.merge(*x,on=['Condition','Function']))

data = pd.merge(get_cell_data(cell_config),
                get_lookup_data(lookup_config),
                on = 'Well Name')

well_summary = summarize_wells(data,well_config)
condition_summary = summarize_conditions(data,condition_config)

# Write to files
data.to_csv('../output/moldev_cleaned.csv',index=False)
well_summary.to_csv('../output/well_summary.csv',index=False)
condition_summary.to_csv('../output/condition_summary.csv',index=False)


# rewrite agg functions to be a dict, rather than two lists. It's a lot to pass around through three or four functions. 
# Currying and named arguments don't mix well