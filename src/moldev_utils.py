import re 
import pandas as pd
import numpy as np
import os
import functools
from toolz import thread_first, assoc

# TODO: improve the way that SEM is calculated and added to data

# [[a]] -> [a]
def concatenate(x):
    "Concatenate sublists into single list."
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        return x[0] + x[1]
    else: 
        return x[0] + x[1] + concatenate(x[2:])

# String -> [{path:String, size:Num}]
def get_csv_files(path):
    """ Return dictionary of filepaths and filesizes for csv files 
        in specified directory. """
    csv_files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.csv')]
    return [{"path": f, "size": os.path.getsize(f)} for f in csv_files]

# String -> [String]
def get_big_and_small_files(path):
    "Return paths to biggest and smallest csv files in provided directory."
    files = get_csv_files(path)

    assert len(files) == 2, "Input folder should contain only two csv files."

    if files[0]['size'] > files[1]['size']:
        return files[0]['path'], files[1]['path']
    else: 
        return files[1]['path'], files[0]['path']

# String -> [Regex] -> Boolean
def matches_any_pattern(s,patterns):
    """ Return True if any of the patterns matches string s. """
    return any([re.search(pattern,s) for pattern in patterns])

# DataFrame -> DataFrame
def drop_unwanted_cols(df):
    """ Drop columns from dataframe if useless or duplicate (ends in '.n') """
    cols = df.columns
    patterns = ['Cell ID','Site ID','MEASUREMENT SET ID','.*ObjectID.*','\.[0-9]*\Z']
    unwanted_cols = [col for col in cols if matches_any_pattern(col,patterns)]
    return df.drop(unwanted_cols,axis=1)

# String -> String
def rename_col(col):
    """ Rename column col to remove whitespace, backslashes, prefixes, 
        and suffixes (esp. large parenthetic suffix). """
    if col.startswith('Cell:'):
        return col.split('(')[0].lstrip("Cell:").rstrip('/').strip(' ')
    else: 
        return col.split('(')[0].rstrip('/').strip(' ')

# Series -> DataFrame
def header_to_column(series):
    """ Given a Series, return a Dataframe with columns: value, label.
    Created by associating header with each value in Series. """
    s = series.dropna() # nans should be ignored
    return pd.DataFrame(dict(values = s.values,
                             label = s.name))

# DataFrame -> DataFrame
def headers_to_column(df):
    """ Given Dataframe, return new Dataframe with two columns: value, label. 
    Created by taking each value in table, and pairing it with its column name. """
    return pd.concat([header_to_column(df[col]) for col in df.columns])

# DataFrame -> String -> Int -> DataFrame
def move_column(df,col,pos):
    """ Return new Dataframe where col in moved to pos.
    if pos is 0, then first column becomes col, and all other columns are shifted right. """
    other_cols = [c for c in df.columns if c != col]
    return df[other_cols[0:pos] + [col] + other_cols[pos:]]

# DataFrame -> [[String,Int]] -> DataFrame
def move_columns(df,cols):
    """ Return DataFrame with columns moved to specified locations.
        Input:
        df - dataframe
        cols - [string,int] """
    cols = sorted(cols,key=lambda x1: x1[1])
    new_df = df
    for name,loc in cols:
        new_df = move_column(new_df,name,loc)
    return new_df

# DataFrame -> DataFrame
def create_condition_lookup(df):
    """ Reshape conditions table into 2-column lookup.
        Input: Dataframe where columns names are conditions 
               and values are well names
        Output: Dataframe with two columns: Well Name, Condition """
    return headers_to_column(df) \
                .reset_index(drop=True) \
                .rename(columns=dict(values = 'Well Name',
                                     label = 'Condition'))
# [a] -> [a]
def get_duplicates(x):
    """ Return any elements that appear more than once in x. """
    return list(set([xi for xi in x if x.count(xi) > 1]))

# DataFrame -> String
def get_repeat_error_message(df):
    "Return message describing where repeats are located."
    wells = get_duplicates(df['Well Name'].values.tolist())
    def get_message(df,well):
        conditions = df[df['Well Name'] == well]['Condition'].values.tolist()
        return "Well {} is listed in conditions: {}".format(well,', '.join(conditions))
    return '\n'.join([get_message(df,well) for well in wells])

# DataFrame -> String -> Series
def get_sem(df,group):
    """ Return standard error of mean for given group as Series. """
    s = df.loc[group,'std'] / df.loc[group,'count'].map(np.sqrt)
    return s.append(pd.Series({'Group':group, 'Function':'sem'}))

# DataFrame -> [Series]
def get_sems(df):
    """ Return list of Series, where each Series contains SEM for each parameter. """
    return [get_sem(df,group) for group in set([i[0] for i in df.index])]    

# DataFrame -> DataFrame
def get_sem_df(df):
    "Return new DataFrame with row containing standard error of mean for each condition."
    return pd.concat(get_sems(df), axis = 1).T

# [a] -> [a]
def reverse(x):
    return list(reversed(x))    

# DataFrame -> String -> [String] -> [String] -> DataFrame
def normalize_by_division(df,newcol,numerator_cols,denominator_cols):
  """ Return new DataFrame, where newcol = sum(numerator_cols)/sum(denominator_cols)"""
  numerator = df[numerator_cols].apply(sum, axis = 1)
  denominator = df[denominator_cols].apply(sum, axis = 1)
  new_df = df.copy()
  new_df[newcol] = numerator / denominator
  return new_df    

# a -> Int -> [a]
def repeat(x,n):
  """ Return list with x repeated n times. """
  return [x for _ in range(n)]

