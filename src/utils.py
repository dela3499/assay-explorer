from pandas import DataFrame as df
import pandas as pd
import numpy as np
import re
from toolz import thread_first,\
                  thread_last,\
                  curry

# a -> a
def identity(x):
    return x

# (a -> b) -> [a] -> [b]
@curry
def map(f,x):
    return [f(xi) for xi in x]

@curry
# (Int -> a -> b) -> [a] -> [b]
def indexed_map(f,x):
    """ Map a function over a list, including the index as the first argument. """
    return [f(i,xi) for i,xi in zip(range(len(x)),x)]

# a -> Int -> [a]
def repeat(x,n):
    """ Return list with x repeated n times. """
    return [x for _ in range(n)]

# DataFrame -> DataFrame
def reset_index(dataframe):
    return dataframe.reset_index(drop=True)

# (a,b) -> b
snd = lambda x: x[1]  

def curry_funcs(funcs):
    for func in funcs:
      try: 
        exec('global {}; {} = curry({})'.format(*[func]*3))
      except:
        exec('{} = curry({})'.format(*[func]*2))

# a -> (a -> [b] -> a) -> [[b]] -> a
def thread_first_repeat(x,f,args):
    """ Execute thread first with f applied once for each set of args. """
    return thread_first(x,*map(lambda x,y: tuple([x] + y),
                               repeat(f,len(args)),
                               args))

# String -> [Regex] -> Boolean
def matches_any_pattern(s,patterns):
    """ Return True if any of the patterns matches string s. """
    return any([re.search(pattern,s) for pattern in patterns])

# DataFrame -> DataFrame
def drop_matching_columns(dataframe,patterns):
    """ Drop columns from dataframe if they match any pattern. """
    matching_columns = [col for col in dataframe.columns
    						if matches_any_pattern(col,patterns)]
    return dataframe.drop(matching_columns,axis=1)

# DataFrame -> String -> [String] -> [String] -> DataFrame
def normalize_by_division(dataframe,newcol,numerator_cols,denominator_cols):
    """ Return new DataFrame, where newcol = sum(numerator_cols)/sum(denominator_cols)"""
    numerator = dataframe[numerator_cols].apply(sum, axis = 1)
    denominator = dataframe[denominator_cols].apply(sum, axis = 1)
    new_df = dataframe.copy()
    new_df[newcol] = numerator / denominator
    return new_df    

# type NormalizeConfig = [[String,[String],[String]]]
# DataFrame -> NormalizeConfig -> DataFrame
def add_normalized_columns(dataframe,config):
    return thread_first_repeat(dataframe,
                               normalize_by_division,
                               config)

# Series {String:[a]} -> DataFrame {value:[a],label:[String]}
def header_to_column(series):
    """ Given a Series, return a Dataframe with columns: value, label.
    Created by associating header with each value in Series. """
    s = series.dropna() # nans should be ignored
    return pd.DataFrame(dict(values = s.values,
                             label = s.name))

# DataFrame -> DataFrame
def headers_to_column(dataframe):
    """ Given Dataframe, return new Dataframe with two columns: value, label. 
    Created by taking each value in table, and pairing it with its column name. """
    reshaped_dataframes = [header_to_column(dataframe[col]) \
                            for col in dataframe.columns]
    return pd.concat(reshaped_dataframes).reset_index(drop=True)

# DataFrame -> [String]
def get_string_columns(dataframe):
    """ Return columns with string values."""
    return [col for col in dataframe.columns \
              if dataframe[col].dtype == 'object']



#   # Add cell count column
#   sizes = df.groupby('Well Name')\
#     .size()\
#     .reset_index()\
#     .rename(columns={0:'Cell Count'})

#   counts = pd.merge(conds,
#                     sizes,
#                     on='Well Name')\
#               .groupby('Condition')\
#               .describe()
  
#   raw_summary = pd.merge(raw_summary,
#                          counts,
#                          left_index=True,
#                          right_index=True)

# ###
#   counts = df.groupby('Well Name')\
#     .size()\
#     .reset_index()\
#     .rename(columns={0:'Cell Count'})

#   counts['Function'] = 'mean'

# DataFrame -> [(DataFrame -> Series)] -> [String] -> DataFrame
def multiaggregate(dataframe,funcs,fnames):
    agg = pd.concat([df(f(dataframe)).T for f in funcs])
    agg['Function'] = fnames
    return agg

# DataFrame -> [(DataFrame -> Series)] -> [String] -> DataFrame
@curry
def summarize(dataframe,funcs = [],fnames = []):
    summary = multiaggregate(dataframe,funcs,fnames)
    
    # Properly set string columns 
    # (drop columns with more than a single unique value.)
    for col in get_string_columns(dataframe):
        if dataframe[col].nunique() == 1:
            summary[col] = dataframe[col].iloc[0]
        else:
            summary = summary.drop(col,axis=1)

    return summary.reset_index(drop=True)

# GroupBy -> [(DataFrame -> Series)] -> [String] -> DataFrame
def groupby_and_summarize(dataframe,col,funcs = [],fnames = []):
    return thread_last(dataframe,
                       lambda x: x.groupby(col),
                       (map, snd),
                       (map,summarize(funcs = funcs, 
                                      fnames = fnames)),
                       pd.concat,
                       reset_index)