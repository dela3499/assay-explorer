from pandas import DataFrame as df
import pandas as pd
import numpy as np
import re
from toolz import thread_first,\
                  thread_last,\
                  curry

# a -> Int -> [a]
def repeat(x,n):
    """ Return list with x repeated n times. """
    return [x for _ in range(n)]

def reset_index(dataframe):
    return dataframe.reset_index(drop=True)

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

@curry
def summarize(dataframe,funcs = [],names = []):
    summary = pd.concat([df(f(dataframe)).T for f in funcs])
    summary['Function'] = names

    # Set string column names (since they can't be aggregated)
    for col in get_string_columns(dataframe):
      vals = list(set(dataframe[col].values))
      assert len(vals) == 2, \
        "Values in '{}' column should be the same in a group. Found multiple values: {}".format(col,vals)
      summary[col] = dataframe[col].iloc[0]

    return summary.reset_index(drop=True)

def get_string_columns(dataframe):
    """ Return columns with string values."""
    return [col for col in dataframe.columns \
              if dataframe[col].dtype == 'object']


def summarize_groups(groups,funcs = [],names = []):
    return thread_last(groups,
                       (map, snd),
                       (map,summarize(funcs = funcs, 
                                      names = names)),
                       pd.concat,
                       reset_index)