from pandas import DataFrame as df
import pandas as pd
import numpy as np
import json
import uuid
import re
from toolz import \
    thread_first,\
    thread_last,\
    curry,\
    assoc
                    
# a -> a
def identity(x):
    return x

# (a -> b) -> [a] -> [b]
@curry
def map(f,x):
    """ Apply f to every element in x and return the result. """
    return [f(xi) for xi in x]

# (a -> b -> c) -> [a] -> [b] -> [c]
@curry
def map2(f,x,y):
    return [f(xi,yi) for xi,yi in zip(x,y)]

# (Int -> a -> b) -> [a] -> [b]
@curry
def indexed_map(f,x):
    """ Map a function over a list, including the index as the first argument. """
    return [f(i,xi) for i,xi in zip(range(len(x)),x)]

# a -> Int -> [a]
def repeat(x,n):
    """ Return list with x repeated n times. """
    return [x for _ in range(n)]

# [[a]] -> [a]
def concatenate(x):
    "Concatenate sublists into single list."
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        return x[0] + x[1]
    else: 
        return x[0] + x[1] + concatenate(x[2:])

# DataFrame -> DataFrame
def reset_index(dataframe):
    return dataframe.reset_index(drop=True)

# (a -> b -> c) -> {a:b} -> [c] 
@curry
def mapdict(f,d):
    """ Map f over key-value pairs of dictionary d. """
    return [f(k,v) for k,v in d.iteritems()]

# [a] -> [a]
def tail(x):
    """ Return list x without first element. """
    return x[1:]

# Random -> Int
def generate_sid():
    return str(uuid.uuid4()).split('-')[-1]

# [String] -> SideEffect
def curry_funcs(funcs):
    """ Curry each function in provided list. """
    for func in funcs:
      try: 
        exec('global {}; {} = curry({})'.format(*[func]*3))
      except:
        exec('{} = curry({})'.format(*[func]*2))

# a -> (a -> [b] -> a) -> [[b]] -> a
def thread_first_repeat(x,f,args):
    """ Execute thread first with f applied once for each set of args. """
    # Need to improve the documentation for this function, and maybe change its implementation.
    # It's really confusing. Try using foldl. I think that's the better option.
    return thread_first(x,*map2(lambda x,y: tuple([x] + y),
                               repeat(f,len(args)),
                               args))

# DataFrame -> String -> (a | [a] | Series[a])
def add_col(dataframe,colname,values):
    "Add column to dataframe with given values."
    dataframe[colname] = values
    return dataframe

def filter_and_drop(df,col,val):
    """ Return DataFrame with rows that match filter. Filter column is dropped. """
    return df[df[col] == val].drop([col],axis=1)

@curry
def normalize_columns(df,fillna=False):
    """ Return new DataFrame, where the norm of each column is the unit value. """
    if fillna :
        df = df.fillna(0)
    return df.apply(lambda x: x.values/np.linalg.norm(x.values))

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

# Num -> Num
def inc(x):
    """ Increment the value of x. """
    return x + 1

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
""" Use: list_a |are_all_in| list_b """
are_all_in = Infix(lambda a,b: is_empty(set(a).difference(set(b))))

# String -> {a:b} -> SideEffects[File]
@curry
def set_model(filepath,k,v):
    """ Save key k and value v to json file. """
    f = open(filepath,'r+')
    fstring = f.read()
    try:
        data = json.loads(fstring)
    except:
        data = dict()
    f.close()
    g = open(filepath,'w+')
    new_data = assoc(data,k,v)
    new_json_data = json.dumps(new_data)
    g.write(new_json_data)
    g.close()
    
# String -> Widget -> a -> SideEffects[File]
@curry
def persist_widget_value(filepath,widget,key):
    """ Save widget value to key in JSON file. """
    widget.on_trait_change(lambda name,value: set_model(filepath,key,value),'value')

# {a:b} -> a -> b
@curry
def maybe_get(d,k,v):
    """ Given a dictionary d, return value corresponding to key k. 
        If k is not present in dictionary, return v. """
    try:
        return d[k]
    except:
        return v    
    
# String -> a -> b
@curry
def maybe_get_model(filepath,k,v):
    """ Try to load json file at filepath and return value associated with key k.
        If this fails (file isn't present or key is absent), then return v. """
    try: 
        f = open(filepath)
        fstring = f.read()
        data = json.loads(fstring)
        f.close()
        return data[k]
    except:
        return v

# DataFrame -> String -> [String] -> [String] -> DataFrame
def normalize_by_division(dataframe,newcol,numerator_cols,denominator_cols):
    """ Return new DataFrame, where newcol = sum(numerator_cols)/sum(denominator_cols)"""
    numerator = dataframe[numerator_cols].apply(sum, axis = 1)
    denominator = dataframe[denominator_cols].apply(sum, axis = 1)
    new_df = dataframe.copy()
    new_df[newcol] = numerator / denominator
    return new_df    

def filter_rows(df,col,val):
    """ Return new DataFrame where the values in col match val. 
        val may be a single value or a list of values. """
    if type(val) == list:
        return df[df[col].isin(val)]
    else:
        return df[df[col] == val]

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