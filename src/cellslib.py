from pandas import DataFrame as df
import pandas as pd
import numpy as np
from toolz import thread_first,\
                  thread_last,\
                  juxt
from toolz.curried import map
execfile('../src/utils.py')                    
from cStringIO import StringIO
import uuid

curry_funcs(['pd.read_csv',
             'df.dropna',
             'df.rename'])                    
                    
def mapdict(f,d):
    return [f(k,v) for k,v in d.iteritems()]

def tail(x):
    return x[1:]

def generate_sid():
    return str(uuid.uuid4()).split('-')[-1]

def create_well_df(cell_dict):
    return thread_last(cell_dict,
                       (mapdict,lambda k,v: {"Cell Type":k,"Well Name":v}),
                       (map, df),
                       pd.concat)

def create_plate_df(well_df,plate_info, plate_name):
    x = well_df.copy()
    x['Plate Name'] = plate_name
    x['Condition'] = plate_info + ' ' + x['Cell Type']
    x['Plate ID'] = generate_sid()
    return x.drop('Cell Type',axis=1)

# CellConfig -> DataFrame
def get_cell_data(c):
    return thread_first(c['path'],
                        open,
                        file.read,
                        (str.split,c['plate_delimiter']),
                        tail,
                        map(StringIO),
                        map(pd.read_csv(delimiter=c['delimiter'], skiprows=c['skiprows'])),
                        pd.concat,
                        df.dropna(axis=1,how='all'),
                        (drop_matching_columns,c['dropcols']),
                        df.rename(columns=c['colrename']),
                        (add_normalized_columns,c['normcols']),
                        c['check'])