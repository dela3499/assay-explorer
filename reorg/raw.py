import pandas as pd
import numpy as np

from toolz import \
    thread_first,\
    thread_last,\
    juxt,\
    curry
    
from utils import \
    curry_funcs,\
    identity,\
    map,\
    drop_matching_columns,\
    add_normalized_columns,\
    generate_sid,\
    mapdict,\
    tail,\
    from_file
    
from cStringIO import \
    StringIO

from pandas import \
    DataFrame as df    
            
curry_funcs(['pd.read_csv',
             'df.dropna',
             'df.rename'])

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

# String -> CellConfig -> DataFrame
def get_plate_data_from_file_with_multiple_plates(path,c):
    return thread_first(path,
                        open,
                        file.read,
                        (str.replace,'\r',''),
                        (str.split,c['plate_delimiter']),
                        tail,
                        map(StringIO),
                        map(pd.read_csv(delimiter=c['delimiter'], skiprows=c['skiprows'])),
                        pd.concat,
                        df.dropna(axis=1,how='all'),
                        (drop_matching_columns,c['dropcols']),
                        df.rename(columns=c['colrename']),
                        (add_normalized_columns,c['normcols']))

# String -> CellConfig -> DataFrame
def get_plate_data(path,c):
    """ Get plate data, drop empty columns, drop selected columns, 
        rename columns, add normalized columns. """
    return thread_first(path,
                        from_file,
                        (str.replace,'\r',''),
                        StringIO,
                        pd.read_csv(delimiter=c['delimiter'], skiprows=c['skiprows']),
                        df.dropna(axis=1,how='all'),
                        (drop_matching_columns,c['dropcols']),
                        df.rename(columns=c['colrename']),
                        (add_normalized_columns,c['normcols']))