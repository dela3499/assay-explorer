from cStringIO import StringIO
from pandas import DataFrame as df
import pandas as pd
import numpy as np
from toolz import thread_first,\
                  thread_last,\
                  juxt,\
                  curry
from utils import curry_funcs,\
                  identity,\
                  map,\
                  drop_matching_columns,\
                  add_normalized_columns,\
                  generate_sid,\
                  mapdict,\
                  tail
            
curry_funcs(['pd.read_csv',
             'df.dropna',
             'df.rename'])

# String -> String
def rename_column(col):
    """ Rename column col to remove whitespace, backslashes, prefixes,
        and suffixes (esp. large parenthetic suffix). """
    if col.startswith('Cell:'):
        return col.split('(')[0].lstrip("Cell:").rstrip('/').strip(' ')
    else:
        return col.split('(')[0].rstrip('/').strip(' ')

cell_config = dict(
    plate_delimiter = "ATF",
    delimiter = '\t',
    skiprows = 4,
    dropcols = ['Cell ID',
                'Site ID',
                'MEASUREMENT SET ID',
                '.*ObjectID.*',
                'Laser focus score',
                '\.[0-9]*\Z'],
    normcols = [['Normalized_ColocSpot_area_sum (coloc)',
                  ['ColocSpots_area_sum'],
                  ['FITC-TxRed_coloc_area_sum']],
                ['Normalized_ColocSpot_area_sum (all)',
                  ['ColocSpots_area_sum'],
                  ['FITC-TxRed_all_area_sum']],
        
                ['Normalized coloc spots (by FITC & TxRed)',
                  ['# of Coloc Spots'],
                  ['# of FITC spots', '# of TxRed spots']],
                ['Normalized coloc spots (by FITC)',
                  ['# of Coloc Spots'],
                  ['# of FITC spots']],
                ['Normalized coloc spots (by TxRed)',
                  ['# of Coloc Spots'],
                  ['# of TxRed spots']],
               
                ['Normalized coloc spots (by FITC in coloc)',
                  ['# of Coloc Spots'],
                  ['# of FITC in ColocSpots']],
                ['Normalized coloc spots (by TxRed in coloc)',
                  ['# of Coloc Spots'],
                  ['# of TxRed in ColocSpots']],
                ['Normalized coloc spots (by FITC-TxRed in coloc)',
                  ['# of Coloc Spots'],
                  ['# of FITC-TxRed in ColocSpots']]],

    
    colrename = rename_column,
    check = identity
    )

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
                        (str.replace,'\r',''),
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