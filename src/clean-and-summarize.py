import pandas as pd
import numpy as np
from moldev_utils import *
#execfile('../src/moldev_utils.py')

# Input Filepaths
cells_path, conds_wells_path = get_big_and_small_files('../input/')

# Import data from CSV files
cells_raw = pd.read_csv(cells_path,skiprows=4)
conds_raw = pd.read_csv(conds_wells_path,skiprows=[1])

# Drop unwanted columns and rename remaining ones
cells = drop_unwanted_cols(cells_raw.dropna(axis=1,how='all')).rename(columns=rename_col)
conds = create_condition_lookup(conds_raw.dropna(axis=1,how='all'))

# Add new derived columns
derived_cols = [['Normalized APB spots',['# of APBs'],['# of FITC spots', '# of TxRed spots']],
                ['Normalized Coloc spots',['# of Coloc Spots'],['# of FITC spots', '# of TxRed spots']],
                ['Normalized Coloc area',['Area of Coloc spots'],['Area of FITC-TxRed spots']]]

cells = thread_first(cells,
                     (normalize_by_division,()))


# a -> (a -> a) -> [[b]] -> a
# def thread_first_repeat(x,f,args):
#   """ Execute thread first with f applied once for each set of args. """



# Check that no wells are listed more than once in table
assert len(conds['Well Name']) == len(conds['Well Name'].unique()), \
  'Found duplicate well names in {}\n'.format(conds_wells_path) + \
  get_repeat_error_message(conds)

# Add condition column to cell data via join (and move to 2nd position)
data = move_column(pd.merge(cells,conds,on='Well Name'),
                   'Condition',1)

# Create summaries
def summarize_conditions(df):
  """ Return summary of data grouped by condition. """
  raw_summary = df.groupby('Condition').describe()

  # Add cell count column
  sizes = df.groupby('Well Name').size().reset_index().rename(columns={0:'Cell Count'})
  counts = pd.merge(conds,sizes,on='Well Name').groupby('Condition').describe()
  raw_summary = pd.merge(raw_summary,counts,left_index=True,right_index=True)

  summary = raw_summary \
              .reset_index(level = [0,1]) \
              .rename(columns={'level_1': 'Function'}) \
              .append(get_sem_df(raw_summary).rename(columns={'Group':'Condition'})) \
              .sort(['Condition','Function'])
  summary = move_columns(summary,[['Condition',0],['Function',1]])    
  return summary

def summarize_wells(df):
  """ Return summary of data grouped by well. """
  raw_summary = df.groupby('Well Name').describe()

  # Add cell count column
  counts = df.groupby('Well Name').size().reset_index().rename(columns={0:'Cell Count'})
  counts['Function'] = 'mean'
  
  summary = raw_summary \
              .reset_index(level = [0,1]) \
              .rename(columns={'level_1': 'Function'}) \
              .append(get_sem_df(raw_summary).rename(columns={'Group':'Well Name'})) \
              .merge(counts,on=['Well Name','Function'],how='left') \
              .sort(['Well Name','Function'])
  summary = move_columns(summary,[['Well Name',0],['Function',1]])
    
  # Add condition column to well data via join (and move to 2nd position)
  return move_column(pd.merge(summary,conds,on='Well Name'),
                              'Condition',1)
  
condition_summary = summarize_conditions(data)
well_summary = summarize_wells(data)

# Write to files
data.to_csv('../output/moldev_cleaned.csv',index=False)
well_summary.to_csv('../output/well_summary.csv',index=False)
condition_summary.to_csv('../output/condition_summary.csv',index=False)
condition_summary.T.to_csv('../output/condition_summary_transpose.csv',header=False)