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

# Check that no wells are listed more than once in table
assert len(conds['Well Name']) == len(conds['Well Name'].unique()), \
  'Found duplicate well names in {}\n'.format(conds_wells_path) + \
  get_repeat_error_message(conds)

# Add condition column to cell data via join (and move to 2nd position)
data = move_column(pd.merge(cells,conds,on='Well Name'),
                   'Condition',1)

# Create summaries (group by condition or well)
def summarize(df, col):
  """ Return summary of data grouped by col. """
  raw_summary = df.groupby(col).describe()

  # Add cell count column
  if col == 'Condition': 
    sizes = df.groupby('Well Name').size().reset_index().rename(columns={0:'Cell Count'})
    counts = pd.merge(conds,sizes,on='Well Name').groupby('Condition').describe()
    raw_summary = pd.merge(raw_summary,counts,left_index=True,right_index=True)   
  
  summary = raw_summary \
              .reset_index(level = [0,1]) \
              .rename(columns={'level_1': 'Function'}) \
              .append(get_sem_df(raw_summary).rename(columns={'Group':col})) \
              .sort([col,'Function'])
  summary = move_columns(summary,[[col,0],['Function',1]])
    
  # Add condition column to well data via join (and move to 2nd position)
  if col == 'Well Name':
      return move_column(pd.merge(summary,conds,on='Well Name'),
                                   'Condition',1)
  else:
      return summary

condition_summary = summarize(data,'Condition')
well_summary = summarize(data,'Well Name')

# Write to files
data.to_csv('../output/moldev_cleaned.csv',index=False)
well_summary.to_csv('../output/well_summary.csv',index=False)
condition_summary.to_csv('../output/condition_summary.csv',index=False)
condition_summary.T.to_csv('../output/condition_summary_transpose.csv',header=False)