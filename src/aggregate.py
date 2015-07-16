execfile('../src/cellslib.py')

# TODO groupby params ('Plate ID' and 'Well Name' are repeated many times, but should only be present in config.)
# TODO import statements are disorganized and spread throughout scripts. Need to create proper import statements across the board
# TODO need to create issues for these todos

funcs = [df.mean,df.std,df.sem,df.count,df.min,df.max]
fnames = ['avg','std','sem','count','min','max']

def get_well_cell_counts(dataframe):
    return thread_last(dataframe.groupby(['Plate ID','Well Name']),
                      (map,lambda x: {"Plate ID": x[0][0],
                                      "Well Name": x[0][1],
                                      "Cell Count": len(x[1]),
                                      "Condition": x[1]['Condition'].iloc[0]}),
                      list,
                      df)

# DataFrame -> WellSummaryConfig -> DataFrame
def summarize_wells(dataframe,c):
    parameters = groupby_and_summarize(dataframe,c['groupby'],funcs,fnames)
    cell_counts = get_well_cell_counts(dataframe)
    cell_counts['Function'] = 'avg'
    cell_counts = cell_counts.drop('Condition',axis=1)
    return pd.merge(parameters,
                    cell_counts,
                    on=['Plate ID','Well Name','Function'],
                    how='left')

# DataFrame -> ConditionSummaryConfig -> DataFrame
def summarize_conditions(dataframe,c):
    return thread_last(dataframe,
                       juxt(identity,get_well_cell_counts),
                       lambda x: [x[0].drop(['Plate ID','Plate Name','Well Name','Date'],axis=1),
                                  x[1].drop(['Plate ID','Well Name'],axis=1)],
                       (map,lambda x: groupby_and_summarize(x,
                                                            c['groupby'],
                                                            funcs,
                                                            fnames)),
                       lambda x: pd.merge(*x,on=['Condition','Function']))

data = pd.read_csv(path)

well_summary = summarize_wells(data,well_config)
condition_summary = summarize_conditions(data,condition_config)

# Write to files
well_summary.to_csv(well_save_path,index=False)
condition_summary.to_csv(condition_save_path,index=False)