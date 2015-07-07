
path = ""
well_save_path = ""
condition_save_path = ""

funcs = [df.mean,df.std,df.sem,df.count,df.min,df.max]
fnames = ['avg','std','sem','count','min','max']

condition_config = dict(groupby = 'Condition', funcs = funcs, fnames = fnames)
well_config = dict(groupby = ['Plate ID','Well Name'], funcs = funcs, fnames = fnames)

def get_well_cell_counts(dataframe):
    return thread_last(dataframe.groupby('Well Name'),
                      (map,lambda x: {"Well Name": x[0],
                                      "Cell Count": len(x[1]),
                                      "Condition": x[1]['Condition'].iloc[0]}),
                      df)

# DataFrame -> WellSummaryConfig -> DataFrame
def summarize_wells(dataframe,c):
    parameters = groupby_and_summarize(dataframe,c['groupby'],c['funcs'],c['fnames'])
    cell_counts = get_well_cell_counts(dataframe)
    cell_counts['Function'] = 'avg'
    cell_counts = cell_counts.drop('Condition',axis=1)
    return pd.merge(parameters,
                    cell_counts,
                    on=['Well Name','Function'],
                    how='left')

# DataFrame -> ConditionSummaryConfig -> DataFrame
def summarize_conditions(dataframe,c):
    return thread_last(dataframe,
                       juxt(identity,get_well_cell_counts),
                       (map,lambda x: groupby_and_summarize(x,
                                                            c['groupby'],
                                                            c['funcs'],
                                                            c['fnames'])),
                       lambda x: pd.merge(*x,on=['Condition','Function']))

data = pd.read_csv(path)

well_summary = summarize_wells(data,well_config)
condition_summary = summarize_conditions(data,condition_config)

# Write to files
well_summary.to_csv(well_save_path,index=False)
condition_summary.to_csv(condition_save_path,index=False)