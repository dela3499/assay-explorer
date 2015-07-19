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
                      df)

# TODO: think about generating summary statistics on the fly, rather than generating them all at once here. I think that might end up being simpler overall. 

# DataFrame -> [String] -> DataFrame
def summarize_wells(dataframe,groups):
    parameters = groupby_and_summarize(dataframe,groups,funcs,fnames)
    cell_counts = get_well_cell_counts(dataframe)
    cell_counts['Function'] = 'avg'
    cell_counts = cell_counts.drop('Condition',axis=1)
    return pd.merge(parameters,
                    cell_counts,
                    on=['Plate ID','Well Name','Function'],
                    how='left')

# DataFrame -> [String] -> DataFrame
def summarize_conditions(dataframe,groups):
    return thread_last(dataframe,
                       juxt(identity,get_well_cell_counts),
                       lambda x: [x[0].drop(['Plate ID','Plate Name','Well Name','Date'],axis=1),
                                  x[1].drop(['Plate ID','Well Name'],axis=1)],
                       (map,lambda x: groupby_and_summarize(x,
                                                            groups,
                                                            funcs,
                                                            fnames)),
                       lambda x: pd.merge(*x,on=['Condition','Function']))