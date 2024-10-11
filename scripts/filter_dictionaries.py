# filter the processed PhageScope dictionaries to only contain a representative from each cluster 


# imports 
import pickle 
import pandas as pd 
import glob 
import re 

# read in the data 
X_inputs = glob.glob('/home/grig0076/scratch/Phynteny_transformer/PhageScope/processed_all_PhageScope_08102024/transfer_09102024/*.X.pkl') 

# get the data that passed clustering with mmseqs 
dbs = glob.glob('/home/grig0076/scratch/databases/PhageScope/metadata/*')
all_meta = pd.DataFrame() 

for m in dbs: 
    meta = pd.read_csv(m, sep = '\t')
    meta['db'] = [re.split('/',m)[-1] for i in range(len(meta))]
    
    all_meta = pd.concat([all_meta, meta]) 
meta_df_representatives = all_meta.drop_duplicates(subset = 'Cluster')
meta_representatves = meta_df_representatives['Phage_ID'].to_list() 

# loop through the chunked dictionaries and keep only the data that passes 
# creat a loop to read in each of the data dictionaries 
X_representatives = dict() 
y_representatives = dict() 

# repeat for the y values as well 
for X in X_inputs:
    
    print(X)
    
    X_d = pickle.load(open(X, 'rb'))
    y_d = pickle.load(open(X[:-5] + 'y.pkl', 'rb'))
    keys = list(X_d.keys()) 
    
    # get the keys to keep 
    keep_keys = set(X_d.keys()).intersection(set(meta_representatves)) 
    keep_keys = list(keep_keys)
    
    # make dictionary of the entries to keep 
    X_keep_dict = dict(zip(keep_keys, [X_d.get(k) for k in keep_keys]))
    y_keep_dict = dict(zip(keep_keys, [y_d.get(k) for k in keep_keys]))
    
    # update
    X_representatives.update(X_keep_dict)
    y_representatives.update(y_keep_dict)
    
    del X_d
    del y_d 

# save the dictionaries containing only the representatives 
pickle.dump(X_representatives, open('/home/grig0076/scratch/Phynteny_transformer/PhageScope/processed_all_PhageScope_08102024/PhageScope_08102024_representatives.X.pkl', 'wb'))
pickle.dump(y_representatives, open('/home/grig0076/scratch/Phynteny_transformer/PhageScope/processed_all_PhageScope_08102024/PhageScope_08102024_representatives.y.pkl', 'wb'))

