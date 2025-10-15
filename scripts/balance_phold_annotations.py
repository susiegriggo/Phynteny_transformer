# script for ensuring an equal representation of each class in the phold annotations 

# imports 
import pickle
import numpy as np
import torch 
import pandas as pd

# read in the phold_y and pharokka_y files 
# read in the phold_y and the pharokka_y 
phold_y = pickle.load(open('/home/grig0076/scratch/databases/PhageScope/pharokka_split_outputs/filtered_genbank/chunks/phold_data.y.pkl', 'rb'))
pharokka_y = pickle.load(open('/home/grig0076/scratch/Phynteny_transformer/PhageScope/mergeddata_07012025/merged_data.y.pkl', 'rb'))

val_labels = list(pharokka_y.keys())
val_labels = [v for v in val_labels if phold_y.get(v) != None] 
num_classes =9

# Initialize summary dictionary to track all counts
summary_counts = {
    'pharokka': {i: 0 for i in range(num_classes)},
    'phold_total': {i: 0 for i in range(num_classes)},
    'phold_not_in_pharokka': {i: 0 for i in range(num_classes)},
    'phold_balanced': {i: 0 for i in range(num_classes)}
}

# Count pharokka annotations
for v in val_labels:
    pharokka_categories = pharokka_y.get(v)
    valid_pharokka = pharokka_categories[pharokka_categories != -1]
    for p in valid_pharokka:
        summary_counts['pharokka'][int(p.detach())] += 1

# Count total phold annotations
for v in val_labels:
    phold_categories = phold_y.get(v)
    valid_phold = phold_categories[phold_categories != -1]
    for p in valid_phold:
        summary_counts['phold_total'][int(p.detach())] += 1

# loop through   
num_classes=9
category_counts = {i: 0 for i in range(num_classes)} 

# loop through each of the validation_labels 
for v in val_labels: 
    
    idx = phold_y.get(v) != pharokka_y.get(v) 
    phold_categories = phold_y.get(v)[idx] 
    for p in phold_categories: 
        category_counts[int(p.detach())] += 1
        summary_counts['phold_not_in_pharokka'][int(p.detach())] += 1


# get the minimum present as the number to use for building the ROC curves 
min_count = np.min(list(category_counts.values())) # this is the integrase class 
category_weights = 1/(list(category_counts.values())/min_count) 

# aded a one onto the end of category_weights for the unknown cateogry 
category_weights = np.append(category_weights, 1)

phold_y_balanced = {} 
for v in val_labels: 
    
    phold_original = phold_y.get(v).clone()
    
    # get the phold predictions  not present using pharokka only 
    idx = phold_y.get(v) != pharokka_y.get(v) 
    phold_categories =  phold_original[idx]  
    
    # get the probability of keeping each prediction
    phold_probs = category_weights[phold_original]
    # update phold_probs so that proteins known with pharokka are 1 
    phold_probs[torch.nonzero(pharokka_y.get(v) != -1, as_tuple=True)] =1
    
    # select which get kept using a Bernoulli distirbution 
    rand_vals = torch.rand(len(phold_probs))
    remove_idx = torch.nonzero(rand_vals > torch.tensor(phold_probs), as_tuple=True)[0] # these are the predictions that would be kept 
    
    # remove predicted categories to balance the data 
    phold_original[remove_idx] = -1 
    
    #update the dictionary 
    phold_y_balanced[v] = phold_original

balanced_category_counts = {i: 0 for i in range(num_classes)} 

# loop through each of the validation_labels 
for v in val_labels: 
    
    idx = phold_y_balanced.get(v) != pharokka_y.get(v) 
    phold_categories = phold_y_balanced.get(v)[idx] 
    for p in phold_categories: 
        balanced_category_counts[int(p.detach())] += 1
        summary_counts['phold_balanced'][int(p.detach())] += 1

# Create summary DataFrame and save as TSV
summary_df = pd.DataFrame({
    'Category': list(range(num_classes)),
    'Pharokka': [summary_counts['pharokka'][i] for i in range(num_classes)],
    'Phold_Total': [summary_counts['phold_total'][i] for i in range(num_classes)],
    'Phold_Not_In_Pharokka': [summary_counts['phold_not_in_pharokka'][i] for i in range(num_classes)],
    'Phold_Balanced': [summary_counts['phold_balanced'][i] for i in range(num_classes)]
})

summary_df.to_csv('/home/grig0076/scratch/databases/PhageScope/pharokka_split_outputs/filtered_genbank/chunks/annotation_summary.tsv', 
                  sep='\t', index=False)

# save to file 
pickle.dump(phold_y_balanced, open('/home/grig0076/scratch/databases/PhageScope/pharokka_split_outputs/filtered_genbank/chunks/phold_data_balanced.y.pkl', 'wb'))