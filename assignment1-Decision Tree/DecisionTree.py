import pandas as pd
from pprint import pprint
from collections import Counter
from random import randint
import copy, sys

def entropy_calculator(probs):  
    import math
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

def calculate_entire_set_entropy(dataset):
    count_pos_neg = Counter(x for x in dataset)
    class_total_instances = len(dataset)*1.0
    probs = [x / class_total_instances for x in count_pos_neg.values()]
    return entropy_calculator(probs)
    
def calculate_information_gain(dataset,information_gain_attr,final_attr):
    df_split = dataset.groupby(information_gain_attr)
    nobs = len(dataset.index) * 1.0
    df_agg_ent = df_split.agg({final_attr : [calculate_entire_set_entropy, lambda x: len(x)/nobs] })[final_attr]
    df_agg_ent.columns = ['Entropy', 'PropObservations']    
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = calculate_entire_set_entropy(dataset[final_attr])
    return old_entropy - new_entropy

def unique(seq, return_counts=False, id=None):
   found = set()
   if id is None:
       for x in seq:
           found.add(x)
   else:
       for x in seq:
           x = id(x)
           if x not in found:
               found.add(x)
   found = list(found)          
   counts = [seq.count(0),seq.count(1)]
   if return_counts:
       return found,counts
   else:
       return found
   
def addition(data):
   sum = 0
   for i in data:
       sum = sum + i
   return sum

def calculate_variance(target_values):
   values = list(target_values)
   elements,counts = unique(values,True)
   variance_impurity = 0
   sum_counts = addition(counts)
   for i in elements:
       variance_impurity += (-counts[i]/sum_counts*(counts[i]/sum_counts))
   return variance_impurity

def calculate_variance_information_gain(dataset,information_gain_attr,final_attr):
    df_split = dataset.groupby(information_gain_attr)
    nobs = len(dataset.index) * 1.0
    df_agg_ent = df_split.agg({final_attr : [calculate_variance, lambda x: len(x)/nobs] })[final_attr]
    df_agg_ent.columns = ['Variance', 'VarObservation']    
    new_varianegain = sum( df_agg_ent['Variance'] * df_agg_ent['VarObservation'] )
    old_variancegain = calculate_variance(dataset[final_attr])
    return old_variancegain - new_varianegain

def id3_algorithm(df, target_attribute_name, attribute_names,impurity,default_class=None):
    
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])
    global node_number_info
    if len(cnt) == 1:
        return next(iter(cnt))  
    elif df.empty or (not attribute_names):
        return default_class  
    else:
        default_class = max(cnt.keys())
        gainz = 0
        if(impurity == "IG"):
            gainz = [calculate_information_gain(df, attr, target_attribute_name) for attr in attribute_names] 
        else:
            gainz = [calculate_variance_information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = gainz.index(max(gainz)) 
        best_attr = attribute_names[index_of_max]
        tree = {best_attr:{}} 
        positiveCount = df['Class'].value_counts()[1];
        negativeCount = df['Class'].value_counts()[0];
        if positiveCount>negativeCount :
            best_class = 1
        elif positiveCount<negativeCount:
            best_class = 0
        else:
            best_class = 'none'
        tree[best_attr]["best_class"] = best_class
        node_number_info = node_number_info + 1
        tree[best_attr]["number"] = node_number_info
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3_algorithm(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,impurity,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree

def classify(instance, tree, default=None):
    attribute = next(iter(tree)) 
    if instance[attribute] in tree[attribute].keys():  
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict): 
            return classify(instance, result)
        else:
            return result 
    else:
        return default
    
#Main Method 
        
#H1 NP Train Accuracy
if(len(sys.argv) != 6):
   sys.exit("Please provide input in format: .\ProgramName <training-set> <validation-set> <test-set> <to-print> to-print:{yes,no} <prune> prune:{yes:no}")
else:
   ifprint = sys.argv[4]
   ifprune = sys.argv[5]
   print("Please wait until you see END")
   print("D1 or D2")
   dataset = pd.read_csv(sys.argv[1])
   node_number_info = 0
   attribute_names = list(dataset.columns)
   attribute_names.remove('Class') 
   tree = id3_algorithm(dataset,'Class',attribute_names,"IG")
   if(ifprint == "yes"):
       print("Printing tree with IG")
       pprint(tree)
   dataset['predicted'] = dataset.apply(classify, axis=1, args=(tree,'No') )  
   print('\n H1 NP Train Accuracy is: ' + str( sum(dataset['Class']==dataset['predicted'] ) / (1.0*len(dataset.index)) ))
   #print(tree)
   #H1 NP Test Accuracy
   dataset = pd.read_csv(sys.argv[3])
   node_number_info = 0
   attribute_names = list(dataset.columns)
   attribute_names.remove('Class') 
   tree = id3_algorithm(dataset,'Class',attribute_names,"IG")
   dataset['predicted'] = dataset.apply(classify, axis=1, args=(tree,'No') )  
   print('\n H1 NP Test Accuracy is: ' + str( sum(dataset['Class']==dataset['predicted'] ) / (1.0*len(dataset.index)) ))
   
   #H1 NP validation Accuracy
   dataset = pd.read_csv(sys.argv[2])
   node_number_info = 0
   attribute_names = list(dataset.columns)
   attribute_names.remove('Class') 
   tree = id3_algorithm(dataset,'Class',attribute_names,"IG")
   dataset['predicted'] = dataset.apply(classify, axis=1, args=(tree,'No') )  
   print('\n H1 NP Validation Accuracy is: ' + str( sum(dataset['Class']==dataset['predicted'] ) / (1.0*len(dataset.index)) ))
   
   #H2 NP validation Accuracy
   dataset = pd.read_csv(sys.argv[1])
   node_number_info = 0
   attribute_names = list(dataset.columns)
   attribute_names.remove('Class') 
   tree = id3_algorithm(dataset,'Class',attribute_names,"VI")
   dataset['predicted'] = dataset.apply(classify, axis=1, args=(tree,'No') )  
   print('\n H2 NP Train Accuracy is: ' + str( sum(dataset['Class']==dataset['predicted'] ) / (1.0*len(dataset.index)) ))
   
   #H2 NP validation Accuracy
   dataset = pd.read_csv(sys.argv[2])
   node_number_info = 0
   attribute_names = list(dataset.columns)
   attribute_names.remove('Class') 
   tree = id3_algorithm(dataset,'Class',attribute_names,"VI")
   dataset['predicted'] = dataset.apply(classify, axis=1, args=(tree,'No') )  
   print('\n H2 NP Test Accuracy is: ' + str( sum(dataset['Class']==dataset['predicted'] ) / (1.0*len(dataset.index)) ))
   
   #H2 NP validation Accuracy
   dataset = pd.read_csv(sys.argv[3])
   node_number_info = 0
   attribute_names = list(dataset.columns)
   attribute_names.remove('Class') 
   tree = id3_algorithm(dataset,'Class',attribute_names,"VI")
   dataset['predicted'] = dataset.apply(classify, axis=1, args=(tree,'No') )  
   print('\n H2 NP Validation Accuracy is: ' + str( sum(dataset['Class']==dataset['predicted'] ) / (1.0*len(dataset.index)) ))
   
   if(ifprint == "yes"):
       print("Printing tree with VI")
       pprint(tree)
   print("END")
   #print(tree)
#   tree = id3_algorithm(dataset,'Class',attribute_names,"IG")
#   test_data = pd.read_csv('test_set.csv')
#   tree3 = post_prune(2,4,tree)
#   print(tree3)
#
#   test_data['predicted3'] = test_data.apply(classify, axis=1, args=(tree3,'1') ) 
#   print( 'Accuracy with pruned Information gain tree ' + (str( sum(test_data['Class']==test_data['predicted3'] ) / (1*len(test_data.index)) )))
   










    