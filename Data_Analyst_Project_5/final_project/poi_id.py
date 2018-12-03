#!/usr/bin/python

import pickle
import pprint
import numpy as np
import pandas as pd



## Load graphical packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
%matplotlib inline

## Load our helper modules.
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester
import myhelpers

## Load Scikit-Learn modules
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# Turn data_dict into dataframe, with missing values types converted. 
df = myhelpers.dict_to_df(data_dict)

# Switch to True to display corresponding information
print_data_exploration = True
print_feature_plot = True
print_feature_selection = True
print_algorithm_optimization = True
print_model_validation = True

############################################################    
## PART 1: Data Exploration
############################################################

if print_data_exploration == True:
    # print size of the dataset
    print "- Number of observations is {}".format(df.shape[0]) # + "\n"
    print "- Number of features is " + str(df.shape[1])
    
    # print number of poi's
    print "- Number of poi's is " + str(sum(df['poi']))

if print_data_exploration == True:
    
    # print the percentage of missing values per feature.
    print "- Percentage of NaN's per feature"
    pprint.pprint(myhelpers.missing_per_feature(df))
    
    # print Missing features per person.
    print "- Missing features per person."
    pprint.pprint(myhelpers.missing_per_person(df))

# The person named 'LOCKHART EUGENE E' is going to be removed, since all his features are of value NaN.
df = df[df["name"]!= "LOCKHART EUGENE E"]

# If we draw the salary-bonus scatter plot, we notice one outlier, which has both huge salary and high bonus
print_feature_plot = False
if print_feature_plot == True:
    sns.scatterplot(data=df, x="salary", y="bonus", hue = "poi")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# It turns out that this observation is actually the sum of all the other observations, so we drop this observation also.
df = df[df["name"]!= "TOTAL"]
print df.shape

# After removing "TOTAL", we have still some outliers with both high bonus and salary. 
print_feature_plot = True
if print_feature_plot == True:
    sns.scatterplot(data=df, x="salary", y="bonus", hue = "poi")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# These two are the enron's biggest bosses and poi, so they are definitely person of interest and we should keep them in the dataset.
print df.loc[(df["bonus"]>0.4e7) & (df["salary"] > 1e6), "name"]

# At last, we remove "THE TRAVEL AGENCY IN THE PARK", because it is not an enron employee, thus irrelevant.
df = df[df["name"]!= "THE TRAVEL AGENCY IN THE PARK"]
print df.shape

# We now replace all the NaNs in financial features with zeros
df_imputed = df.fillna(0)
print df_imputed.isnull().sum()

# Turn our data into dictionary for tester.py.
data_cleaned = myhelpers.df_to_dict(df_imputed)

############################################################    
## PART 2: Features Selections
############################################################

# Use original features to set up benchmark performance
feat_default = data_cleaned[(data_cleaned.keys())[0]].keys()
feat_default.remove("poi")
feat_default = ["poi"] + feat_default  ## Move the "poi" feature to the head of the list.

# Benchmark performance
if print_feature_selection:
    print "Settings:\n - Features: default \n"
    print "- Tuning: default"
    
    clf_AdB = AdaBoostClassifier(random_state=0)
    tester.test_classifier(clf_AdB, data_cleaned, feat_default) ## print result
    
    clf_GNB = GaussianNB()
    tester.test_classifier(clf_GNB, data_cleaned, feat_default) ## print result
    
    clf_MLP = MLPClassifier(solver = "lbfgs",random_state=0) ## Sensitive to feature scaling
    tester.test_classifier(clf_MLP, data_cleaned, feat_default) ## print result
    
    clf_LgR = LogisticRegression(solver = 'liblinear',random_state=0, \
                                 max_iter=200, C=10, tol=0.001) ## Sensitive to feature scaling
    tester.test_classifier(clf_LgR, data_cleaned, feat_default, change_scale = True) ## print result
    
    clf_RdF = RandomForestClassifier(n_estimators=60, random_state=0,
                                    min_impurity_decrease=1e-07)
    tester.test_classifier(clf_RdF, data_cleaned, feat_default) ## print result
    
    clf_SVC = SVC(kernel='linear', max_iter = 500,random_state=0, C=10) ## Sensitive to feature scaling
    tester.test_classifier(clf_SVC, data_cleaned, feat_default, change_scale = True) ## print result
    
    print "\n"

# Update feature list by removing features with more than 50% NaN's.
feat_default_1 = list(np.array(feat_default)[np.array(map(lambda x: x not in ["loan_advances", "director_fees", "restricted_stock_deferred", "deferral_payments", "deferred_income", "long_term_incentive"], feat_default))])

# Result from First update of features.
if print_feature_selection:
    print "Settings:\n - Features: removed features with more than 50% of NaN's \n"
    print " - Tuning: default"    
    
    clf_AdB = AdaBoostClassifier(random_state=0)
    tester.test_classifier(clf_AdB, data_cleaned, feat_default_1) ## print result
    
    clf_GNB = GaussianNB()
    tester.test_classifier(clf_GNB, data_cleaned, feat_default_1) ## print result
    
    clf_MLP = MLPClassifier(solver = "lbfgs",random_state=0,
                            hidden_layer_sizes=(100, 80)) ## Sensitive to feature scaling
    tester.test_classifier(clf_MLP, data_cleaned, feat_default_1) ## print result
    
    clf_LgR = LogisticRegression(solver = 'liblinear',random_state=0, \
                                 max_iter=200, C=10, tol=0.001) ## Sensitive to feature scaling
    tester.test_classifier(clf_LgR, data_cleaned, feat_default_1, change_scale = True) ## print result
    
    clf_RdF = RandomForestClassifier(n_estimators = 50, random_state=0)
    tester.test_classifier(clf_RdF, data_cleaned, feat_default_1) ## print result
    
    clf_SVC = SVC(kernel='linear', max_iter = 200,random_state=0, C=10) ## Sensitive to feature scaling
    tester.test_classifier(clf_SVC, data_cleaned, feat_default_1, change_scale = True) ## print result
    print "\n"


### Now we create some new features.

data_cleaned = \
myhelpers.add_feature(data_cleaned, "fraction_to_poi","/", \
                      'from_this_person_to_poi', "from_messages") # fraction of messages sent to poi

data_cleaned = \
myhelpers.add_feature(data_cleaned, "fraction_from_poi","/", \
                      'from_poi_to_this_person', "to_messages") # fraction of messages from poi

data_cleaned = \
myhelpers.add_feature(data_cleaned, "fraction_poi","/", \
                      'from_this_person_to_poi', 'from_poi_to_this_person',"from_messages", "to_messages") # fraction of messages related to poi

data_cleaned = \
myhelpers.add_feature(data_cleaned, "total_gain","+", \
                      'total_payments', "total_stock_value") # total gain in terms of dollar

print len(data_cleaned.keys()[0]) ## Now 24 features in total


# New features added
feat_new = feat_default + ["fraction_to_poi", "fraction_from_poi", "fraction_poi", "total_gain"]


# Look into the performances with feat_new
if print_feature_selection:
    print "Settings:\n - Features: created 4 new features \n"
    print " - Tuning: default"    
    
    clf_AdB = AdaBoostClassifier(random_state=0)
    tester.test_classifier(clf_AdB, data_cleaned, feat_new) ## print result
    
    clf_GNB = GaussianNB()
    tester.test_classifier(clf_GNB, data_cleaned, feat_new) ## print result
    
    clf_MLP = MLPClassifier(solver = "lbfgs",random_state=0) ## Sensitive to feature scaling
    tester.test_classifier(clf_MLP, data_cleaned, feat_new) ## print result
    
    clf_LgR = LogisticRegression(solver = 'liblinear',random_state=0, \
                                 max_iter=200, C=10, tol=0.001) ## Sensitive to feature scaling
    tester.test_classifier(clf_LgR, data_cleaned, feat_new, change_scale = True) ## print result
    
    clf_RdF = RandomForestClassifier(n_estimators = 50, random_state=0)
    tester.test_classifier(clf_RdF, data_cleaned, feat_new) ## print result
    
    clf_SVC = SVC(kernel='linear', max_iter = 500,random_state=0, C=10) ## Sensitive to feature scaling
    tester.test_classifier(clf_SVC, data_cleaned, feat_new, change_scale = True) ## print result
    
    print "\n"    

    
# Get the most important features based on SelectKBest

# KBest for f_classif
if print_feature_selection:
    feat_new_Kbest_f_classif = myhelpers.getKbest(dt_dict=data_cleaned, features_list=feat_new, change_scale = True, K=15, score_function= f_classif)
    feat_new_Kbest_f_classif['rank'] = np.array(range(1,16,1))
    print feat_new_Kbest_f_classif

# KBest for mutual_info_classif
if print_feature_selection:
    feat_new_Kbest_mi_classif = myhelpers.getKbest(dt_dict=data_cleaned, features_list=feat_new, change_scale = True, K=15, score_function= mutual_info_classif)
    feat_new_Kbest_mi_classif['rank'] = np.array(range(1,16,1))
    print feat_new_Kbest_mi_classif
    
# New feature list with 10 most important features from SelectKBest
feat_new_Kbest = list(set(feat_new_Kbest_f_classif.iloc[0:9,0]).union(set(feat_new_Kbest_mi_classif.iloc[0:9,0])))
feat_new_Kbest = ['poi'] + feat_new_Kbest

# Get new performance with feat_K
if print_feature_selection:
    print "Settings:\n - Features: 14 best features from SelectKBest \n"
    print " - Tuning: default"    
    
    clf_AdB = AdaBoostClassifier(random_state=0)
    tester.test_classifier(clf_AdB, data_cleaned, feat_new_Kbest) ## print result
    
    clf_GNB = GaussianNB()
    tester.test_classifier(clf_GNB, data_cleaned, feat_new_Kbest) ## print result
    
    clf_MLP = MLPClassifier(random_state=0) ## Sensitive to feature scaling
    tester.test_classifier(clf_MLP, data_cleaned, feat_new_Kbest) ## print result
    
    clf_LgR = LogisticRegression(solver = 'liblinear', random_state=0) ## Sensitive to feature scaling
    tester.test_classifier(clf_LgR, data_cleaned, feat_new_Kbest) ## print result
    
    clf_RdF = RandomForestClassifier(n_estimators = 50, random_state=0)
    tester.test_classifier(clf_RdF, data_cleaned, feat_new_Kbest) ## print result
    
    clf_SVC = SVC(kernel='linear', max_iter = 500,random_state=0) ## Sensitive to feature scaling
    tester.test_classifier(clf_SVC, data_cleaned, feat_new_Kbest) ## print result
    
    print "\n"



############################################################    
## PART 3: Algorithm tuning
############################################################    

# optimization for each classifier.

if print_algorithm_optimization:
    myhelpers.tune_AdB(dt_dict = data_cleaned, features_list = feat_new_Kbest, scaler = True, mycv = 2)

if print_algorithm_optimization:
    myhelpers.tune_GNB(dt_dict = data_cleaned, features_list = feat_new_Kbest, scaler = True, mycv = 2)

if print_algorithm_optimization:
    myhelpers.tune_MLP(dt_dict = data_cleaned, features_list = feat_new_Kbest, scaler = True, mycv = 2)

if print_algorithm_optimization:
    myhelpers.tune_LgR(dt_dict = data_cleaned, features_list = feat_new_Kbest, scaler = True, mycv = 2)

if print_algorithm_optimization:
    myhelpers.tune_RdF(dt_dict = data_cleaned, features_list = feat_new_Kbest, scaler = True, mycv = 2)

if print_algorithm_optimization:
    myhelpers.tune_SVC(dt_dict = data_cleaned, features_list = feat_new_Kbest, scaler = True, mycv = 2)


# Validate the algorithm
print_model_validation = True

size_of_test_set = 0.3 # 0.4 # 0.2
sss = StratifiedShuffleSplit(n_splits=5, test_size = size_of_test_set, random_state=0)
if print_model_validation:
    myhelpers.tune_AdB(data_cleaned, feat_new_Kbest, True, mycv = sss) ## tester.test_classifier() is used. 
    
## Store data.
my_clf = myhelpers.my_classifier_AdB(data_cleaned, feat_new_Kbest, True, mycv = sss)
my_dataset = data_cleaned
my_features = feat_new_Kbest

dump_classifier_and_data(my_clf, my_dataset, my_features)
