#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
sys.path.append("../tools/")
import tester

from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def dict_to_df(dt_dict):
    """
    1. Turn dictionary to dataframe
    2. convert NaN strings to dataframe NaN's
    """
    
    ## Turn dictionary to DataFrame
    df = pd.DataFrame.from_dict(data=dt_dict, orient = "index")
    df = df.replace('NaN',np.nan) ## Replace string 'NaN' by np.nan

    ## Reset dataframe row index
    df = df.reset_index()
    df['name']=df['index']
    df['index'] = df.index

    ## Create a deep copy for replacing operation.
    df1 = df.copy(deep = True)
    
    return df1

def df_to_dict(df):
    """
    convert dataframe to dictionary;
    drop unnecessary columns.
    """
    df1 = df.drop(['index', 'email_address'], axis=1)
    df1.index = list(df['name'])
    df2 = df1.drop(["name"], axis = 1)
    result = df2.to_dict('index')
    return result

def missing_per_feature(df, feature = "poi"):
    
    """
    Print missing values per feature and per person
    """
    
    poi_true = df[feature]==True
    poi_false = df[feature]==False 
    
    result = \
    pd.DataFrame({"perc_missing_total": df.isnull().sum()/float(df.shape[0]),\
              "perc_missing_{}_True".format(feature): df[poi_true].isnull().sum()/float(df[poi_true].shape[0]),\
              "perc_missing_{}_False".format(feature): df[poi_false].isnull().sum()/float(df[poi_false].shape[0])}).\
    sort_values(['perc_missing_total'], ascending = False)
    
    return result

def missing_per_person(df):
    
    result = pd.concat([pd.Series(map(lambda x: df.loc[x,:].isnull().sum(), df['index'])),df['name']], axis = 1)
    result.columns.values[0] = 'n_nulls'
    result['perc_features_missing'] = result["n_nulls"]/float((df.shape[1]-2))
    
    return result.sort_values(["n_nulls"], ascending = False)


def add_feature(d, newf, op, *args):
    """
    Create new features.
    
    Arguments of the function:
    d: data dictionary.
    args: input features
    newf: the resulting new feature
    op: operation, either "+" or "/"
 
    Output: a data dictionary with the new feature added.
    """
    li = list(args)
    if op == "+":
        for person in d:
            d[person][newf] = d[person][li[0]] + d[person][li[1]]
            # print d[person][newf]
    if op == "/":
        if len(li) == 2:
            t1 = li[0]
            t2 = li[1]
            for person in d:
                if d[person][t2] == 0:
                    d[person][newf] = 0.0
                else:
                    d[person][newf] = d[person][t1] / float(d[person][t2])
                # print d[person][newf]
        if len(li) == 4:
            t1 = li[0]
            t2 = li[1]
            t3 = li[2]
            t4 = li[3]
            for person in d:
                if d[person][t3] + d[person][t4] == 0:
                    d[person][newf] = 0.0
                else:
                    d[person][newf] = (d[person][t1] + d[person][t2]) / float(d[person][t3]+d[person][t4])
                # print d[person][newf]
    return d

def getKbest(dt_dict, features_list, score_function, change_scale = False, K = 10):
    """
    Return the k most important features by SelectKBest
    """
    # Get values of all features.
    data = featureFormat(dt_dict, features_list, sort_keys=True)
    
    # Split labels (poi) and other features
    labels, features = targetFeatureSplit(data)
    
    # feature list, without the dependent variable poi
    feat = features_list[1:]
    
    if change_scale == True:
        # Set up the scaler
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)        
    
    if score_function == f_classif:

        selector = SelectKBest(f_classif, k=K)
        
    elif score_function == mutual_info_classif:
        
        def my_score(X, y):
            """
            The result of mutual_info_classif depends upon the random_state.
            """
            return mutual_info_classif(X, y, random_state=0)
        
        selector = SelectKBest(my_score, k=K)
    
    fitted = selector.fit(features, labels)
    
    scores = fitted.scores_
    # print scores
    supports = fitted.get_support(indices = False)
    # print supports
    # print feat
    new_features = []
    
    new_scores = []
    
    for support_i, feature_i, score_i in zip(supports, feat, scores):
        if support_i:
            new_features.append(feature_i)
            new_scores.append(score_i)

    result = pd.DataFrame({"feature": new_features, "score": new_scores}) # , "ranking": range(1,16,1)

    return result.sort_values(by=["score"], ascending = False)


###############################
## Part 3 Algorithm tuning
###############################

## Tune AdaBoost

def tune_AdB(dt_dict, features_list, scaler, mycv):
    """
    Fit and Print optimal parameter grid by Pipleine and GridSearchCV.
    """
    # Keep only features from features_list
    data = featureFormat(dt_dict, features_list, sort_keys=True)
    
    labels, features = targetFeatureSplit(data)

    if scaler:
        adb = Pipeline([('scaler', MinMaxScaler()),
                       ('adb', AdaBoostClassifier())])
    else:
        adb = Pipeline([('adb', AdaBoostClassifier())])

    param_grid = {'adb__n_estimators': range(10,80,10),
                 'adb__random_state':[0],
                 'adb__learning_rate': [0.001, 0.05, 0.1, 0.3, 0.5,0.7,0.9, 1]}

    clf_adb = GridSearchCV(adb, param_grid, scoring = 'f1', cv = mycv, iid=True).fit(features, labels).best_estimator_

    tester.test_classifier(clf_adb, dt_dict, features_list)

    return

## Tune GaussianNB

def tune_GNB(dt_dict, features_list, scaler, mycv):
    """
    Fit and Print optimal parameter grid by Pipleine and GridSearchCV.
    """
    # Keep only features from features_list
    data = featureFormat(dt_dict, features_list, sort_keys=True)
    
    labels, features = targetFeatureSplit(data)

    if scaler:
        gnb = Pipeline([('scaler', MinMaxScaler()),
                       ('gnb', GaussianNB())])
    else:
        gnb = Pipeline([('gnb', GaussianNB())])

    param_grid = {"gnb__var_smoothing": [1e-6,1e-3, 0.1, 1,1.5, 2]}

    clf_gnb = GridSearchCV(gnb,param_grid, scoring = 'f1', cv = mycv, iid=True).fit(features, labels).best_estimator_

    tester.test_classifier(clf_gnb, dt_dict, features_list)

    return

## Tune Neural network

def tune_MLP(dt_dict, features_list, scaler, mycv):
    """
    Fit and Print optimal parameter grid by Pipleine and GridSearchCV.
    """
    # Keep only features from features_list
    data = featureFormat(dt_dict, features_list, sort_keys=True)
    
    labels, features = targetFeatureSplit(data)

    if scaler:
        mlp = Pipeline([('scaler', MinMaxScaler()),
                       ('mlp', MLPClassifier())])
    else:
        mlp = Pipeline([('mlp', MLPClassifier())])

    param_grid = {'mlp__random_state':[0],
                 'mlp__activation':["relu", "logistic", "tanh"],
                 'mlp__solver':["lbfgs"],
                 'mlp__alpha':np.linspace(0.00001,0.1,20),
                 'mlp__hidden_layer_sizes':[(20,30,40), (15,20,30)]}

    clf_mlp = GridSearchCV(mlp, param_grid, scoring = 'f1', cv = mycv, iid=True).fit(features, labels).best_estimator_

    tester.test_classifier(clf_mlp, dt_dict, features_list)

    return

## Logistic regression

def tune_LgR(dt_dict, features_list, scaler, mycv):
    """
    Fit and Print optimal parameter grid by Pipleine and GridSearchCV.
    """
    # Keep only features from features_list
    data = featureFormat(dt_dict, features_list, sort_keys=True)
    
    labels, features = targetFeatureSplit(data)

    if scaler:
        lgr = Pipeline([('scaler', MinMaxScaler()),
                       ('lgr', LogisticRegression())])
    else:
        lgr = Pipeline([('lgr', LogisticRegression())])

    param_grid = {"lgr__fit_intercept":[True, False],
                 "lgr__C": [10,2,1.5,1,0.8,0.5,0.2,0.1,1e-1,1e-2,1e-3,1e-4],
                 "lgr__random_state":[0]}

    clf_lgr = GridSearchCV(lgr,param_grid, scoring = 'f1', cv = mycv, iid=True).fit(features, labels).best_estimator_

    tester.test_classifier(clf_lgr, dt_dict, features_list)

    return


## Tune Random forest.
def tune_RdF(dt_dict, features_list, scaler, mycv):
    """
    Fit and Print optimal parameter grid by Pipleine and GridSearchCV.
    """
    # Keep only features from features_list
    data = featureFormat(dt_dict, features_list, sort_keys=True)
    
    labels, features = targetFeatureSplit(data)

    if scaler:
        rf = Pipeline([('scaler', MinMaxScaler()),
                       ('rf', RandomForestClassifier())])
    else:
        rf = Pipeline([('rf', RandomForestClassifier())])

    param_grid = {'rf__n_estimators': range(10,110,10),
                  'rf__min_samples_split' :[2,3,4,5],
                  'rf__min_samples_leaf' : [1,2,3],
                 'rf__random_state':[0],
                 'rf__max_features':[None]}

    clf_rf = GridSearchCV(rf,param_grid, scoring = 'f1', cv = mycv, iid=True).fit(features, labels).best_estimator_

    tester.test_classifier(clf_rf, dt_dict, features_list)

    return

## Tune Support Vector Machines
def tune_SVC(dt_dict, features_list, scaler,mycv):
    """
    Fit and Print optimal parameter grid by Pipleine and GridSearchCV.
    """
    # Keep only features from features_list
    data = featureFormat(dt_dict, features_list, sort_keys=True)
    
    labels, features = targetFeatureSplit(data)

    if scaler:
        svc = Pipeline([('scaler', MinMaxScaler()),
                       ('svc', SVC())])
    else:
        svc = Pipeline([('svc', SVC())])

    param_grid = {'svc__C': [1, 1e2, 1e3, 1e4],
                  'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1 ],
                  'svc__degree': [1, 2, 3],
                  'svc__kernel':["linear","poly","rbf","sigmoid"],
                 "svc__random_state":[0]}

    clf_svc = GridSearchCV(svc,param_grid, scoring = 'f1', cv = mycv, iid=True).fit(features, labels).best_estimator_

    tester.test_classifier(clf_svc, dt_dict, features_list)

    return

## my classifier result 

## Tune AdaBoost

def my_classifier_AdB(dt_dict, features_list, scaler, mycv):
    """
    Fit and Print optimal parameter grid by Pipleine and GridSearchCV.
    """
    # Keep only features from features_list
    data = featureFormat(dt_dict, features_list, sort_keys=True)
    
    labels, features = targetFeatureSplit(data)

    if scaler:
        adb = Pipeline([('scaler', MinMaxScaler()),
                       ('adb', AdaBoostClassifier())])
    else:
        adb = Pipeline([('adb', AdaBoostClassifier())])

    param_grid = {'adb__n_estimators': range(10,80,10),
                 'adb__random_state':[0],
                 'adb__learning_rate': [0.001, 0.05, 0.1, 0.3, 0.5,0.7,0.9, 1]}

    clf_adb = GridSearchCV(adb, param_grid, scoring = 'f1', cv = mycv, iid=True).fit(features, labels).best_estimator_

    return clf_adb



## log function
def log0(x):
    if x == 0:
        return 0
    else:
        return math.log(x)

vlog0 = np.vectorize(log0)

vlog0([0,0])

#map(log0, array([0,0]))


