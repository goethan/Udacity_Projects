# Identify Fraud From Enron Financial/Emails Data

This machine learning project is part of the Udacity Data Analyst Nanodegree. Some other projects could be found below:
- [Titanic Data Investigation (EDA)](https://cdn.rawgit.com/brbisheng/ProgrammingFoundations/06081bf9/Final_stage/Titanic_final_Sheng_BI_IPND_1st_modification.html) 
- Prosper Data Investigation (EDA)
  - [Part I](https://cdn.rawgit.com/brbisheng/Udacity_Projects/5ed5e4f7/Sheng_BI_EDA_Prosper_Part1.html)
  - [Part II](https://cdn.rawgit.com/brbisheng/Udacity_Projects/5ed5e4f7/Sheng_BI_EDA_Prosper_Part2.html)
  - [Part III](https://cdn.rawgit.com/brbisheng/Udacity_Projects/5ed5e4f7/Sheng_BI_EDA_Prosper_Part3.html)
- [Data Wrangling](https://cdn.rawgit.com/brbisheng/Udacity_Projects/9d73b4c6/Final_Report.html)

Enron, established in 1985, was one of the largest US energy companies. At the end of 2001, it was discovered that its reported financial condition was actually sustained by creatively planned accounting fraud. The company collapsed into bankruptcy in 2002 due to widespread corporate fraud. During the federal investigation, confidential information was made public, including massive emails and detailed financial data for executives.

We will build a POI (Person of Interest) identifier to spot culpable individuals involved in enron scandal. We will machine learning techniques and the scikit-learn Python library.

Resources¶
We will have a few resources to ease the process:

final_project_dataset.pkl is the dataset we're going to investigate
Starter code and helper functions are provided: it reads in the data, takes our features of choice, then puts them into a numpy array (the format sklearn functions assume). We just have to engineer the features, pick and tune an algorithm, and to test and evaluate our identifier. For example:
featureFormat() to convert the dictionary into a numpy array of features: this is the only way we can make it work with Scickit Learn.
targetFeatureSplit(), to separate the feature (returned by featureFormat()) we want to predict from the others.
Data is preprocessed: Enron email and financial data are combined into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)


1. Summarize for us the goal of this project and how machine learning is useful in trying to 
accomplish it. As part of your answer, give some background on the dataset and how it 
can be used to answer the project question. Were there any outliers in the data when 
you got it, and how did you handle those? 

The main goal of this project is to use both financial and email data from Enron to build a predictive model that could potentially identify a "person of interest" (POI), i.e. Enron employees who may have committed fraud, based on the aforementioned public data.

2. What features did you end up using in your POI identifier, and what selection process did 
you use to pick them? Did you have to do any scaling? Why or why not? As part of the 
assignment, you should attempt to engineer your own feature that does not come 
ready­made in the dataset ­­ explain what feature you tried to make, and the rationale 
behind it. (You do not necessarily have to use it in the final analysis, only engineer and 
test it.) In your feature selection step, if you used an algorithm like a decision tree, 
please also give the feature importances of the features that you use, and if you used an 
automated feature selection function like SelectKBest, please report the feature scores 
and reasons for your choice of parameter values.  [relevant rubric items: “create new 
features”, “properly scale features”, “intelligently select feature”] 
 
3. What algorithm did you end up using? What other one(s) did you try? How did model 
performance differ between algorithms?  [relevant rubric item: “pick an algorithm”] 
 4. What does it mean to tune the parameters of an algorithm, and what can happen if you 
don’t do this well?  How did you tune the parameters of your particular algorithm? (Some 
algorithms do not have parameters that you need to tune ­­ if this is the case for the one 
you picked, identify and briefly explain how you would have done it for the model that 
was not your final choice or a different model that does utilize parameter tuning, e.g. a 
decision tree classifier).  [relevant rubric item: “tune the algorithm”] 
 
5. What is validation, and what’s a classic mistake you can make if you do it wrong? How 
did you validate your analysis?  [relevant rubric item: “validation strategy”] 
 
6. Give at least 2 evaluation metrics and your average performance for each of them. 
Explain an interpretation of your metrics that says something human­understandable 
about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”] 



plan:
1. I will finish understand the targetted dataset for each miniproject.
2. I do the py file in cloud.
3. I finish this summary.
