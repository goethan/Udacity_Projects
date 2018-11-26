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

## Contents of The Project

The structure of the code is as follows:

- `poi_id.py`: main script
- `myhelpers.py`: contains all the helper funtions used in `poi_id.py`.
- `tester.py`: contains code for testing the algorithm performance

In this README.md document, you could find the answers for all the project questions. This README.md consists of 3 parts:

- Data Exploration: we describe data, identify and remove outliers
- Features Section: we create and select features using appropriate statistical methods.
- Algorithm Tuning: we tune parameters for each classifer and validate our model.

## Data Exploration

> 1. Summarize for us the goal of this project and how machine learning is useful in trying to 
accomplish it. As part of your answer, give some background on the dataset and how it 
can be used to answer the project question. Were there any outliers in the data when 
you got it, and how did you handle those? 

The goal of this project is to develop a predictive model that helps identify a POI (person of interest). A POI is a former Enron employee who may have been related to fraud, based on the enron data which comprise both financial and emails information of the employees.

The features extracted from enron data belong to one of the 3 categories:

- financial features (independent variables)
  - `financial_features_list = ['salary', 'deferral_payments', 'total_payments',\
                           'loan_advances', 'bonus', 'restricted_stock_deferred', \
                           'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', \
                           'other', 'long_term_incentive', 'restricted_stock', 'director_fees']`
- email features (independent variables)
  - `email_features_list = ['to_messages', 'email_address', 'from_poi_to_this_person', \
                         'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']`

- `poi` (dependent variable/label)

As a quick summary, the dataset 
- contains a total of 146 data points, of which 18 are POIs and 128 are not .
- There are 14 financial features, 6 email features.

### Looking into NaN data.

We notice significant amount of 'NaN's in data. Below is a table showing the fraction of missing values per feature.

|             | frac_missing_for_non-poi | frac_missing_for_poi	| frac_missing_total |
|-------------|--------------------------|----------------------|--------------------|
|loan_advances|	0.976562	               |0.944444	            |0.972603            |
|director_fees|	0.867188	               |1.000000	            |0.883562            |
|restricted_stock_deferred|	0.859375	   |1.000000	            |0.876712            |
|deferral_payments| 0.734375	| 0.722222	| 0.732877|
|deferred_income|	0.703125	| 0.388889	| 0.664384|
|long_term_incentive|	0.578125	|0.333333	|0.547945|
|bonus|	0.484375	| 0.111111	| 0.438356|
|from_messages|	0.437500	| 0.222222	| 0.410959|
|to_messages|	0.437500	| 0.222222	| 0.410959|
|from_poi_to_this_person|	0.437500|	0.222222|	0.410959|
|shared_receipt_with_poi|	0.437500|	0.222222|	0.410959|
|from_this_person_to_poi|	0.437500|	0.222222|	0.410959|
|other| 0.414062	| 0.000000	| 0.363014|
|expenses|	0.398438	|0.000000	| 0.349315|
|salary|	0.390625	| 0.055556	|0.349315|
|exercised_stock_options|	0.296875	|0.333333	|0.301370|
|restricted_stock|	0.273438	|0.055556|	0.246575|
|email_address|	0.273438	|0.000000|	0.239726|
|total_payments|	0.164062	|0.000000|	0.143836|
|total_stock_value|	0.156250|	0.000000|	0.136986|
|poi|	0.000000|	0.000000|	0.000000|

We can not discard any data at this point, because the 'NaN's may not necessarily represent missing data. Indeed, they may well represent that the variable in question is of value zero. For example, if there are no director fees, the value of this variable should be zero, thus replacing the 'NaN's by zero may help us make influential predictions. This conjecture is supported by the `variable-descriptioin.pdf` document here, which gives information on how each variable is formed. We thus will proceed with the NaN's being replaced by zero.

### Outliers

To spot potential outliers, we start by plotting the scatterplot of the two most interesting variabels: `salary` and `bonus`.
![alt text](https://github.com/brbisheng/Udacity_Projects/blob/master/tentative/supporting_materials/salary-bous-scatterplot.png)

There is clearly an outlier with both enormous salary and bonus. It turns out that this outlier is named `TOTAL`, which is the sum of all other obervations. Thus we will remove this observation, and we obtain the following graph:

![salary-bonus-without-Total](https://github.com/brbisheng/Udacity_Projects/blob/master/tentative/supporting_materials/salary-bouns-scatterplot-without-TOTAL.png)

Graphically, there seem to be stil 4 outliers, either with unusually high salary or unusually high bonus. It turns out the the two observations with both high salary and bonus are SKILLING JEFFREY and LAY KENNETH, the two biggest bosses and poi's of Enron. We definitely shall keep these two observations. In addition, we have FREVERT MARK with exceptionally high salary and LAVORATO JOHN with exceptionally high bonus. After examination, we find that the values of the other features of these two persons seem to be consistent with similar observations, thus we will also keep them.

### Irrelevant data

Besides, we are going to remove two observations. The first is named 'LOCKHART EUGENE E', because every feature of this person is NaN, thus it provides no information to help predict. The second is named ''THE TRAVEL AGENCY IN THE PARK'', because this is obviously not an enron employee, thus is irrelevant to our problem. 

In the end, we will have 143 observations to proceed.

> 2. What features did you end up using in your POI identifier, and what selection process did 
you use to pick them? Did you have to do any scaling? Why or why not? As part of the 
assignment, you should attempt to engineer your own feature that does not come 
ready­made in the dataset ­­ explain what feature you tried to make, and the rationale 
behind it. (You do not necessarily have to use it in the final analysis, only engineer and 
test it.) In your feature selection step, if you used an algorithm like a decision tree, 
please also give the feature importances of the features that you use, and if you used an 
automated feature selection function like SelectKBest, please report the feature scores 
and reasons for your choice of parameter values.

Given that this is a classification problem, I used the `SelectKBest` from sklearn package to filter features based on their scores. I considered two score functions `f_classif` and `mutual_info_classif`, for each of which I extracted the 15 features with highest scores. I then select 10 features which have high scores both for  

to select best 10 influential features and used those featuers for all the upcoming algorithm. Unsurprisingly, 9 out of 10 features related to financial data and only 1 features called shared_receipt_with_poi (messages from/to the POI divided by to/from messages from the person) were attempted to engineere by us. Main purpose of composing ratio of POI message is we expect POI contact each other more often than non-POI and the relationship could be non-linear. The initial assumption behind these features is: the relationship between POI is much more stronger than between POI and non-POIs, and if we quickly did back-of-the-envelope Excel scatter plot, there might be truth to that hypothesis. The fact that shared_receipt_with_poi is included after using SelectKBest proved that this is a crucial features, as they also slightly increased the precision and recall of most of the machine learning algorithms used in later part of the analysis (e.g precision & recall for Support Vector Classifer before adding new feature are 0.503 & 0.223 respectively, while after adding new feature, the results are 0.504 & 0.225)

After feature engineering & using SelectKBest, I then scaled all features using min-max scalers. As briefly investigated through exporting CSV, we can see all email and financial data are varied by several order of magnitudes. Therefore, it is vital that we feature-scaling for the features to be considered evenly. For a comprehensive look on the chosen features, we can look at their respective score after using SelectKBest by the table below:

> 3. What algorithm did you end up using? What other one(s) did you try? How did model 
performance differ between algorithms? 


 
> 4. What does it mean to tune the parameters of an algorithm, and what can happen if you 
don’t do this well?  How did you tune the parameters of your particular algorithm? (Some 
algorithms do not have parameters that you need to tune, if this is the case for the one 
you picked, identify and briefly explain how you would have done it for the model that 
was not your final choice or a different model that does utilize parameter tuning, e.g. a 
decision tree classifier). 

> 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How 
did you validate your analysis? 

Validation comprises set of techniques to make sure our model generalizes with the remaining part of the dataset. A classic mistakes, which was briefly mistaken by me, is over-fitting where the model performed well on training set but have substantial lower result on test set. In order to overcome such classic mistake, we can conduct cross-validation (provided by the evaluate function in poi_id.py where I start 1000 trials and divided the dataset into 3:1 training-to-test ratio. Main reason why we would use StratifiedSuffleSplit rather than other splitting techniques avaible is due to the nature of our dataset, which is extremely small with only 14 Persons of Interest. A single split into a training & test set would not give a better estimate of error accuracy. Therefore, we need to randomly split the data into multiple trials while keeping the fraction of POIs in each trials relatively constant.
 
> 6. Give at least 2 evaluation metrics and your average performance for each of them. 
Explain an interpretation of your metrics that says something human-understandable 
about your algorithm’s performance. 

For this assignment, I used precision & recall as 2 main evaluation metrics. The best performance belongs to logistic regression (precision: 0.382 & recall: 0.415) which is also the final model of choice, as logistic regression is also widely used in text classification, we can actually extend this model for email classification if needed. 

Precision refers to the ratio of true positive (predicted as POI) to the records that are actually POI while recall described ratio of true positives to people flagged as POI. Essentially speaking, with a precision score of 0.386, it tells us if this model predicts 100 POIs, there would be 38 people are actually POIs and the rest 62 are innocent. With recall score of 0.4252, this model finds 42% of all real POIs in prediction. This model is amazingly perfect for finding bad guys without missing anyone, but with 42% probability fo wrong

With a precision score of 0.38, it tells us that if this model predicts 100 POIs, then the chance would be 38 people who are truely POIs and the rest 62 are innocent. On the other hand, with a recall score of 0.415, this model can find 42% of all real POIs in prediction. Due to the nature of the dataset, accuracy is not a good measurement as even if non-POI are all flagged, the accuracy score will yield that the model is a success.

