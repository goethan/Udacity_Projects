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

It looks like there still are four additional points with higher salary and bonus, in a range that could potentially consider them as outliers.

Person Name	Salary	isPOI
SKILLING JEFFREY K	1111258	True
LAY KENNETH L	1072321	True
FREVERT MARK A	1060932	False
PICKERING MARK R	655037	False
After closer inspection, and despite not all of them were POIs, the rest of their data seemed consistent across the board and all of them looked like valid and meaningful data points.

Incomplete Data

Another potential source of outliers are the ones that don't add meaningful information to the mix, such as persons with little or no relevant information at all.

In order to spot these data points, the get_incompletes() function returns a list of the names with no feature data above a certain threshold.

With get_incompletes() set at 90%, which means that the persons returned by the function have only less than 10% of the data completed, it returns this list.

['WHALEY DAVID A',
 'WROBEL BRUCE',
 'LOCKHART EUGENE E',
 'THE TRAVEL AGENCY IN THE PARK',
 'GRAMM WENDY L']
After inspecting closely each person one by one, there's no meaningful information we can derive from these persons and on top of that, none of each is a POI, therefore, they will be removed from the data set.


> 2. What features did you end up using in your POI identifier, and what selection process did 
you use to pick them? Did you have to do any scaling? Why or why not? As part of the 
assignment, you should attempt to engineer your own feature that does not come 
ready­made in the dataset ­­ explain what feature you tried to make, and the rationale 
behind it. 

(You do not necessarily have to use it in the final analysis, only engineer and 
test it.) In your feature selection step, if you used an algorithm like a decision tree, 
please also give the feature importances of the features that you use, and if you used an 
automated feature selection function like SelectKBest, please report the feature scores 
and reasons for your choice of parameter values.  [relevant rubric items: “create new 
features”, “properly scale features”, “intelligently select feature”] 
 
> 3. What algorithm did you end up using? What other one(s) did you try? How did model 
performance differ between algorithms? 


 
> 4. What does it mean to tune the parameters of an algorithm, and what can happen if you 
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
