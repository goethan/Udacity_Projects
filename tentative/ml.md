# Identify Fraud From Enron Financial/Emails Data

This machine learning project is part of the the Udacity Data Analyst Nanodegree. Some other projects could be found below:
- [Titanic Data EDA](https://cdn.rawgit.com/brbisheng/ProgrammingFoundations/06081bf9/Final_stage/Titanic_final_Sheng_BI_IPND_1st_modification.html) 
- Prosper Data EDA



In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

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
