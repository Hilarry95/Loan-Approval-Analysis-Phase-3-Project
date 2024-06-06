# Loan Approval Analysis for Bank of India
## Overview
The Bank of India provides Loan Services to many customers and are looking for a way to automate the process and find an effective and industrious way to carry out the exercise. This project implements machine learning algorithms to analyzes data from previous clients to identify patterns that can be used to flag an applicant as a risky applicant or 
![alt text](Visualizations/images.jpg)
## Business Understanding
The loan approval project aims to significantly enhance the efficiency and accuracy of the loan approval process through advanced machine learning techniques. Faced with challenges such as data quality, the proposed solution focuses on achieving a balance between performance and interpretability. With an accuracy target of 80%, the model is expected to provide reliable predictions that can streamline bank operations, reduce approval times, and maintain fair lending practices. 
### Objectives
* To
* To
* To
## Data Understanding
The Loan Approval Dataset was sourced from Kaggle.The dataset has comprehensive information regarding a loan applicant's financial status, personal attributes such as age and marital status.The dataset has a blend of both numerical and categorical features.
The rows indicate a loan applicant whereas the different columns display information regarding the said applicant having the last column 'Risk_Flag as a binary input to show if the applicant is risky or not
## Data Cleaning
Overall the dataset was pretty clean, did not have any missing, null or duplicate values
Further cleaning was done to remove outliers from the dataset which would have impacted the results of the models
Moreover data was split to train and test sets, categorical columns converted to numerical and scaling done to have them all in one scale
## EDA
The dataset was analyzed using various statistical and visualization techniques to understand the distribution of the data and the relationships
## Modeling
Six models were created from the dataset, the baseline model was a simple logistic regression from which we were able to find the best test size to implement
![alt text](Visualizations/Logreg.png)
The second model was an iteration of the LogisticRegression model with tuned parameters
![alt text](Visualizations/Logreg2.png)
The third model was a Decision Tree Classifier which had an accuracy score of 87.1% but proved to be overfiting to the train data
![alt text](Visualizations/DecisionTree.png)
The fourth model was an iteration of the DecisionTree Classifier with tuned parameters and balanced class weights and had an accuracy score of 54% but did not overfit to the training data
![alt text](Visualizations/DecisionTree2.png)
The fifth model was a Random Forest Classifier which had an accuracy score of 67.4% and did not overfit to the training data
![alt text](Visualizations/RandomForest.png)
The last and best performed model was an XGBoost model that implemented gradient boosting algorithm to learn from weaker learner models and produce a strong model with an accuravcy score of 89.3% and did not show signs to overfit to the training data
![alt text](Visualizations/XGBoost.png)
## Recommendation
 * Conduct fairness analysis to prevent discrimination against any group and nsure the model complies with relevant regulations
 * Provide training for stakeholders and implement a feedback mechanism to improve the model continuously
 * Continuously monitor model performance post-deployment and Plan for regular updates and retraining with new data to maintain    accuracy and relevance.
 * Explore the use of advanced machine learning techniques like deep learning to further improve model accuracy and    continuously explore new features that can improve the predictive power of the model.



