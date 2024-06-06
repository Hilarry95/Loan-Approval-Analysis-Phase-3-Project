import pandas as pd
import numpy as np

from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns



class Loading:
    def __init__(self) :
        pass
    
    def load_data(self,data_path):
        #loading data
        df = pd.read_json(data_path)

        #print data shape
        data_shape = df.shape
        print(f"The dataset contains {data_shape[0]} loan applicants with {data_shape[1]} attributes")
    
        #Print Data information
        print(df.info())

        #Target Column Value Counts
        target_value_counts = df['Risk_Flag'].value_counts()
        print("\nRisk_Flag value counts:")
        print("0 (No Risk): ", target_value_counts.get(0, 0))
        print("1 (Risk): ", target_value_counts.get(1, 0))

        return df
    
class Cleaning:
    def __init__(self):
        pass
        
    def identify_issues(self,dataset):
        #initiate an empty dictionary
        issues = {}
    
        # Identify missing values
        missing_values = dataset.isnull().sum()
    
        # Identify duplicate rows
        duplicate_rows = dataset.duplicated().sum()
    
        # Identify null values
        null_values = dataset.isna().sum()
    
        #adding them to the issues dictionary
        issues['missing_values'] = missing_values
        issues['duplicate_rows'] = duplicate_rows
        issues['null_values'] = null_values
        
        return issues
        
        
    
class Preprocessing: 
    def __init__(self) :
        pass
    
    def prep(self,X, y,size):
        #Convert categorical features to numeric
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
        
        #Scaling the X_train and X_test
        scaler = StandardScaler()
        X_train_scal = scaler.fit_transform(X_train)
        X_test_scal = scaler.transform(X_test)

        return X_train_scal, X_test_scal, y_train, y_test

class Analysis:
    def __init__(self):
        pass

    def targer_analysis(self, df):
        # Distribution of the Target Variable
        sns.countplot(x='Risk_Flag', data=df)
        plt.title('Distribution of the Risk Flag')
        plt.show()

    def categorical_analysis(self,column_list, dataset):
        #Analysis for categorical columns
        for column in column_list:
            sns.countplot(x=column, data=dataset)
            plt.title(f'Distribution of {column}')
            plt.show()

    def category_by_target(self, column_list, dataset):
        #Analysis of Categorical columns by the risk flag
        for column in column_list:
           sns.countplot(x=column, hue='Risk_Flag', data=dataset)
           plt.title(f'{column} by Risk_Flag')
           plt.show()   

    def correlation_map(self, dataset):
        correlation = dataset.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap ')
        plt.show()         


class Modeling:
    def __init__(self):
        pass

    def models(self,classifier,X_train,y_train,X_test,y_test):
            # Fit the model
            classifier.fit(X_train,y_train)

            #Make predictions
            y_hat_train = classifier.predict(X_train)
            y_hat_test = classifier.predict(X_test)  

            #Print the accuracy scores for the model test
            train_acc = accuracy_score(y_train,y_hat_train)
            test_acc = accuracy_score(y_test,y_hat_test)
            print("\n"f"The model has an accuracy of {test_acc*100:.2f}% on the test test")
            print(f"The Model has an accuracy of {train_acc*100:.2f}% on the train test")
            
            return classifier.score(X_test, y_test)
        
class Evaluation:
    def __init__(self):
        pass
    
    def Evaluate(self,classifier, X_test, y_test):
        # Make predictions
        y_pred = classifier.predict(X_test)
        # Take the probability of the positive class
        y_pred_probabilities = classifier.predict_proba(X_test)[:, 1]  
    
        # Calculate evaluation metrics
        precision = precision_score(y_test, y_pred,zero_division=1)
        recall = recall_score(y_test, y_pred,zero_division=1)
        f1 = f1_score(y_test, y_pred,zero_division=1)
        roc_auc = roc_auc_score(y_test, y_pred_probabilities)
    
         # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()
    
        # Create dictionary to store evaluation results
        evaluation_results = {
                             'Precision': precision,
                             'Recall': recall,
                             'F1-score': f1,
                             'ROC-AUC': roc_auc
                             }
    
        return evaluation_results
             