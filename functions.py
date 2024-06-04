import pandas as pd


class Data_cleaning:
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
    
    #train test split
    def splitting(self,X, y,size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    # Fit SMOTE to training data
    def fix_imbalance(self, X_train, y_train,smote):    
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled
    
    #encoding the categorical columns
    def encode_categorical_columns(df):
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        numerical_columns = df.select_dtypes(include=['number']).columns
    
        # Encode categorical columns using pd.get_dummies
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
        return categorical_columns, numerical_columns, df_encoded
    
class Analysis:
    def __init__(self):
        pass

    def correlation_heatmap(df):
        plt.figure(figsize=(10, 8))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.show()

    def Target(df):    
        # Distribution of Target Variable
        sns.countplot(x='Risk_Flag', data=df)
        plt.title('Distribution of Risk_Flag')
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
        print(f"{classifier} has an accuracy of {train_acc*100:.2f}% on the train test")
        print("\n"f"{classifier} has an accuracy of {test_acc*100:.2f}% on the test test")

        #check for overfitting
        if train_acc > test_acc:
            print("Model is overfitting")
        else:
            print("Model is not overfitting")

        #confusion matrix plot
        ConfusionMatrixDisplay.from_estimator(estimator=classifier,X=X_train,y=y_train)  
        plt.show()

                      
        
    
    
        
