'''

  Naive Bayes: 



 phases of crisp ML-Q (cross industry standard process for machine learning with quality assurance):
     
 1] data understanding and business understanding 
 2] data preparation(data cleaning)
 3] model building (data mining)    
 4] model evaluation
 5] model deployment
 6] Monitoring and Maintainance      
   

Problem Statements:

1. Prepare a classification model using the Naive Bayes algorithm for the salary dataset. 
   Train and test datasets are given separately. Use both for model building. And predict Salary

     
1] step : data understanding and business understanding: 

business objectives  : Achieve Salary Equity: Ensure salary equity within the organization by identifying and addressing disparities based on various factors, including age, education, occupation, and other features.
business constraints :Cost Savings: Identify areas where cost savings can be achieved without compromising the competitiveness of compensation packages.
    

Success Criteria:-
    
Business success criteria        : Salary Equity: Success criteria include achieving salary equity within the organization 
Machine Learning success criteria:  high accuracy of predictive naive bayes models is often a primary success criteria.
Economic success criteria        :  Economic success criteria may include increased revenue due to improved employee performance and talent retention.



Data Collection: 
dataset is already present in our LMS.



Data description:

age: Age of the individual.
workclass: The type of employment, such as "State-gov," "Self-emp-not-inc," or "Private."
education: The level of education, e.g., "Bachelors," "HS-grad."
educationno: Numeric representation of education level.
maritalstatus: Marital status, e.g., "Never-married," "Married-civ-spouse."
occupation: The occupation of the individual, such as "Adm-clerical" or "Exec-managerial."
relationship: The person's relationship status, e.g., "Not-in-family," "Husband."
race: The race or ethnicity of the individual.
sex: The gender of the individual, either "Male" or "Female."
capitalgain: Capital gains for the individual.
capitalloss: Capital losses for the individual.
hoursperweek: The number of hours worked per week.
native: The native country or place of origin.
Salary: The target variable, indicating whether the individual's salary is less than or equal to 50K (<=50K) or greater than 50K (>50K).    

'''






'''
2] step : data preprocessing (data cleaning) :
'''       


import pandas as pd

data = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\ASSIGNMENTS\SUPERVISED LEARNING\CLASSIFICATION\naive bayes\SalaryData_Train.csv")

# MySQL Database connection
# Creating engine which connect to MySQL
user = 'user1' # user name
pw = 'user1' # password
db = 'salary_db' # database

from sqlalchemy import create_engine
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
data.to_sql('salary_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from salary_tbl'
df = pd.read_sql_query(sql, con = engine) 
 
df.shape
df.dtypes
df.info()
df.describe() 
df.duplicated().sum() # some duplicated rows are present here
df1 = df.drop_duplicates() # drop that duplicated rows
df1.isnull().sum() # not any null values.



# outlier treatment :
import seaborn as sns
sns.boxplot(df1)

df1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# here we see some outliers are present here, 

from AutoClean import AutoClean 
treatment = AutoClean(df1, mode = 'manual', outliers = 'winz') 

df1 = treatment.output 
# to check any outliers are present or not here
df1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# here we succesfully replace all outliers with inliers. 





# Split data into features (X) and target variable (y)
X = df1.drop("Salary", axis = 1) 
y = df1["Salary"]






# in independent features we have both categorical as well numerical variable 

numeric_features = X.select_dtypes(exclude = ['object']).columns 
numeric_features


categorical_features = X.select_dtypes(include=['object']).columns
categorical_features




#-------------------------------------------------------------------------------------------------------------


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Imputation to handle missing values
# MinMaxScaler to convert the magnitude of the columns to a range of 0 to 1num_pipeline1 = Pipeline(steps=[('impute1', SimpleImputer(strategy = 'most_frequent'))])
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean')),('scale',MinMaxScaler())])

# Encoding - One Hot Encoder to convert Categorical data to Numeric values
# Categorical features
encoding_pipeline = Pipeline(steps = [('onehot', OneHotEncoder(sparse_output = False))]) #(sparse_output = Flase) dosent give output as sparse_matrix 



#==============================================================================================
from sklearn.compose import ColumnTransformer
# Creating a transformation of variable with ColumnTransformer()
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features), ('categorical', encoding_pipeline, categorical_features)])
# Fit the data
clean = preprocessor.fit(X) 

import joblib
# Save the pipeline
joblib.dump(clean, 'clean_NB ') 

# Transform the original data
clean2 = pd.DataFrame(clean.transform(X), columns = clean.get_feature_names_out()) 
clean2.columns

# complete EDA for X variables

#=====================================================================================================
# target variable :
    
y.info() 

#=====================================================================================================




'''
step: 3] model building (data mining)
    
'''


from sklearn.naive_bayes import MultinomialNB 

mnb = MultinomialNB()
mnb.fit(clean2, y)



from sklearn.metrics import accuracy_score

# Make predictions on the training data
y_train_pred = mnb.predict(clean2)

# Calculate the training accuracy
training_accuracy = accuracy_score(y, y_train_pred)
training_accuracy


# save this model
import pickle 
pickle.dump(mnb, open('multinomialNB.pkl', 'wb'))










# testing on new data:
    
############################
# Predictions on New Data

# Load the preprocessing pipeline
clean = joblib.load('clean_NB')
# Load the Naive Bayes model
mnb = pickle.load(open('multinomialNB.pkl', 'rb'))


test = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\ASSIGNMENTS\SUPERVISED LEARNING\CLASSIFICATION\naive bayes\SalaryData_Test.csv")    

# Split data into features (X) and target variable (y)
Y2 = test["Salary"]
Y2.shape


# Apply the preprocessing pipeline to the testing data
test_clean = pd.DataFrame(clean.transform(test), columns=clean.get_feature_names_out())

# Make predictions on the testing data 
y_test_pred = mnb.predict(test_clean)


# Calculate the training accuracy
testing_accuracy = accuracy_score(Y2, y_test_pred) 
testing_accuracy


'''
solution:  By building a multinomial naive bayes predictive model for salary, the client can make more informed decisions about compensation,
           budgeting, and financial planning. This can help in optimizing salaries for employees and managing overall costs.

'''








