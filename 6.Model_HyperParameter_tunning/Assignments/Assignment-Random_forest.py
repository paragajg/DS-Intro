
# coding: utf-8

# ## Boston Housing Data Set
# 
# ### Assignment Goals:
# - Build Machine Learning model to predict Category of Income of an individual
# - Use pipeline and grid search to build strategy for experimenting your ML model 

# ### Load Libraries

# In[1]:


# Data manipulation libraries
import pandas as pd
import numpy as np

##### Scikit Learn modules needed for Logistic Regression
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler , StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Library to store and load models
import joblib

# Plotting libraries
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load Data

# In[2]:


df = pd.read_csv("boston_housing_data.csv")
df.head()


# In[3]:


print(df.describe())
df.isna().sum()


# ### Visualize Data
# 
# - Use correlation plot (as shown in Decision Tree & Regression Models in class) to study correlation between numerical variables

# In[ ]:


df.columns


# In[4]:


# Explore data visually
# Build Correlation Matrix to
correlation = df[["age","fnlwgt","education_num","capital_gain","capital_loss","hr_per_week"]].corr()
#print(correlation)

fig , ax = plt.subplots()
fig.set_figwidth(8)
fig.set_figheight(8)
sns.heatmap(correlation,annot=True,cmap="YlGnBu")


# ### Visualization Insight:
# From correlation matrix we can see that there isn't a strong correlation between the numerical variables. The Pearson correlation coeeficients are also closer to 0 demonstrating weak correlations among variables.

# ### Build Strategy for your Machine Learning Pipeline
# - Define transformation of categorical variables
# - Define scaling for numerical variables

# In[5]:


# We create the preprocessing pipelines for both numeric and categorical data.

numeric_features = ['age', 'fnlwgt',
                    'education_num','capital_gain', 'capital_loss', 'hr_per_week'] # add names of numerical variables which you want to add for building model
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])## Add your choice of scaler

categorical_features = ['workclass','education','marital_status', 'occupation', 'relationship', 'race',
       'sex'] #  add names of categorical variables which you want to add for building model
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # Experiment with other label encoding techiques as well
 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_jobs=-1))])# Change classifier and try RandomForest & Logistic Regession as well


# ### Split your data

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(df[['age', 'workclass', 'fnlwgt', 'education',
       'education_num', 'marital_status', 'occupation', 'relationship', 'race',
       'sex', 'capital_gain', 'capital_loss', 'hr_per_week']],df['income'],test_size=0.2,random_state=42)


# In[7]:


# Fit your model to check accuracy
clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))


# ### Experiment with Hyper Parameters using Grid Search
# 
# Reference on Grid Search https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
# 
# - Refer to individual models on scikit learn to know more about options in hyper parameters associated with Decision Trees , Logistic Regression and Random Forest

# In[8]:


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__criterion': ["gini","entropy"],
    'classifier__n_estimators': [10,100],
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
grid_search.fit(X_train, y_train)

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
grid_search.fit(X_train, y_train)

print(("best Model from grid search: %.3f"
       % grid_search.score(X_test, y_test)))


# In[9]:


# Print your best combination of hyper parameters
grid_search.best_params_


# ### Store your model using joblib Library

# In[10]:


joblib.dump(grid_search,"Randomforest.model")

