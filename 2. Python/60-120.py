
# coding: utf-8

# ## Philosphy:  Lets learn programming language to solve a problem and not to solve coding complexities.

# ### Mall Customer Segmentation Data
# https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
# 
# #### Description of dataset: 
# This file contains the basic information (ID, age, gender, income, spending score) about the customer
# 
# #### Explore the dataset for follwoing questons:
# - Are male customers more likely to spend as compared to female customers ?
# - Does age of a customer impact spending nature ?
# - Does Annual Income play a role in spendings by a customer ?
# - Build a Machine Learning model that learns to predict Spending S|core if Age,Gender & Annual Income provided to it.

# In[57]:


# Pythonic way to read a file
with open('Mall_Customers.csv',"r") as file: # f is a file handler, while "r" is the mode (r for read)
    for line in file:
        print(line)


# ### Refer below for more details on filehandlers in python
# https://docs.python.org/3.7/tutorial/inputoutput.html

# In[ ]:


# Initiate lists to store individual column values
read_list = []
customerID =[]
gender = []
age =[]
annual_income_dollar = []
spending_score = []

# Use Python I/O file handlers
with open('Mall_Customers.csv') as file:
    for f in file:
        temp = f.split(',') # splitting each line as values are separated by ','
        print(temp)
        customerID.append(temp[0]) # appending each value to respective list
        gender.append(temp[1])
        age.append(temp[2])
        annual_income_dollar.append(temp[3])
        spending_score.append(temp[4])

# Create dictionary to combine data
mall_dict = {customerID[0]:customerID[1:], gender[0]:gender[1:],age[0]:age[1:], 
             annual_income_dollar[0]: annual_income_dollar[1:], 
             spending_score[0]:spending_score[1:]}


# In[ ]:


mall_dict


# In[ ]:


temporary_list = []
for x in spending_score[1:]:
    temp = x.split('\n')
    temp = temp[0]
    temp = int(temp)
    temporary_list.append(temp)
print(temporary_list)

listComprehension = [int(x.split('\n')[0]) for x in spending_score[1:]]
listComprehension


# In[ ]:


mall_dict['spending_score'] = [int(x.split('\n')[0]) for x in spending_score[1:]]
#print(mall_dict)


# __List Comprehensions__. List comprehensions is a pythonic way to provide a concise way to create lists. It consists of brackets containing an expression followed by a for clause, then zero or more for or if clauses. 

# In[ ]:


mall_dict = {customerID[0]:customerID[1:], gender[0]:gender[1:],age[0]:[int(x) for x in age[1:]], 
             annual_income_dollar[0]: [int(x) for x in annual_income_dollar[1:]]
             , spending_score[0]: [int(x.split('\n')[0]) for x in spending_score[1:]]}
print(mall_dict)


# ### Now thats tedious !!!! To Write so many lines to export a file :(

# ### Libraries
# Often times, we need either internal or external help for complicated computation tasks. In these occasions, we need to _import libraries_. 
# 
# #### Built in Packages
# Python provides many built-in packages to prevent extra work on some common and useful functions
# We will use __math__ as an example.
# 
# #### Most Commonly used Packags for Machine Learning in Python:
# - Pandas http://pandas.pydata.org/pandas-docs/stable/reference/index.html
# - Numpy https://docs.scipy.org/doc/numpy/user/quickstart.html
# - Scipy https://docs.scipy.org/doc/scipy/reference/
# - Matplotlib https://matplotlib.org/contents.html
# - Scikit learn https://scikit-learn.org/stable/user_guide.html

# ### Pandas (Python Data Analysis Library) is a great package for data structures: DataFrame)
# A great library to slic and dice data and visualize it in a columnar format. Its API provides wide range of functionalities that makes data analysis a fun and true strength of Python as prefered choice over other programming languages.

# In[71]:


# Export the Mall dataset using Pandas
import pandas as pd # loading pandas library and giving it an alise pd to make code less verbose
import numpy as np


# In[69]:


df = pd.read_csv('Mall_Customers.csv')


# In[75]:


type(df)


# In[72]:


df.head(n=10)


# In[73]:


df.columns


# In[74]:


print(df.head()) # top n reocrds
print(df.tail(n=10)) # last n records
print(df.shape) # rows x columns


# In[76]:


df.describe()


# In[54]:


df.groupby(['Gender']).mean()

