
# coding: utf-8

# ## Basic Statistics using NumPy, Pandas and Jupyter Notebook

# ### What is Statistics?
# Statistics is a discipline that uses data to support claims about populations. These “populations” are what we refer to as “distributions.” Most statistical analysis is based on probability, which is why these pieces are usually presented together. More often than not, you’ll see courses labeled “Intro to Probability and Statistics” rather than separate intro to probability and intro to statistics courses. This is because probability is the study of random events, or the study of how likely it is that some event will happen.

# In[1]:


## As a practice we will always load the packages at the top
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns # seaborn is a package built on top of matplotlib and used for statistical plots
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load data
df = pd.read_csv('Mall_Customers.csv')


# ## Describing Data
# 
# <img src ="./img/data_description.jpg">

# ### Mean
# arithmetic average of a range of values or quantities, computed by dividing the total of all values by the number of values.

# In[3]:


df.rename(columns={'CustomerID':'id','Spending Score (1-100)':'score','Annual Income (k$)':'income'},inplace=True)
df.head() # Visualize first 5 rows of data


# In[4]:


# Mean of Age
meanAge = np.mean(df['Age'])
print('Mean of Age of Customers in data: %.2f years'%meanAge)
print('Mean of Age of Customers using pandas: %.2f years'% df['Age'].mean())


# ### Median
# Denotes value or quantity lying at the midpoint of a frequency distribution of observed values or quantities, such that there is an equal probability of falling above or below it. Simply put, it is the middle value in the list of numbers.

# In[5]:


medianAge = np.median(df['score'])
print('Median of Spending Score of Customers in data: %.2f'% medianAge)
print('Median of Spending Score of Customers using pandas: %.2f'% df['score'].median())


# ### Mode
# It is the number which appears most often in a set of numbers.

# In[7]:


from scipy import stats # scipy is a package built on top of numpy and used mainly for Statistics & Probability
print('Mode of score is {} '.format(stats.mode(df['score'])))
print('Mode of score is %s '% stats.mode(df['score'])[0][0]) # extracting only the mode value
# using pandas
print('Mode of score is %s '% df['score'].mode()) # pandas


# In[ ]:


df['score'].value_counts() # Counts occurrence of each value in a given column


# ### Variance
# 
# > Once two statistician of height 4 feet and 5 feet have to cross a river of AVERAGE depth 3 feet. Meanwhile, a third person comes and said, "what are you waiting for? You can easily cross the river"
# 
# It's the average distance of the data values from the *mean*
# 
# <img style="float: left;" src="img/variance.png" height="320" width="320">

# In[9]:


#Using scipy.stats built in function
print('Variation in Age of customers is {0:.2f}'.format(np.var(df['Age'])))


# In[ ]:


# writing user defined function
def variance(x):
    #Mean = np.mean(x)
    diffSqr = np.subtract(x,np.mean(x))**2
    sumSqr = np.sum(diffSqr)
    variance = sumSqr / (len(x)-1)
    return variance

variance(df['Age'])


# ### Standard Deviation
# It is the square root of variance. This will have the same units as the data and mean.

# In[10]:


print('Standard Deviation in Age of customers is {0:.2f}'.format(np.std(df['Age'])))


# ### Range
# Its is the difference in Maximum and Minimum value of a data set

# In[11]:


rng = df['income'].max() - df['income'].min()
print('Range of annual incomes of Customer is {0:.2f}'.format(rng))


# # Distribution
# > the way in which something is shared out among a group or spread over an area
# 
# ### Random Variable
# > a variable whose value is subject to variations due to chance (i.e. randomness, in a mathematical sense). A random variable can take on a set of possible different values (similarly to other mathematical variables), each with an associated probability [wiki](https://en.wikipedia.org/wiki/Random_variable)
# 
# **Types**
# 
# 1. Discrete Random Variables <br>
#     Eg: Genders of the buyers buying shoe
# 2. Continuous Random Variables <br>
#     Eg: Shoe Sales in a quarter
#     
# ### Probability Distribution
# > Assigns a probability to each measurable subset of the possible outcomes of a random experiment, survey, or procedure of statistical inference. [wiki](https://en.wikipedia.org/wiki/Probability_distribution)
# 
# #### Probability Mass Function
# probability mass function (pmf) is a function that gives the probability that a discrete random variable is exactly equal to some value
# 
# #### Discrete probability distribution(Cumulative Mass Function)
# probability distribution characterized by a probability mass function
# 
# #### Probability Density Function
# function that describes the relative likelihood for this random variable to take on a given value
# 
# #### Continuous probability distribution(Cumulative Density function)
# probability that the variable takes a value less than or equal to `x`
# 
# ### Central Limit Theorem
# Given certain conditions, the arithmetic mean of a sufficiently large number of iterates of independent random variables, each with a well-defined expected value and well-defined variance, will be approximately normally distributed, regardless of the underlying distribution. [wiki](https://en.wikipedia.org/wiki/Central_limit_theorem)
# 
# #### Normal Distribution
# A bell shaped distribution. It is also called Gaussian distribution
# 
# <img style="float: left;" src="img/normaldist.png" height="220" width="220">
# <br>
# <br>
# <br>
# <br>
# 
# 
# 
# **PDF**
# <br>
# <br>
# <img style="float: left;" src="img/normal_pdf.png" height="320" width="320">
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# 
# **CDF**
# <br>
# <br>
# 
# 
# <img style="float: left;" src="img/normal_cdf.png" height="320" width="320">
# 
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# 
# 
# #### Skewness
# Measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. [wiki](https://en.wikipedia.org/wiki/Skewness)
# 
# <img style="float: left;" src="img/skewness.png" height="620" width="620">
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# #### Kurtosis
# Measure of the "peakedness" of the probability distribution of a real-valued random variable [wiki](https://en.wikipedia.org/wiki/Kurtosis)
# <br>
# <br>
# <img style="float: left;" src="img/kurtosis.png" height="420" width="420">
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# 
# #### Binomial Distribution
# 
# Binomial distribution with parameters `n` and `p` is the discrete probability distribution of the number of successes in a sequence of n independent yes/no experiments, each of which yields success with probability p. A success/failure experiment is also called a Bernoulli experiment or Bernoulli trial; when n = 1, the binomial distribution is a Bernoulli distribution  [wiki](https://en.wikipedia.org/wiki/Binomial_distribution)
# <br>
# <br>
# <img style="float: left;" src="img/binomial_pmf.png" height="420" width="420">
# <br>
# <br>
# <br>
# 
# 
# #### Exponential Distribution
# Probability distribution that describes the time between events in a Poisson process, i.e. a process in which events occur continuously and independently at a constant average rate. It has the key property of being memoryless. [wiki](https://en.wikipedia.org/wiki/Exponential_distribution)
# <br>
# <br>
# <img style="float: left;" src="img/exponential_pdf.png" height="420" width="420">
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# 
# #### Uniform distribution
# All values have the same frequency [wiki](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
# 
# 
# <br> 
# <br>
# <img style="float: left;" src="img/uniform.png" height="420" width="420">
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# 
# 
# 
# 
# ### 6-sigma philosophy
# <img style="float: left;" src="img/6sigma.png" height="520" width="520">

# ### Histograms
# Most commonly used representation of a distribution.

# In[15]:


fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
fig.set_figwidth(16)
fig.set_figheight(16)

sns.distplot(df['Age'],ax =ax1,kde=False,bins=20)
sns.distplot(df['income'],ax =ax2, color = 'g',bins = 10)
sns.distplot(df['score'],ax =ax3, color = 'red')
ax1.set_title('Subplots of histogram')


# ### Five Data Summary of Numerical Data

# In[16]:


df['income'].describe()


# In[18]:


fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(8)
sns.boxplot(x=df['income'])


# In[21]:


df['income'].describe()


# In[20]:


fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(8)
x = sns.boxplot(x="Gender", y="income", data=df)
x = sns.swarmplot(x="Gender", y="income", data=df,color=".25")


# ### Correlation
# 
# Extent to which two or more variables fluctuate together. A positive correlation indicates the extent to which those variables increase or decrease in parallel; a negative correlation indicates the extent to which one variable increases as the other decreases.
# 
# <img style="float: left;" src="img/correlation.gif" height="270" width="270">
# 
# <br>
# <br>
# <br>
# <br>
# 
# #### Question2: Does age of a customer impact spending nature ?

# In[22]:


fig,ax = plt.subplots()
plt.scatter(x = df['Age'], y = df['score'])
print('Correlation between Age and score is %.2f'% stats.pearsonr(df['Age'],df['score'])[0])


# #### Question3: Does Annual Income play a role in spendings by a customer ?

# In[23]:


fig,ax = plt.subplots()
plt.scatter(x = df['income'], y = df['score'])
print('Correlation between Age and score is %.2f'% stats.pearsonr(df['income'],df['score'])[0])


# ### Summary
# 1. Data variable types:
#     - Quantitative: Continuous data (measurable), Discrete data (countable)
#     - Qualitative: Nominal (no order) , Ordinal (ordered / ranked data)
# 2. Difference between Population and Samples
# 3. Difference between Statistic & Parameter
# 4. Data described using - Central Tendency , Variation, Shape
# 5. Distribution - spread of data
# 6. Normal Distribution:
#     - mean = median = mode
#     - 68% of data falls between +1 to-1 std around mean
#     - 95.8% of data falls between +2 to -2 std around mean
#     - 99.78% of data falls between +3 to - 3 std around mean
# 7. Histogram - visualize frequency of data across categories / bins
# 8. Box plot - used to visualize 5 point data summary
# 9. Correlation - relation between two continuous variables
#     - Using Pearson Correlation Coefficient (-1 to +1)
#     - Scatter plots
