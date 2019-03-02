
# coding: utf-8

# ## Unsupervised Learning Algorithms
# 
# ### Clustering Analysis
# - #### Kmeans Clustering Algorithm
# - #### Agglomerative Hierrarchial Clustering

# ### Kmeans Clustering
# 
# - Clustering algorithms seek to learn, from the properties of the data, an optimal division or discrete labeling of groups of points.
# 
# - Many clustering algorithms are available in Scikit-Learn and elsewhere, but perhaps the simplest to understand is an algorithm known as k-means clustering
# 
# Reading Reference for Clustering Algorithms -
# https://scikit-learn.org/stable/modules/clustering.html#clustering

# ### Step1: Load Libraries

# In[2]:


import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering


# ### Step2: Load Data

# In[70]:


df_raw = pd.read_csv("./data/world-happiness-report/2017.csv")
df_raw.head()


# ### Step3: Explore Data

# In[71]:


df_raw.describe()


# In[89]:


# Plot Correlation Heatmap to analyse correlation between continuous variables
df = df_raw[['Happiness.Score','Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.', 'Freedom', 
          'Generosity','Trust..Government.Corruption.','Dystopia.Residual']] #Subsetting the data
cor = df.corr() #Calculate the correlation of the above variables

fig,ax = plt.subplots()
fig.set_figwidth(8)
fig.set_figheight(8)
sns.heatmap(cor, square = True,annot=True,cmap="YlGnBu") #Plot the correlation as heat map


# ### Step4: Preprocessing the data

# In[ ]:


# Scale all the variables to similar scale to avoid any biasness due to variabtion in their measurement scale
#Scaling of data
ss = StandardScaler()
ss.fit_transform(df)


# ### Step5: Kmean Approach to identify clusters
# The k-means algorithm searches for a pre-determined number of clusters within an unlabeled multidimensional dataset. It accomplishes this using a simple conception of what the optimal clustering looks like:
# 
# - The "cluster center" is the arithmetic mean of all the points belonging to the cluster.
# - Each point is closer to its own cluster center than to other cluster centers.
# 
# Those two assumptions are the basis of the k-means model.

# In[90]:


def doKmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)


# In[91]:


clust_labels, cent = doKmeans(df, 2)
kmeans = pd.DataFrame(clust_labels)
df.insert((df.shape[1]),'kmeans',kmeans)


# In[ ]:


# Just to check if kmeans column added to the dataframe
df.head()


# In[ ]:


# Plot scatter plot
fig, (ax0,ax1) = plt.subplots(ncols= 2)
fig.set_figwidth(16)
fig.set_figheight(6)

# Scatter plots of Corruption vs GDP
scatter = ax0.scatter(df['Economy..GDP.per.Capita.'],df['Trust..Government.Corruption.'],
                     c=kmeans[0],s=50)
ax0.set_title('K-Means Clustering')
ax0.set_xlabel('GDP per Capita')
ax0.set_ylabel('Corruption')

# Scatter plots of Corruption vs GDP

scatter = ax1.scatter(df['Freedom'],df['Trust..Government.Corruption.'],
                     c=kmeans[0],s=50)
ax1.set_title('K-Means Clustering')
ax1.set_xlabel('Freedom')
ax1.set_ylabel('Corruption')

plt.colorbar(scatter)


# In[95]:


# Convert Cluster to

df["cluster_name"] = df["kmeans"].apply(lambda x: "cluster0" if x == 0 else "cluster1")


# In[96]:


df.tail()


# In[109]:


fig, (ax,ax2,ax3,ax4) = plt.subplots(nrows = 4)
fig.set_figwidth(16)
fig.set_figheight(24)
x = sns.boxplot(x="cluster_name", y='Freedom', data=df, ax= ax)
x = sns.swarmplot(x="cluster_name", y='Freedom', data=df,color=".25",ax= ax)
# Cluster analysis of employment
x = sns.boxplot(x="cluster_name", y='Economy..GDP.per.Capita.', data=df, ax= ax2)
x = sns.swarmplot(x="cluster_name", y='Economy..GDP.per.Capita.', data=df,color=".25",ax= ax2)
# Cluster analysis of employment
x = sns.boxplot(x="cluster_name", y='Happiness.Score', data=df, ax= ax3)
x = sns.swarmplot(x="cluster_name", y='Happiness.Score', data=df,color=".25",ax= ax3)
# Cluster analysis of employment
x = sns.boxplot(x="cluster_name", y='Family', data=df, ax= ax4)
x = sns.swarmplot(x="cluster_name", y='Family', data=df,color=".25",ax= ax4)

