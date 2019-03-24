
# coding: utf-8

# # Search Engine
# 
# ## NLP: Document Similarity
# 
# ### Keys Concepts
# - Term Document Matrix
# - Cosine Similarity

# In[14]:


import warnings
from collections import OrderedDict
from pathlib import Path
from random import randint
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# sklearn for feature extraction & modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# Iteratively read files
import glob
import os

# For displaying images in ipython
from IPython.display import HTML, display


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14.0, 8.7)
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format


# ### Load data

# In[44]:


# User defined function to read and store bbc data from multipe folders
def load_data(folder_names,root_path):
    fileNames = [path + '/' + 'bbc' +'/'+ folder + '/*.txt' for path,folder in zip([root_path]*len(folder_names),
                                                                               folder_names )]
    doc_list = []
    tags = folder_names
    for docs in fileNames:
        #print(docs)
        #print(type(docs))
        doc = glob.glob(docs) # glob method iterates through the all the text documents in a folder
        for text in doc:
            with open(text, encoding='latin1') as f:
                topic = docs.split('/')[8]

                lines = f.readlines()
                heading = lines[0].strip()
                body = ' '.join([l.strip() for l in lines[1:]])
                doc_list.append([topic, heading, body])
        print("Completed loading data from folder: %s"%topic)
    
    print("Completed Loading entire text")
    
    return doc_list


# In[ ]:


folder_names = ['business','entertainment','politics','sport','tech']
docs = load_data(folder_names = folder_names, root_path = os.getcwd())


# In[ ]:


docs = pd.DataFrame(docs, columns=['Category', 'Heading', 'Article'])
print(docs.head())
print('\nShape of data is {}\n'.format(docs.shape))
print(docs.info())


# <h2>Documents Similarity</h2>
# <h3>From Documents -- DTM -- Cosine Similarity</h3>
# 
# HTML("<table><tr><td><img src="images/docs_to_dtm.png" alt="dtm" style="width:100%"></td><td><img src="images/cosine.jpg" alt="Forest" style="width:100%"></td></tr></table>")
# 
# <br>
# 
# 
# ### Important
# The cosine similarity is the cosine of the angle between two vectors.
# - Cosine Similarity can take value between 0 to 1.
# - closer to 0 means dissimilar documents
# - closer to 1 means similar documents

# ## Find Documents Similar to a New document : First Step to Build Search Engine

# ### Convert Raw text --> Parsed Text --> Document Term Matrix

# ### Why to Use Term Frequency Inverse Document Frequency over Term Frequency
# 
# - TF*IDF is an information retrieval technique that weighs a term’s frequency (TF) and its inverse document frequency (IDF). Each word or term has its respective TF and IDF score. The product of the TF and IDF scores of a term is called the TF*IDF weight of that term.
# 
# __Put simply, the higher the TF*IDF score (weight), the rarer the term and vice versa.__
# 
# __TFidf__ - is comprised of following two components
# - __TF: Term Frequency__, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: 
# 
# __TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).__
# 
# __IDF: Inverse Document Frequency__, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: 
# 
# __IDF(t) = log_e(Total number of documents / Number of documents with term t in it).__
# 
# __TFidf__ = TF * IDF
# 
# **Important**
# Higher the TFidf , more important the significance of word to that document
# 
# ### Important Reading Article on TFidf
# https://www.kdnuggets.com/2018/08/wtf-tf-idf.html

# In[71]:


vectorizer = TfidfVectorizer(stop_words = "english")


# In[75]:


vectors = vectorizer.fit_transform(docs["Heading"])
print("Shape of tfidf matrix: {}".format(vectors.shape))


# In[76]:


from sklearn.metrics.pairwise import cosine_similarity


# In[79]:


new_query = ["World facing imminent danger across global war theaters"]
new_query_vector = vectorizer.transform(new_query)
new_query_vector


# In[80]:


sim = cosine_similarity(X = vectors, Y = new_query_vector)


# In[119]:


# Extract Index of Maximum valued similar document
argmax = np.argmax(sim)
print("Index of maximum valued similar doc: %s"%argmax)
print("Retrieved Document Header: %s"%docs["Heading"][argmax])


# In[142]:


import bottleneck


# In[ ]:


# To Extract Top 10 Similar Documents against the new query
ind = np.argsort(sim,axis = 0)[::-1][:10]
for i in ind:
    print(docs["Heading"].values[i])


# In[177]:


def retrieve_doc(new_query,raw_docs):
    vectorizer = TfidfVectorizer(stop_words = "english")
    vectors = vectorizer.fit_transform(docs["Heading"])
    print("Shape of tfidf matrix: {}".format(vectors.shape))
    new_query = [new_query]
    new_query_vector = vectorizer.transform(new_query)
    sim = cosine_similarity(X = vectors, Y = new_query_vector)
    ind = np.argsort(sim,axis = 0)[::-1][:10]
    for i in ind:
        print(docs["Heading"].values[i])


# In[182]:


newQuery = "Outbreak of deadly virus"


# In[ ]:


retrieve_doc(new_query= newQuery  , raw_docs= docs["Heading"])

