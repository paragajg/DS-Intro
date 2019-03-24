
# coding: utf-8

# # News Aggregator
# 
# ## NLP: Topic Modelling
# 
# ### Keys Concepts
# - Latent Dirichlet Allocation
# - Latent Semantic Analysis

# In[12]:


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
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
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

# In[3]:


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


# <h2>Latent Dirichlet Allocation</h2>
# <h3>From Documents -- DTM -- LDA Model</h3>
# 
# Topic modeling aims to automatically summarize large collections of documents to facilitate organization and management, as well as search and recommendations. At the same time, it can enable the understanding of documents to the extent that humans can interpret the descriptions of topics
# 
# <img src="images/lda2.png" alt="lda" style="width:60%">
# <img src="images/docs_to_lda.png" alt="ldaflow" style="width:100%">

# ## Find Topics in Documents and group them: First Step to Build News Aggregator

# ### Convert Raw text --> Parsed Text --> Document Term Matrix --> Topic Models --> Extract Topics

# ### How Topic Modelling Works
# 
# LDA  assumes topics are probability distributions over words, and documents are distributions over topics. Which implies that documents cover only a small set of topics, and topics use only a small set of words frequently.
# 
# __The Mathematical Formula:__
# 
# <img src="images/ldaformula.png" alt="ldaflow" style="width:100%">
# 
# 
# ### Important Reading Article on Topic Models
# https://medium.com/@lettier/how-does-lda-work-ill-explain-using-emoji-108abf40fa7d

# In[ ]:


# Split Data to train & test Topic Models
train_docs, test_docs = train_test_split(docs, 
                                         stratify=docs.Category, 
                                         test_size=50, 
                                         random_state=42)


# In[ ]:


train_docs.shape, test_docs.shape


# In[ ]:


vectorizer = TfidfVectorizer(max_df=.2, 
                             min_df=.01, 
                             stop_words='english')

train_dtm = vectorizer.fit_transform(train_docs["Article"])
words = vectorizer.get_feature_names()
train_dtm


# ### Assign Number of Topics to be extracted

# In[ ]:


n_components = 5
topic_labels = ['Topic {}'.format(i) for i in range(1, n_components+1)]


# In[ ]:


lda_base = LatentDirichletAllocation(n_components=n_components,
                                     n_jobs=-1,
                                     learning_method='batch',
                                     max_iter=10)
lda_base.fit(train_dtm)


# In[ ]:


# Save the model on disk
joblib.dump(lda_base,'lda_10_iter.pkl')


# ### Explore topics & word distributions

# In[ ]:


# pseudo counts
topics_count = lda_base.components_
print(topics_count.shape)
topics_count[:5]


# In[ ]:


topics_prob = topics_count / topics_count.sum(axis=1).reshape(-1, 1)
topics = pd.DataFrame(topics_prob.T,
                      index=words,
                      columns=topic_labels)
topics.head()


# In[ ]:


sns.heatmap(topics, cmap='Blues')


# In[ ]:


top_words = {}
for topic, words_ in topics.items():
    top_words[topic] = words_.nlargest(10).index.tolist()
pd.DataFrame(top_words)


# ### Evaluate Fit on Train Set

# In[ ]:


train_preds = lda_base.transform(train_dtm)
train_preds.shape


# In[ ]:


train_eval = pd.DataFrame(train_preds, columns=topic_labels, index=train_docs.Category)
train_eval.head()


# In[ ]:


train_eval.groupby(level='Category').mean().plot.bar(title='Avg. Topic Probabilities')


# In[ ]:


df = train_eval.groupby(level='Category').idxmax(
    axis=1).reset_index(-1, drop=True)
sns.heatmap(df.groupby(level='Category').value_counts(normalize=True)
            .unstack(-1), annot=True, fmt='.1%', cmap='Blues', square=True)
plt.title('Train Data: Topic Assignments')


# ### Evaluate Fit on Test Set

# In[ ]:


test_dtm = vectorizer.transform(test_docs.Article)
test_dtm


# In[ ]:


test_preds = lda_base.transform(test_dtm)
test_eval = pd.DataFrame(test_preds, columns=topic_labels, index=test_docs.Category)
test_eval.head()


# In[ ]:


df = test_eval.groupby(level='Category').idxmax(
    axis=1).reset_index(-1, drop=True)
sns.heatmap(df.groupby(level='Category').value_counts(normalize=True)
            .unstack(-1), annot=True, fmt='.1%', cmap='Blues', square=True)
plt.title('Topic Assignments');


# In[56]:


# Create Training Dataframe for Comparing Misclassified documents
train_opt_eval = pd.DataFrame(data=lda_base.transform(train_dtm),
                              columns=topic_labels,
                              index=train_docs.Category)


# In[57]:


test_opt_eval = pd.DataFrame(data=lda_base.transform(test_dtm),
                              columns=topic_labels,
                              index=test_docs.Category)


# In[ ]:


## Comparing Side by Side
fig, axes = plt.subplots(ncols=2)
source = ['Train', 'Test']
for i, df in enumerate([train_opt_eval, test_opt_eval]):
    df = df.groupby(level='Category').idxmax(
    axis=1).reset_index(-1, drop=True)
    sns.heatmap(df.groupby(level='Category').value_counts(normalize=True)
            .unstack(-1), annot=True, fmt='.1%', cmap='Blues', square=True, ax=axes[i])
    axes[i].set_title('{} Data: Topic Assignments'.format(source[i]));


# ### Misclassified Documents

# In[ ]:


test_assignments = test_opt_eval.groupby(level='Category').idxmax(axis=1)
test_assignments = test_assignments.reset_index(-1, drop=True).to_frame('predicted').reset_index()
test_assignments['heading'] = test_docs.Heading.values
test_assignments['Article'] = test_docs.Article.values
test_assignments.head()

