
# coding: utf-8

# ## Introduction to Natural Language Processing

# ### Purpose of Text Mining
# 
# - To extract information from text
# - To identify similarity between sentences \ documents
# - Summarize the intent of a text article \ review
# - Translate human language context to Numerical representation for computers to analyze

# ### Flow of NLP projects
#  <img src="images/nlpflow.png" alt="nlp_flow" style="width:80%;height:200px;">

# ### Load Libraries

# In[14]:


import string
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from gensim.models.phrases import Phraser, Phrases
import pandas as pd


# In[ ]:


# Download stopwords to be removed during parsing
nltk.download('stopwords')


# In[ ]:


# Download data
nltk.download('gutenberg')


# ### Gutenberg free ebooks project
# - Its a corpus / collection of more than 50,000+ ebooks
# - nltk has a small sample of the corpus for studying NLP
# - Gutenberg website: www.gutenberg.org/
# - NLTK reference site: http://www.nltk.org

# In[5]:


from nltk.corpus import gutenberg


# In[6]:


gberg_sents = gutenberg.sents()


# ### Iteratively Access the sentences

# In[ ]:


gutenberg.fileids()


# In[ ]:


gberg_sents[4]


# ### Text Parsing

# #### Lowering of Text

# In[ ]:


lower_text = [w.lower() for w in gberg_sents[4]]
lower_text


# #### Remove stopwords and punctuation:

# In[ ]:


stpwrds = stopwords.words('english') + list(string.punctuation)
stpwrds


# In[ ]:


[w.lower() for w in gberg_sents[4] if w not in stpwrds]


# #### Stemming
# - stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form—generally a written word form.

# In[ ]:


stemmer = PorterStemmer() # Initiate PorterStemmer
[stemmer.stem(w.lower()) for w in gberg_sents[4] if w not in stpwrds]


# ### Assignment - Identify difference between Lemmetization and Stemming in NLP. Perform Lemmetization on the above sentence and observe the difference.

# #### N grams
# In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams typically are collected from a text or speech corpus.

# In[ ]:


phrases = Phrases(gberg_sents) # train detector
# create a more efficient Phraser object for transforming sentences
bigram = Phraser(phrases)
bigram.phrasegrams # output count and score of each bigram


# In[ ]:


"Suresh lives in New York City".split()


# In[ ]:


bigram["Suresh lives in New York City".split()]


# #### Preprocess the Gutenberg corpus

# In[ ]:


lower_sents = []
for s in gberg_sents:
    lower_sents.append([w.lower() for w in s if w not in list(string.punctuation)])

lower_sents[0:5]


# In[19]:


lower_bigram = Phraser(Phrases(lower_sents))


# In[ ]:


lower_bigram.phrasegrams # miss taylor, mr woodhouse, mr weston


# In[ ]:


# By changing the thresholds of the bigram phraser
lower_bigram = Phraser(Phrases(lower_sents, min_count=32, threshold=64))


# In[26]:


final_sentences = [lower_bigram[sentence] for sentence in lower_sents]


# ### Linguistic & Semmantic Annotation

# #### Linguistic Annotation -
# - It includes the application of grammatical rules to identify the boundary of a sentence despite ambiguous punctuation, and a token's role in a sentence for Part of Speech tagging. 
# - It also permits the identification of common root forms for stemming and lemmatization to group related words:
# <br>
# <br>
# <br>
# 
# __POS annotations:__
# - It helps disambiguate tokens based on their function (this may be necessary when a verb and noun have the same form), which increases the vocabulary but may result in better accuracy.
# <br>
# <br>
# 
# __Dependency parsing:__
# - It identifies hierarchical relationships among tokens, is commonly used for translation, and is important for interactive applications that require more advanced language understanding, such as chatbots.
# <br>
# <br>
# 
# __Named entity recognition (NER)__ 
# - aims to identify tokens that represent objects of interest, such as people, countries, or companies. 
# - It is a critical ingredient for applications that, for example, aim to predict the impact of news events or sentiment.

# ### Extract POS tagging from Gutenberg corpus

# In[55]:


from nltk import pos_tag


# In[ ]:


pos_tag(final_sentences[5])


# #### Transform a processed text to Document Term Matrix / Term Document Matrix

# In[57]:


sample = ['Mr. Toad saw the car approching him at a distance.',
         'When you finish your coding you can have your dinner. You need to embrace hard work',
         'The time has come for humans to embrace Computers as equal partners.',
         'Cricket is not just a sport but a passion in India']


# In[59]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


count_vect = CountVectorizer()
dtm = count_vect.fit_transform(sample)
dtm


# In[ ]:


dtm.todense()


# In[ ]:


count_vect.vocabulary_


# ### Building a sample predictive model for News feeds data

# In[65]:


from sklearn.datasets import fetch_20newsgroups


# In[70]:


twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)


# In[ ]:


twenty_train.target_names


# In[ ]:


len(twenty_train.data), len(twenty_train.filenames)


# In[ ]:


# Target lables
twenty_train.target[:10]


# #### Converting to Bag of Words
# - Assign a fixed integer id to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).
# - For each document #i, count the number of occurrences of each word w and store it in X[i, j] as the value of feature #j where j is the index of word w in the dictionary.

# In[ ]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


# #### Training a classifier

# In[91]:


from sklearn.naive_bayes import MultinomialNB
#from sklearn.tree import DecisionTreeClassifier
clf = MultinomialNB().fit(X_train_counts, twenty_train.target)


# ### Lets try to predict

# In[92]:


docs_new = ['World crisis looming around', 'OpenGL on the GPU is fast']


# In[ ]:


X_new_counts = count_vect.transform(docs_new)
predicted = clf.predict(X_new_counts)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))


# ### Further Reading
# - https://medium.com/civis-analytics/an-intro-to-natural-language-processing-in-python-framing-text-classification-in-familiar-terms-33778d1aa3ca
