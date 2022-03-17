#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[20]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


# # Importing Data

# In[2]:


data = pd.read_json('C:\\Users\\Omar Atef\\Desktop\\Knowledge Officer\\Datasets\\articles.json')


# # Balancing Data

# In[3]:


data_eng = data[data['category'] == 'Engineering']
data_pd = data[data['category'] == 'Product & Design']
data_sb = data[data['category'] == 'Startups & Business']

engineering_data = data_eng
product_design_data = data_pd.head(len(data_pd) - 294)
startup_business_data = data_sb.head(len(data_sb) - 513)


# In[4]:


data_balanced = pd.concat([engineering_data, product_design_data, startup_business_data])
data_balanced = shuffle(data_balanced)
data_balanced = data_balanced.reset_index(drop=True)

data_new = data_balanced.drop(['title'], axis=1)

data_new['category'] = data_new['category'].replace({'Engineering': 0, 'Product & Design': 1, 'Startups & Business': 2})


# In[5]:


len(data_new)*0.7


# # Train-Test Split

# In[6]:


data_train = data_new.iloc[:1170,:]
data_test = data_new.iloc[1170:,:]


# In[7]:


x_train = data_train['body']
y_train = data_train['category']
x_test = data_test['body']
y_test = data_test['category']


# # Data Preprocessing

# In[8]:


cv = CountVectorizer(stop_words='english', analyzer='word', min_df=2)


# In[9]:


x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)


# In[10]:


x_train_cv.toarray()


# In[11]:


x_test_cv.toarray()


# In[12]:


tfidf = TfidfTransformer()


# In[13]:


x_train_tfidf = tfidf.fit_transform(x_train_cv)
x_test_tfidf = tfidf.transform(x_test_cv)


# In[14]:


x_train_tfidf.shape


# In[15]:


x_test_tfidf.shape


# # Training and Testing the Model

# In[16]:


NB = MultinomialNB().fit(x_train_tfidf, y_train)


# In[17]:


predictions = NB.predict(x_test_tfidf)


# In[18]:


np.mean(predictions == y_test)


# In[22]:


scores = cross_val_score(NB, x_train_tfidf, y_train, cv=5, scoring="accuracy")
scores


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




