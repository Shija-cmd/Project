#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[69]:


import pandas as pd 
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import pickle


# ### Loading dataset

# In[70]:


data = pd.read_csv("Shortlistingsystem_MASTERS.csv")


# In[71]:


data.head()


# ### Droping serial No column

# In[74]:


data.drop(columns='Serial No.',inplace=True)


# In[75]:


data.head()


# In[76]:


data.dtypes


# ### Defining dependent and independent variables

# In[79]:


x = data.iloc[:, :3]
y = data.iloc[:, -1]


# ### Training the model

# In[80]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
model = LogisticRegression()
model.fit(x_train,y_train)


# ### Saving the model

# In[81]:


model = pickle.dump(model,open('model.pkl','wb'))


# ### Loading the model to compare the result

# In[83]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4.5, 1.0, 5.5]]))


# In[ ]:




