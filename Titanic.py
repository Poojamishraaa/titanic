#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


titanic_data=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/titanic_train.csv")
titanic_data


# In[4]:


titanic_data.head()


# In[5]:


titanic_data.shape


# In[6]:


sns.countplot(x='Survived',data=titanic_data)


# In[8]:


sns.countplot(x='Survived',hue='Sex',data=titanic_data,palette='winter')


# In[10]:


sns.countplot(x='Survived',hue='Sex',data=titanic_data,palette='PuBu')


# In[12]:


titanic_data['Age'].plot.hist()


# In[15]:


titanic_data['Fare'].plot.hist(bins=20,figsize=(10,5))


# In[16]:


sns.countplot(x='SibSp',data=titanic_data,palette='rocket')


# In[18]:


titanic_data['Parch'].plot.hist()


# In[19]:


sns.countplot(x='Parch',data=titanic_data,palette='summer')


# In[20]:


titanic_data.isnull().sum()


# In[21]:


sns.heatmap(titanic_data.isnull(),cmap='spring')


# In[22]:


sns.boxplot(x='Pclass',y='Age',data=titanic_data)


# In[23]:


titanic_data.head()


# In[25]:


titanic_data.drop('Cabin',axis=1,inplace=True)


# In[26]:


titanic_data.head()


# In[28]:


titanic_data.dropna (inplace=True)


# In[30]:


sns.heatmap(titanic_data.isnull(),cbar=False)


# In[32]:


titanic_data.isnull().sum()


# In[34]:


titanic_data.head(2)


# In[36]:


pd.get_dummies(titanic_data['Sex'].head())


# In[37]:


sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)


# In[39]:


sex.head(3)


# In[42]:


embarked=pd.get_dummies(titanic_data['Embarked'])
embarked.head(3)


# In[43]:


embarked=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embarked.head(3)


# In[45]:


pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
pcl.head(3)


# In[47]:


titanic_data=pd.concat([titanic_data,sex,embark,pcl],axis=1)
titanic_data.head(3)


# In[51]:


titanic_data.drop(['Name','PassengerId','Pclass','Ticket','Sex','Embarked'],axis=1,inplace=True)
titanic_data.head(3)


# In[62]:


x=titanic_data.drop('Survived',axis=1)
y=titanic_data['Survived']


# In[63]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=4)


# In[66]:


from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()


# In[67]:


lm.fit(x_train,y_train)


# In[70]:


prediction=lm.predict(x_test)


# In[71]:


from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)


# In[ ]:




