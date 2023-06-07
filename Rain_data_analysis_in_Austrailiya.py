#!/usr/bin/env python
# coding: utf-8

# # 1. Problem Statement: 

# # Design a predictive model with the use of machine learning algorithms to forecast whether or not it will rain tomorrow in Australia.

# # 2. Data Source:

# # The dataset is taken from Kaggle and contains about 10 years of daily weather observations from many locations across Australia.

# #Dataset Description:

# #Number of columns: 23
# Number of rows: 145460
# Number of Independent Columns: 22
# Number of Dependent Column: 1

# # 3. Importing Libraries:

# In[112]:


#The first step in any Data Analysis step is importing necessary libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # visualizing data
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


## Load Data Set:
# Dataset can be loaded using a method read_csv().


# In[6]:


RainData = pd.read_csv(r'D:\work\python\Projects\Rain data analysis austrailia\archive\weatherAUS.csv')


# In[7]:


print(RainData.shape)


# # 4. Data Preprocessing:

# In[8]:


RainData.head()


# In[11]:


RainData.info()


# In[31]:


RainData.count().sort_values()


# In[13]:


#check for null values

RainData.isnull().sum()


# In[33]:


# Drop null Values

RainData.dropna(inplace=True)


# In[65]:


# drop date column
RainData.drop(['Date'], axis=1, inplace=True)


# In[66]:


RainData.isnull().sum()


# In[67]:


RainData.describe()


# In[68]:


RainData.describe(include=[object])


# In[69]:


RainData.describe(include='all')


# In[70]:


RainData.nunique()


# In[39]:


RainData.value_counts()


# In[71]:


RainData.value_counts()/len(RainData)


# # 5. Finding Categorical and Numerical Features in a Data set:

# In[72]:


categorical_features = [column_name for column_name in RainData.columns if RainData[column_name].dtype == 'O']
print("Number of Categorical Features: {}".format(len(categorical_features)))
print("Categorical Features: ",categorical_features)


# In[73]:


# view the categorical variables

RainData[categorical_features].head()


# In[74]:


numerical_features = [column_name for column_name in RainData.columns if RainData[column_name].dtype != 'O']
print("Number of Numerical Features: {}".format(len(numerical_features)))
print("Numerical Features: ",numerical_features)


# # 6.Explore RainTomorrow target variable

# In[75]:


RainData['RainTomorrow'].isnull().sum()


# In[44]:


#View number of unique values

RainData['RainTomorrow'].nunique()


# In[45]:


#View the unique values
RainData['RainTomorrow'].unique()


# In[46]:


#View the frequency distribution of values

RainData['RainTomorrow'].value_counts()


# In[47]:


#View percentage of frequency distribution of values

RainData['RainTomorrow'].value_counts()/len(RainData)


# In[48]:


#Visualize frequency distribution of RainTomorrow variable

f, ax = plt.subplots(figsize=(6, 8))
ax = sns.countplot(x="RainTomorrow", data=RainData, palette="Set1")
plt.show()


# # Explore RainToday variable

# In[88]:


# print number of labels in RainToday variable

print('RainToday contains', len(RainData['RainToday'].unique()), 'labels')


# In[89]:


# check labels in WindGustDir variable

RainData['RainToday'].unique()


# In[90]:


# check frequency distribution of values in WindGustDir variable

RainData.RainToday.value_counts()


# In[91]:


# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(RainData.RainToday, drop_first=True, dummy_na=True).head()


# In[92]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(RainData.RainToday, drop_first=True, dummy_na=True).sum(axis=0)


# In[76]:


categorical_features = [column_name for column_name in RainData.columns if RainData[column_name].dtype == 'O']
RainData[categorical_features].isnull().sum()


# In[53]:


RainData['RainTomorrow'].value_counts().plot(kind='bar')


# # 7. Exploratory Data Analysis:

# In[77]:


#Sunshine vs Rainfall:

sns.lineplot(data=RainData,x='Sunshine',y='Rainfall',color='green')


# In[78]:


#Sunshine vs Evaporation:

sns.lineplot(data=RainData,x='Sunshine',y='Evaporation',color='blue')


# # 8. Correlation:

# In[79]:


plt.figure(figsize=(20,20))
sns.heatmap(RainData.corr(), linewidths=0.5, annot=False, fmt=".2f", cmap = 'viridis')


# In[80]:


X = RainData.drop(['RainTomorrow'],axis=1)
y = RainData['RainTomorrow']


# In[81]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[93]:


print("Length of Training Data: {}".format(len(X_train)))
print("Length of Testing Data: {}".format(len(X_test)))


# In[94]:


# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = RainData.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = RainData.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = RainData.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = RainData.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')


# #Check the distribution of variables

# In[95]:


# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = RainData.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = RainData.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = RainData.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = RainData.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')


# In[96]:


correlation = RainData.corr()


# In[97]:


plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           
plt.show()


# In[98]:


num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']


# In[102]:


sns.pairplot(RainData[num_var], kind='scatter', diag_kind='hist', palette='Rainbow')
plt.show()


# In[107]:



pip install pandas_profiling


# In[6]:


import ydata_profiling
import pandas as pd


# In[12]:


# read the file
df = pd.read_csv(r'D:\work\python\Projects\Rain data analysis austrailia\archive\weatherAUS.csv')

# run the profile report
profile = df.profile_report(title='Pandas Profiling Report')
   


# In[9]:


profile


# In[ ]:




