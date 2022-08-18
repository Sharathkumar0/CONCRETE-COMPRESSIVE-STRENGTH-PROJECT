#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libraries
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Load the dataset
dataset = pd.read_excel('Concrete_Data.xls')


# In[3]:


#Data shape
dataset.shape


# In[4]:


#Checking for NA values
dataset.isna().sum()


# In[5]:


# Assigning the dependent and independent variables
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[7]:


#Creating pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

stdscaler = StandardScaler()
RFR = RandomForestRegressor()

RFR_model = Pipeline([
    ('Standard_Scaler',stdscaler),
    ('Regressor',RFR)
])


# In[30]:


#Perform train and split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)


# In[31]:


#Train the model
RFR_model.fit(xtrain,ytrain)


# In[32]:


# Train score
round(RFR_model.score(xtrain,ytrain),2)


# In[33]:


# Test score
round(RFR_model.score(xtest,ytest),2)

