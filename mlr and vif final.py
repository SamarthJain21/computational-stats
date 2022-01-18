#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# In[18]:


df = pd.read_csv(r'insurance.csv')

max = 1
for i in range(0,1338):
    if(df['charges'][i]>max):
        max = df['charges'][i]
for i in range(0,1338):
    df['charges'][i] = df['charges'][i]*54/max
    
for i in range(0,len(df)):
    if(df['smoker'][i]=="yes"):
        df['smoker'][i]=float(1.0)
    else:
        df['smoker'][i]=float(0)

for i in range(0,len(df)):
    if(df['sex'][i]=="female"):
        df['sex'][i]=float(1.0)
    else:
        df['sex'][i]=float(0)
# df.values
df = df.drop(columns = 'region')

print(df)


# In[19]:


y = df['charges']
df = df.drop(columns = 'charges')


# In[22]:


x0 = []
for i in range(len(df)):
    x0.append(1)
df.insert(0,'X0',x0)


# In[23]:


# df


# In[24]:


def beta(X,Y):
    X = np.array(X)
    Y = np.array(Y) 
    beta = np.matmul(np.transpose(X),X)
    beta = np.matmul(np.linalg.inv(beta),np.matmul(np.transpose(X),Y))
    return beta


# In[25]:


df.sex = df.sex.astype(float)
df.smoker = df.smoker.astype(float)
df.dtypes
beta = beta(df,y)
beta


# In[26]:


data1 = np.array(df)
X1 = data1[:,1]
X2 = data1[:,2]
X3 = data1[:,3]
X4 = data1[:,4]
X5 = data1[:,5]

y1 = beta[0]+beta[1]*X1+beta[2]*X2+beta[3]*X3+beta[4]*X4+beta[5]*X5
# X5.shape
print("y = ",beta[0]+beta[1],'X1 +',beta[2],'X2 +',beta[3],'X3 +',+beta[4],'X4 +',beta[5],'X5')


# In[27]:


def yPredicted(beta,X1,X2,X3,X4,X5):
    return beta[0]+beta[1]*X1+beta[2]*X2+beta[3]*X3+beta[4]*X4+beta[5]*X5


# In[28]:


y1 = (y1-np.array(y))**2
rmsq = np.sum(y)/np.shape(y)
rmsq


# In[29]:


print("Please tell your age")
age = int(input())
print("Please tell your sex (0 for male and 1 for female)")
sex = int(input())
print("Please tell your bmi")
bmi = float(input())
print("How many children do you have?")
children = int(input())
print("Do you smoke(1 for yes , 0 for no)")
smoker = int(input())
print(yPredicted(beta,age,sex,bmi,children,smoker)/54*max)


# In[31]:


# df


# In[32]:


def vif(X,Y):
    N = X.shape[0]
    XYsum = 0
    Xsum = 0
    Ysum = 0
    X2sum = 0
    Y2sum = 0
    for i in range(N):
        Xsum +=X[i]
        Ysum +=Y[i]
        XYsum+=X[i]*Y[i]
        X2sum +=X[i]**2
        Y2sum +=Y[i]**2
    num = (N*XYsum )- (Xsum*Ysum)
    den = math.sqrt((N*X2sum)-(Xsum**2))*math.sqrt((N*Y2sum)-(Ysum**2))
    r= (num/den)
    r2 = r*r
    vif = 1/(1-r2)
    return vif


# In[33]:


print("For age and sex",vif(X1,X2))
print("For age and bmi",vif(X1,X3))
print("For age and children",vif(X1,X4))
print("For age and smoker",vif(X1,X5))
print()
print("For sex and bmi",vif(X2,X3))
print("For sex and children",vif(X2,X4))
print("For sex and smoker",vif(X2,X5))
print()
print("For bmi and children",vif(X3,X4))
print("For bmi and smoker",vif(X3,X5))
print()
print("For children and smoker",vif(X4,X5))
print()
print("VIF for all the columns is less than 5, therefore multicolinearity doesn't exist")


# In[ ]:





# In[ ]:




