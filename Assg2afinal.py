#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv (r'/home/sam/public_html/python/sem3/covidData1.csv')
# df


# In[3]:


print("Enter 1 for beds with O2 supply")
print("Enter 2 for beds without O2 supply")
print("Enter 3 for beds in ICU")

choice = int(input())


# In[4]:


if(choice == 1):
    title="Beds with O2 supply"
    plt.bar(df['Name'],df['BedswithO2'])
    plt.xlabel('Country')
    plt.ylabel('Number of Beds')
    plt.title(title)
    plt.xticks(rotation = 90)
    
elif(choice ==2):
    title="Beds without O2 supply"
    plt.bar(df['Name'],df['BedsWithoutO2'])
    plt.xlabel('Country')
    plt.ylabel('Number of Beds')
    plt.xticks(rotation = 90)
    
    plt.title(title)
    
elif(choice==3):
    title="Beds in ICU"
    plt.bar(df['Name'],df['BedsInICU'])
    plt.xlabel('Country')
    plt.xticks(rotation = 90)
    plt.ylabel('Number of Beds')
    plt.title(title)
    
else:
    print("Invalid Choice")

    


# In[5]:


if(choice == 1):
    limit = int(input("Enter the limit of beds "))
    print('inside 1')
    data=df.query(f'BedswithO2<{limit}')
    plt.bar(data['Name'],data['BedswithO2'])
    plt.xlabel('Country')
    plt.ylabel('Number of Beds')
    plt.xticks(rotation = 90)
    plt.title(title)
    
    print(data)
elif(choice ==2):
    limit = int(input("Enter the limit of beds "))
    data=df.query(f'BedsWithoutO2<{limit}')
    plt.bar(data['Name'],data['BedsWithoutO2'])
    plt.xlabel('Country')
    plt.ylabel('Number of Beds')
    plt.xticks(rotation = 90)
    plt.title(title)
elif(choice==3):
    limit = int(input("Enter the limit of beds "))
    data=df.query(f'BedsInICU<{limit}')
    plt.bar(data['Name'],data['BedsInICU'])
    plt.xlabel('Country')
    plt.ylabel('Number of Beds')
    plt.xticks(rotation = 90)
    plt.title(title)
else:
    print("Invalid Choice")
    


# In[ ]:





# In[ ]:




