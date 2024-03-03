#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv(r'C:\Users\Sakib\Downloads\COVID-19-Bangladesh.csv')
df.head(5)


# In[7]:


df.info()


# In[8]:


df['date'] = pd.to_datetime(df['date'])


# In[9]:


df.isnull().sum()


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


df.head(5)


# In[13]:


df['date'].describe()


# # Total-Confirmed
# -After 4/8/2020 the covide affected rate is much higher than before date.
# -after4/12/2020 the confirmed was increasing like in massive way 

# In[33]:


df['total_confirmed'].describe()


# In[32]:


plt.figure(figsize=(10,5))
sns.histplot(y='total_confirmed',x='date',data=df)


# In[38]:


plt.figure(figsize=(20,5))
sns.countplot(x='total_confirmed', data=df)


# In[41]:


count = len(df[df['total_confirmed'] > 500])
print(count)


# In[42]:


df.head(5)


# # Total_Recovered
# -From 03/08/2020-03/22/2022 the recovered rate was below 10 means a crucial time 
# -The recovered rate was in continuous row, also normalize 

# In[ ]:





# In[44]:


df['total_recovered'].describe()


# In[55]:


plt.figure(figsize=(5,10))
sns.histplot(x='total_recovered',y='date', data=df)



# In[58]:


plt.figure(figsize=(5,10))
sns.boxplot(y='total_recovered', data=df)


# In[59]:


df.head(5)


# # Total Deaths
# -after 2020/04/15 the death rate was increasing in booming rate

# In[60]:


df['total_deaths'].describe()


# In[66]:


plt.figure(figsize=(5,5))
sns.histplot(x='total_deaths',y='date', data=df)


# In[67]:


df['new_confirmed'].describe()


# In[69]:


plt.figure(figsize=(5,5))
sns.histplot(x='new_confirmed',y='date', data=df,bins=10)


# In[70]:


df['infectionRate(%)'].describe()


# In[71]:


df.head(5)


# # total_confirmed vs total_recovered
# Positive correlation between confirmed cases and recovery rates observed.
# Instances of relatively high recovery rates despite low confirmed cases, and vice versa.
# Outliers or unique scenarios identified, suggesting further investigation needed.

# In[79]:


cross_tab = pd.crosstab(df['total_confirmed'], df['total_recovered'],normalize='columns')
cross_tab


# In[75]:


sns.heatmap(cross_tab)


# # 'mortalityRate(%)' vs 'infectionRate(%)' vs 'recoveryRate(%)
# 
# Cases with very low mortality rates and infection rates tend to have very low or low recovery rates.
# 
# There is a trend of higher mortality rates being associated with higher infection rates and lower recovery rates.
# 
# As the infection rate increases from very low to very high, the recovery rate tends to decrease.
# 
# Most cases with a very low mortality rate have a very low or low recovery rate, regardless of the infection rate.

# In[83]:


df[['mortalityRate(%)','infectionRate(%)','recoveryRate(%)']].describe()


# In[84]:


df[['mortalityRate(%)', 'infectionRate(%)', 'recoveryRate(%)']].hist(figsize=(10, 6))


# In[ ]:





# In[89]:


Q1 = df['mortalityRate(%)'].quantile(0.25)
Q3 = df['mortalityRate(%)'].quantile(0.75)
IQR = Q3 - Q1
IQR


# In[90]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['mortalityRate(%)', 'infectionRate(%)', 'recoveryRate(%)']])
plt.title('Box Plot of Mortality Rate, Infection Rate, and Recovery Rate')
plt.xlabel('Variables')
plt.ylabel('Rate (%)')
plt.xticks(ticks=[0, 1, 2], labels=['Mortality Rate', 'Infection Rate', 'Recovery Rate'])
plt.tight_layout()
plt.show()


# In[92]:


mortality_bins = pd.cut(df['mortalityRate(%)'], bins=5, labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
infection_bins = pd.cut(df['infectionRate(%)'], bins=5, labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
recovery_bins = pd.cut(df['recoveryRate(%)'], bins=5, labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

# Now, let's create the crosstab
cross_tab = pd.crosstab(index=[mortality_bins, infection_bins], columns=recovery_bins, margins=True)

cross_tab


# In[93]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot
ax.scatter(df['mortalityRate(%)'], df['infectionRate(%)'], df['recoveryRate(%)'])

# Set labels and title
ax.set_xlabel('Mortality Rate')
ax.set_ylabel('Infection Rate')
ax.set_zlabel('Recovery Rate')
ax.set_title('3D Scatter Plot of Mortality Rate, Infection Rate, and Recovery Rate')

plt.show()


# In[94]:


sns.pairplot(df[['mortalityRate(%)', 'infectionRate(%)', 'recoveryRate(%)']])
plt.suptitle('Scatter Plot Matrix of Mortality Rate, Infection Rate, and Recovery Rate')
plt.show()


# In[95]:


df


# # released_from_quarantine' VS 'date'
# 
# It indicates whether there are consistent trends or fluctuations in the release of individuals over time.
# 
# The relationship between individuals released from quarantine and dates can shed light on the effectiveness of quarantine measures.
# 
# Patterns in release dates might suggest underlying strategies or policies related to quarantine management.

# In[96]:


df['released_from_quarantine']


# In[97]:


cross_tab = pd.crosstab(df['released_from_quarantine'], df['date'])

cross_tab


# In[98]:


plt.figure(figsize=(12, 8))
sns.heatmap(cross_tab, cmap="YlGnBu")
plt.title('Heatmap of Individuals Released from Quarantine by Date')
plt.xlabel('Date')
plt.ylabel('Released from Quarantine')
plt.show()


# In[99]:


df.head(5)


# In[100]:


df[['total_deaths','released_from_quarantine']].describe()


# In[101]:


cross_tab = pd.crosstab(df['total_deaths'], df['released_from_quarantine'])

cross_tab


# In[102]:


plt.figure(figsize=(10, 8))
sns.heatmap(cross_tab, cmap="YlGnBu", annot=True, fmt='d')
plt.title('Heatmap of Total Deaths vs Released from Quarantine')
plt.xlabel('Released from Quarantine')
plt.ylabel('Total Deaths')
plt.show()


# In[ ]:




