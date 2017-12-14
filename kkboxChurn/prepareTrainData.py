
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm

# In[3]:


trainDf = pd.read_csv('data/train_v2.csv')
uniqueUsers = trainDf.msno
uniqueUsers


# In[4]:


userDf = pd.read_csv('data/user_grouped.csv')
# userDf.head()


# In[5]:
activeUsers = trainDf.loc[trainDf.msno.isin(userDf.msno),:].sort_values('msno')
# 'active_users_label_test.npy'
activeUsers.to_csv('active_user_label_train.csv')

memberDf = pd.read_csv('data/members_v3.csv').loc[:,['msno','bd']]
memberDf.head()


# In[6]:


averageAge = 29.490912779050635
memberDf.loc[(memberDf['bd'] > 100) | (memberDf['bd'] < 5),'bd'] = averageAge
memberDf.head()


# In[7]:


userDf = userDf.loc[userDf['msno'].isin(uniqueUsers),:]


# In[8]:


userDf = userDf.join(memberDf.set_index('msno'), on='msno')


# In[9]:


userDf.head()


# In[10]:


#
userDf['bd'] = userDf['bd'].fillna(averageAge)
print userDf.shape
print userDf.dropna(subset=['bd']).shape


# In[11]:


print userDf.shape
print userDf.dropna(subset=['total_secs']).shape


# In[17]:

## store all users' info into an array 
grouped = userDf.groupby('msno')
columns = ['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs','bd']
output_list=[]
for name, group in tqdm(grouped):
    output_list.append( np.array(group.loc[:,columns]))

np.save('active_data_train.npy',np.array(output_list))
    



