#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn


# In[2]:


#importing 2017-2020 data
df = pd.read_csv("data/2017-2020.csv",low_memory=False, parse_dates=["Year"])


# In[3]:


df.info()


# In[4]:


df.isna().sum() #eturns the number of missing values in each column


# In[5]:


df.columns


# In[6]:


fig, ax = plt.subplots()
ax.scatter(df["Year"][:1000], df["OC"][:1000])


# In[7]:


df.Year[:1000]


# In[8]:


df.Year.dtype


# In[9]:


df.OC.plot.hist()


# In[10]:


df.Year.dtype


# In[11]:


df.Year[:1000]


# In[12]:


df.head()


# In[13]:


#transpose
df.head().T


# In[14]:


df.Year.head(20)


# In[15]:


df.sort_values(by=["Year"], inplace=True, ascending=True)
df.Year.head(20)


# In[16]:


df_tmp = df.copy()


# In[17]:


df_tmp["tYear"] = df_tmp.Year.dt.year


# In[18]:


df_tmp.head().T


# In[19]:


df_tmp.drop("Year", axis=1, inplace=True)  #Drop specified labels from rows or columns. Remove rows or columns by specifying label names and corresponding axis


# In[20]:


df_tmp.College_Name.value_counts()


# In[21]:


df_tmp.head()


# In[22]:


len(df_tmp)


# In[23]:


#EDA


# In[24]:


df_tmp.OC.sort_values()


# In[25]:


df_tmp.ST.sort_values()


# In[26]:


#machine learning model


# In[27]:


len(df_tmp)


# # Modelling

# In[68]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1,random_state=42)

model.fit(df_tmp.drop("OC", axis=1), df_tmp["OC"])


# In[32]:


df_tmp.info()


# In[33]:


df_tmp["College_Name"].dtype


# In[34]:


df.isna().sum()


# In[35]:


#model driven EDA


# # Convert string to categories

# In[36]:


df_tmp.head().T


# In[37]:


pd.api.types.is_string_dtype(df_tmp["College_Name"])


# In[38]:


for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[39]:


#category- Finite list of text values


# In[40]:


for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()    #Set the Categorical to be ordered.


# In[41]:


df_tmp.info()  #changed from obj to category


# In[42]:


df_tmp.College_Name.cat.categories


# In[43]:


df_tmp.College_Name.cat.codes


# df_tmp.isnull().sum()/len(df_tmp)

# # Save preprocessed data

# In[44]:


df_tmp.to_csv("data/train_tmp.csv",index=False)


# In[45]:


df_tmp = pd.read_csv("data/train_tmp.csv",low_memory=False)
df_tmp.head().T


# In[46]:


df_tmp.isna().sum()


# # Fill missing values
# Fill numerical missing values first

# In[47]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# In[48]:


df_tmp.BCM


# In[49]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[50]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            df_tmp[label] = content.fillna(content.median())


# In[51]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# df_tmp.MBCDNC_is_missing.value_counts()

# df_tmp.ST_is_missing.value_counts()

# In[52]:


df_tmp.isna().sum()


# In[53]:


for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[54]:


for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        df_tmp[label] = pd.Categorical(content).codes+1


# In[55]:


pd.Categorical(df_tmp["College_Name"]).codes+1


# In[56]:


df_tmp.info()


# In[57]:


df_tmp.to_csv("data/train_tmp.csv",index=False)


# In[58]:


df_tmp = pd.read_csv("data/train_tmp.csv",low_memory=False)
df_tmp.head().T


# In[59]:


df_tmp.head().T


# In[60]:


df_tmp.to_csv("data/train_tmp.csv",index=False)


# In[61]:


df_tmp = pd.read_csv("data/train_tmp.csv",low_memory=False)
df_tmp.head().T


# In[62]:


df_tmp.isna().sum()


# In[63]:


df_tmp.head()


# In[64]:


len(df_tmp)


# In[65]:


get_ipython().run_cell_magic('time', '', '# Instantiate model\nmodel = RandomForestRegressor(n_jobs=-1, random_state=42)\n\n# Fit the model\nmodel.fit(df_tmp.drop("OC", axis=1), df_tmp["OC"])')


# In[66]:


# Score the model
model.score(df_tmp.drop("OC", axis=1), df_tmp["OC"])


# # Splitting data into train/validation sets

# In[ ]:


df_tmp.tYear


# In[ ]:


df_tmp.tYear.value_counts()


# In[ ]:


df_val = df_tmp[df_tmp.tYear == 2017]
df_train = df_tmp[df_tmp.tYear != 2017]

len(df_val), len(df_train)


# In[ ]:


X_train, y_train = df_train.drop("OC", axis=1), df_train.OC
X_valid, y_valid = df_val.drop("OC", axis=1), df_val.OC

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[ ]:


y_train


# # Testing our model on a subset (to tune the hyperparameters)

# In[ ]:


get_ipython().run_cell_magic('time', '', ' model = RandomForestRegressor(n_jobs=-1, \n                              random_state=42)\n\n model.fit(X_train, y_train)')


# In[ ]:


len(X_train)


# In[ ]:


model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=100)


# In[30]:


get_ipython().run_cell_magic('time', '', '# Cutting down on the max number of samples each estimator can see improves training time\nmodel.fit(X_train, y_train)')


# (X_train.shape[0] * 100) / 1000000

# 10000 * 100

# In[29]:


show_scores(model)


# # Make predictions on test data

# In[312]:


df_test = pd.read_csv("data/2021 - Sheet1.csv",
                      low_memory=False,
                      parse_dates=["Year"])

df_test.head()


# In[313]:


test_preds = model.predict(df_test)


# # Preprocessing the data (getting the test dataset in the same format as our training dataset)

# In[315]:


def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    df["tYear"] = df.Year.dt.year
    #df_tmp["Year"] = df_tmp.Year.dt.year
    
    df.drop("Year", axis=1, inplace=True)
    
    
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                
                df[label+"_is_missing"] = pd.isnull(content)
                
                df[label] = content.fillna(content.median())
    
        
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
           
            df[label] = pd.Categorical(content).codes+1
    
    return df


# In[316]:


df_test = preprocess_data(df_test)
df_test.head()


# In[317]:


test_preds =model.predict(df_test)  #Python predict() function enables us to predict the labels of the data values on the basis of the trained model


# In[318]:


X_train.head()


# In[319]:


set(X_train.columns) - set(df_test.columns)


# df_test["set()"] = False
# df_test.head()

# In[320]:


test_preds = model.predict(df_test)


# test_preds = ideal_model.predict(df_test)

# df_test["set()"] = False
# df_test["Branch Code_is_missing"] = False
# df_test["Branch Name_is_missing"] = False
# df_test["College_Name_is_missing"] = False
# df_test.head()

# In[321]:


test_preds


# In[308]:


df_preds = pd.DataFrame()
df_preds["College_Name"] = df_test["College_Name"]
df_preds["OC"] = test_preds
df_preds


# In[322]:


df_preds.to_csv("data/predictions.csv", index=False)


# In[323]:


dff = pd.read_csv("data/predictions.csv",low_memory=False)


# In[325]:


fig, ax = plt.subplots()
ax.scatter(dff["College_Name"][:1000], dff["OC"][:1000])
plt.xlabel("College_Name")
plt.ylabel("Cutoff mark")
plt.title("Predicted Cutoff Marks of OC category 2021")


# In[ ]:





# In[ ]:





# In[ ]:




