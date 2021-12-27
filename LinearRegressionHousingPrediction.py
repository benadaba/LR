#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
import streamlit as st
from collections import defaultdict

# ### Check out the Data
# https://www.youtube.com/watch?v=JwSS70SZdyM&t=7099s
# In[256]:

st.write("# Housing Prediction App")

USAhousing = pd.read_csv('USA_Housing.csv')
st.subheader("Preview the data")
st.write(USAhousing.head())



# In[257]:


USAhousing.head()


# In[258]:


USAhousing.info()


# In[259]:


USAhousing.describe()

st.sidebar.header("User Housing info")



# In[260]:


#USAhousing.columns


# # EDA
# 
# Let's create some simple plots to check out the data!

# In[261]:

st.header("Get a pairplot of the variables")
pairplot = sns.pairplot(USAhousing)
st.pyplot(pairplot)


# In[262]:

st.header("Get a distribution plot of the variables")
#streamlit displays matplotlib figure object so create a figure and axis and create your plot on there
distplot_fig, ax = plt.subplots() 
ax = sns.distplot(USAhousing['Price'])
st.pyplot(distplot_fig)


# In[263]:

st.header("Get the heatmap of the variables")
htmap_fig, ax = plt.subplots() 
htmap= sns.heatmap(USAhousing.corr())
st.pyplot(htmap_fig)

# ## Training a Linear Regression Model
# 
# Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.
# 
# ### X and y arrays

# In[264]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

def get_user_housing_data():
    income = st.sidebar.number_input('Income', min_value =1000, max_value =100000)
    age = st.sidebar.number_input("Age", min_value=18, max_value=67) 
    rooms = st.sidebar.number_input("No. of Rooms", min_value=1, max_value=6) 
    bedrooms = st.sidebar.number_input("No. of BedromsRooms", min_value=1, max_value=6) 
    population = st.sidebar.number_input("Population", min_value=1000000, max_value=5000000)
    values = defaultdict(int)
    values['income']=income
    values['age'] =age
    values["rooms"] = rooms
    values["bedrooms"] = bedrooms
    values["population"] = population
    return pd.DataFrame(data=values, index=[0])


df = get_user_housing_data()
st.subheader("User submitted Housing Data")
st.write(df)

# ## Train Test Split
# 
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[265]:


from sklearn.model_selection import train_test_split


# In[266]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Creating and Training the Model

# In[267]:


from sklearn.linear_model import LinearRegression


# In[268]:


lm = LinearRegression()


# In[269]:


lm.fit(X_train,y_train)


# ## Model Evaluation
# 
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[270]:


# print the intercept
st.subheader("Intercept")
st.write(lm.intercept_)


# In[277]:

st.subheader("Co-efficients")
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
st.write(coeff_df)


# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164883.28 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$122368.67 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$2233.80 **.
# - Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.15 **.
# 
# Does this make sense? Probably not because I made up this data. If you want real data to repeat this sort of analysis, check out the [boston dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html):
# 
# 

#     from sklearn.datasets import load_boston
#     boston = load_boston()
#     print(boston.DESCR)
#     boston_df = boston.data

# ## Predictions from our Model
# 
# Let's grab predictions off our test set and see how well it did!

# In[279]:


# predictions = lm.predict(X_test.head(1))
predictions = lm.predict(df)
st.subheader("X-test")
X_test


# In[282]:

st.subheader("Predictions")
st.write(predictions)
st.write(type(predictions))
if len(predictions==1):
    pass
else:
    st.subheader("Predictions Plot")
    prediction_fig , ax = plt.subplots()
    ax = plt.scatter(y_test,predictions)
    st.pyplot(prediction_fig)


# **Residual Histogram**

# In[281]:

st.subheader("Residuals Histogram")
residual_fig, ax = plt.subplots()
ax = sns.distplot((y_test-predictions),bins=50);
st.pyplot(residual_fig)

# ## Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.

# In[275]:


from sklearn import metrics


# In[276]:

if len(predictions==1):
    pass
else:
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#saving the model
# Saving the model
import pickle
pickle.dump(lm, open('housing_clf.pkl', 'wb'))

# This was your first real Machine Learning Project! Congrats on helping your neighbor out! We'll let this end here for now, but go ahead and explore the Boston Dataset mentioned earlier if this particular data set was interesting to you! 
# 
# Up next is your own Machine Learning Project!
# 
# ## Great Job!

