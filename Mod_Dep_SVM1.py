#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[36]:


model=pickle.load(open('C:/Users/Lenovo/svm.pkl','rb'))
model


# In[37]:


st.title('Support Vector Classification Model Deployment')


# In[38]:


def user_input_parameters():
    age=st.sidebar.number_input('age')
    systolic_bp=st.sidebar.number_input('systolic_bp')
    diastolic_bp=st.sidebar.number_input('diastolic_bp')
    cholesterol=st.sidebar.number_input('cholesterol')
    data={'age':age, 'systolic_bp':systolic_bp, 'diastolic_bp':diastolic_bp, 'cholesterol':cholesterol}    
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_parameters()
if st.button("Predict Diabetic Retinopathy"):
     predicted=model.predict(df)
     st.subheader('Predict Diabetic Retinopathy')
     st.write(predicted)
import nbconvert

converter = nbconvert.ScriptExporter()
body, _ = converter.from_filename("Mod_Dep_SVM1.ipynb")

with open("Mod_Dep_SVM1.py", "w", encoding="utf-8") as f:
    f.write(body)


# In[39]:


df


# In[ ]:




