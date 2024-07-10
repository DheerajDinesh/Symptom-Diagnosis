import numpy as np
import pandas as pd
import streamlit as st
import pickle

st.header('Disease Prediction')
st.write('Machine Learning Classification model to predict what disease a person acquires based on 132 symptoms of around 5000 patients. Result provides a prognosis from 41 different diseases.')
st.write('Select the symptoms that is observed in a patient and select the predict button.')
# Input choices creation
df=pd.read_csv('/content/drive/MyDrive/Disease_Prediction/Testing.csv')
cols= df.columns
cols=cols.drop(['prognosis'])


# Prediction value input
data={}
for i in cols:
  data[i]=st.checkbox(i)
train_data=list(data.values())
train_data=np.array(train_data)


# Final Prediction
if st.button('predict'):
    model = pickle.load(open('/content/disease.pkl','rb'))
    result = model.predict([train_data])    
    st.text(result)