#import library
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

st.write("# Second-hand Car Price Prediction")

# Import data
data = pd.read_csv("Car_info.csv")
#Clean data
data_filter = (
    data.iloc[:, :-35]
    .drop("text", axis=1)
    .drop("link", axis=1)
    .drop("publisher", axis=1)
)
year_split = data_filter["publish_date"].str.split("-", n=1, expand=True)
data_filter["year_publish"] = year_split[0]
data_filter.drop(columns=["publish_date"], inplace=True)
#Prepare data
data_filter["id"] = le.fit_transform(data_filter["model"])
data_dropna = data_filter.dropna()
data_dropna["age"] = data_dropna["year_publish"].astype(int) - data_dropna["year"]
carid_map = dict(zip(data_filter.model, data_filter.id))
unique_brand = pd.unique(data_filter["brand"])
#assign variable
X = np.array(data_dropna[["id", "year", "mile", "age"]])
y = np.array(data_dropna["price"])
#Fit data
reg = LinearRegression().fit(X, y)
#Get input for prediction
brand_input = st.selectbox("Brand", unique_brand)
unique_model = pd.unique(
    (data_filter["model"].where(data_filter["brand"] == brand_input).dropna())
)

model_input = st.selectbox("Model", unique_model)
mile_input = st.slider("Mile", 1000, 1000000, 1000)
year_input = st.slider("Year", 2000, 2030)
thisyear = 2022
car_age_input = 2022 - year_input

for i, j in carid_map.items():
    if i == model_input:
        id_input = j
#Predict
output = (reg.predict(np.array([[id_input, year_input, mile_input, car_age_input]]))).round()

#Output
st.write("#### Predicted Price(THB)")
st.write(output)

st.write("## Code")
##########
st.write("##### Import library")
code1 = """
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
"""
st.code(code1, language="python")
##########
st.write("#### Read CSV file")
code2 = """
data = pd.read_csv("Car_info.csv")
"""
st.code(code2, language="python")
##########
st.write("#### Clean data")
code3 = """
data_filter = (
    data.iloc[:, :-35]
    .drop("text", axis=1)
    .drop("link", axis=1)
    .drop("publisher", axis=1)
)
year_split = data_filter["publish_date"].str.split("-", n=1, expand=True)
data_filter["year_publish"] = year_split[0]
data_filter.drop(columns=["publish_date"], inplace=True)
"""
st.code(code3, language="python")
##########
st.write("#### Prepare Data")
code4 = """
data_filter["id"] = le.fit_transform(data_filter["model"])
data_dropna = data_filter.dropna()
data_dropna["age"] = data_dropna["year_publish"].astype(int) - data_dropna["year"]
carid_map = dict(zip(data_filter.model, data_filter.id))
unique_brand = pd.unique(data_filter["brand"])
"""
st.code(code4, language="python")
##########
st.write("#### Assign Variable")
code5 = """
X = np.array(data_dropna[["id", "year", "mile", "age"]])
y = np.array(data_dropna["price"])
"""
st.code(code5, language="python")
##########
st.write("#### Fit data")
code6 = """
reg = LinearRegression().fit(X, y)
"""
st.code(code6, language="python")
##########
st.write("#### Get input for prediction")
code7 = """
model_input = st.selectbox("Model", unique_model)
mile_input = st.slider("Mile", 1000, 1000000, 1000)
year_input = st.slider("Year", 2000, 2030)
thisyear = 2022
car_age_input = 2022 - year_input

for i, j in carid_map.items():
    if i == model_input:
        id_input = j
"""
st.code(code7, language="python")
##########
st.write("#### Predict")
code8 = """
output = reg.predict(np.array([[id_input, year_input, mile_input, car_age_input]]))
"""
st.code(code8, language="python")
##########
st.write("#### Print output")
code9 = """
st.write("#### Predicted Price(THB)")
st.write(output)
"""
st.code(code9, language="python")
##########