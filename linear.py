import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import streamlit as st

df = pd.read_csv("carprice.csv")

#Training a model
reg = linear_model.LinearRegression()
reg.fit(df[['Age_of_Car_Years']],df['Resale_Price_INR'])

st.title("Car Price Prediction (Linear Regression)")

#Slider
age = st.slider("Select age of the car(years):",1,5,12)

#Prediction and Display
pred_price = reg.predict(pd.DataFrame({'Age_of_Car_Years': [age]}))[0]
st.write(f"### Estimated Resale Price: ₹ {pred_price:,.0f}")

#Plotting the graph
fig, ax = plt.subplots()
ax.set_title("Car Price Prediction")
ax.set_xlabel("Years")
ax.set_ylabel("Price")
m,b = np.polyfit(df['Age_of_Car_Years'],df['Resale_Price_INR'],1)
ax.plot(
    df['Age_of_Car_Years'],
    m*df['Age_of_Car_Years']+b,
    color = 'Red',
    label = 'Regression Line'
)
ax.scatter(age,pred_price,s=50,color='blue',label='Prediction')
ax.legend()
st.pyplot(fig)