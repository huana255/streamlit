import streamlit as st
import numpy as np
import pickle as pkl

st.header("flower predicter")
st.image("flower.jpg", width = 100)

with st.form('my_form'):
    st.write("Enter the info on your iris below:")
    sepal_length = st.number_input("Sepal Length (cm)")
    sepal_width = st.number_input("Sepal Width (cm)")
    petal_length = st.number_input("Petal Length (cm)")
    petal_width = st.number_input("Petal Width (cm)")
    st.form_submit_button()

with open ('forest.pkl', 'rb') as f:
    forest = pkl.load(f)

new_flower = np.array([sepal_length, sepal_width, petal_length, petal_width])

prediction = forest.predict(new_flower.reshape(1, -1))
st.write(prediction)