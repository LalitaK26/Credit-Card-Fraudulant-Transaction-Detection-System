import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
# load data
data = pd.read_csv(r'C:\Users\lalit\Downloads\Credit-Card-Fraudulant-Transaction-Detection-Model-main (1)\Credit-Card-Fraudulant-Transaction-Detection-Model-main\creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)
# app
st.title("credit card fraud detection model")
input_df = st.text_input("enter all required features values")
input_df_splited = input_df.split(",")
submit = st.button("submit")

if submit:
    # Remove double quotes from the values
    non_empty_values = [value.replace('"', '') for value in input_df_splited if value.strip()]

    # Convert non-empty strings to floating-point numbers
    features = np.asarray(non_empty_values, dtype=np.float64)

    prediction = model.predict(features.reshape(1,-1))

    if prediction[0] == 0:
        st.write("legitimate transaction")

    else:
        st.write("fraudulant transaction")






