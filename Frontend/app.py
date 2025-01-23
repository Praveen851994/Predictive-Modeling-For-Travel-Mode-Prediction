import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load models (assuming the models are already saved as .pkl)
rf_model = joblib.load('random_forest_model.pkl')
knn_model = joblib.load('knn_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
svc_model = joblib.load('svc_model.pkl')

# Function to display travel mode info
def display_info():
    st.header("Travel Modes Information")

    travel_modes = {
        "Bus": {
            "Description": "Buses are large vehicles designed to carry multiple passengers on designated routes.",
            "Advantages": "Affordable, accessible, and reduces traffic congestion.",
            "Disadvantages": "Limited routes and schedules, can be slow."
        },
        "E-bike": {
            "Description": "E-bikes are bicycles equipped with an electric motor to assist with pedaling.",
            "Advantages": "Eco-friendly, faster than walking, and requires less effort than a traditional bike.",
            "Disadvantages": "Battery life can limit distance, and higher initial cost."
        },
        "Bike": {
            "Description": "Bicycles are human-powered vehicles that can be used for transportation or recreation.",
            "Advantages": "Great for short distances, eco-friendly, and promotes health.",
            "Disadvantages": "Weather-dependent and can be unsafe in traffic."
        },
        "Walk": {
            "Description": "Walking is a simple and natural way of moving from one place to another on foot.",
            "Advantages": "Health benefits, zero emissions, and free.",
            "Disadvantages": "Limited to short distances and can be time-consuming."
        },
        "Car": {
            "Description": "Cars are motor vehicles with four wheels designed primarily for passenger transportation.",
            "Advantages": "Convenient for long distances and can carry more passengers.",
            "Disadvantages": "Traffic congestion, pollution, and cost of maintenance."
        },
        "Train": {
            "Description": "Trains are vehicles that run on tracks and are used for transporting passengers and goods over long distances.",
            "Advantages": "Fast for long distances, eco-friendly, and avoids traffic.",
            "Disadvantages": "Limited routes and can be expensive."
        }
    }
    
    for mode, details in travel_modes.items():
        st.subheader(mode)
        st.write("**Description:**", details["Description"])
        st.write("**Advantages:**", details["Advantages"])
        st.write("**Disadvantages:**", details["Disadvantages"])
        st.write("---")

# Function for prediction
def predict(x, y, z, model):
    input_data = pd.DataFrame([[x, y, z]], columns=['x', 'y', 'z'])

    if model == "Random Forest":
        prediction = rf_model.predict(input_data)[0]
    elif model == "KNN":
        prediction = knn_model.predict(input_data)[0]
    elif model == "Decision Tree":
        prediction = dt_model.predict(input_data)[0]
    elif model == "SVC":
        prediction = svc_model.predict(input_data)[0]
    else:
        prediction = "Model not found."
    
    return prediction

# Login Page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if username == "Admin" and password == "123":
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

# Main application
if 'logged_in' not in st.session_state:
    login_page()
else:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction"])
    
    if page == "Home":
        display_info()
    
    elif page == "Prediction":
        st.title("Travel Mode Prediction")
        
        # Create columns for input and model selection
        col1, col2 = st.columns(2)

        with col1:
            x = st.number_input("Enter x value:", format="%.2f")
            y = st.number_input("Enter y value:", format="%.2f")
        
        with col2:
            z = st.number_input("Enter z value:", format="%.2f")
            model_selection = st.selectbox("Select Model", ["Random Forest", "KNN", "Decision Tree", "SVC"])

        if st.button("Predict"):
            prediction = predict(x, y, z, model_selection)
            st.write(f"Prediction using {model_selection}: {prediction}")