import streamlit as st
import pandas as pd
import joblib

# Load the saved model and preprocessing objects
try:
    dt_model = joblib.load('streamlit_app/decision_tree_model.joblib')
    le = joblib.load('streamlit_app/label_encoder.joblib')
    training_columns = joblib.load('streamlit_app/training_columns.joblib')
except FileNotFoundError:
    st.error("Model or preprocessing files not found. Please ensure 'decision_tree_model.joblib', 'label_encoder.joblib', and 'training_columns.joblib' are in the 'streamlit_app' directory.")
    st.stop()

# Set up the Streamlit application title and header
st.title("Student Placement Prediction")
st.header("Enter student details to predict placement")

# Create input fields for each feature
input_data = {}
for col in training_columns:
    if col == 'Internship_Experience_Yes':
        # Handle the original categorical input and convert to the encoded format
        internship_experience = st.selectbox('Internship Experience', ['No', 'Yes'])
        input_data[col] = 1 if internship_experience == 'Yes' else 0
    elif col == 'IQ':
        input_data[col] = st.number_input(col, min_value=0, max_value=200, value=100)
    elif col in ['Prev_Sem_Result', 'CGPA']:
        input_data[col] = st.number_input(col, min_value=0.0, max_value=10.0, value=7.0, step=0.01)
    elif col in ['Academic_Performance', 'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']:
         input_data[col] = st.number_input(col, min_value=0, max_value=10, value=5)
    else:
        # For any other columns, use a generic text input or number input
        input_data[col] = st.number_input(col)


# Create a button to trigger the prediction
if st.button('Predict Placement'):
    # Preprocess the user input
    input_df = pd.DataFrame([input_data])

    # Ensure the columns are in the same order as the training data
    input_df = input_df[training_columns]

    # Make prediction
    prediction_encoded = dt_model.predict(input_df)

    # Inverse transform the prediction (Decision Tree model predicts 'Yes'/'No' directly)
    prediction = prediction_encoded[0]

    # Display the result
    if prediction == 'Yes':
        st.success(f"Prediction: The student is likely to be Placed.")
    else:
        st.warning(f"Prediction: The student is likely to be Not Placed.")
