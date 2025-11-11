import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('ada_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

ada_model = load_model()
scaler = load_scaler()

# 2. Set up the Streamlit application title and description
st.title('Kidney Stone Disease Prediction App')
st.write('Enter the patient details below to predict the likelihood of Kidney Stone Disease.')
st.write('---')

# 3. Create input widgets for user data
st.header('Patient Information')

age = st.number_input('Age', min_value=1, max_value=120, value=40)
ua = st.number_input('Uric Acid (UA)', min_value=0.0, max_value=20.0, value=5.0, format="%.2f")
creat = st.number_input('Creatinine (CREAT)', min_value=0.1, max_value=10.0, value=1.0, format="%.2f")
phosph = st.number_input('Phosphate (PHOSPH)', min_value=0.1, max_value=10.0, value=3.0, format="%.2f")
calcium = st.number_input('Calcium (CALCIUM)', min_value=0.1, max_value=15.0, value=9.0, format="%.2f")
albumin = st.number_input('Albumin (ALBUMIN)', min_value=1.0, max_value=6.0, value=4.0, format="%.2f")
gender_options = {'Male': 1.0, 'Female': 2.0}
gender_selection = st.selectbox('Gender', options=list(gender_options.keys()))

# 4. Implement a function to preprocess the user input
def preprocess_input(age, ua, creat, phosph, calcium, albumin, gender_selection):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'AGE': age,
        'UA': ua,
        'CREAT': creat,
        'PHOSPH': phosph,
        'CALCIUM': calcium,
        'ALBUMIN': albumin,
        'GENDER': gender_options[gender_selection]
    }])

    # Convert GENDER to the numerical value before one-hot encoding
    input_data['GENDER'] = input_data['GENDER'].astype(float)

    # Scale numerical features
    numerical_cols = ['AGE', 'UA', 'CREAT', 'PHOSPH', 'CALCIUM', 'ALBUMIN']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # One-hot encode 'GENDER'
    # The original training data had GENDER_2.0, meaning Female (2.0) was encoded as 1
    # and Male (1.0) was encoded as 0 because of drop_first=True.
    gender_encoded = 1 if input_data['GENDER'].iloc[0] == 2.0 else 0
    input_data['GENDER_2.0'] = gender_encoded
    input_data = input_data.drop(columns=['GENDER'])

    # Ensure column order matches X_train (based on prior analysis)
    # The order of columns in X_train was AGE, UA, CREAT, PHOSPH, CALCIUM, ALBUMIN, GENDER_2.0
    ordered_cols = ['AGE', 'UA', 'CREAT', 'PHOSPH', 'CALCIUM', 'ALBUMIN', 'GENDER_2.0']
    input_data = input_data[ordered_cols]
    
    return input_data

# 5. Add a button to trigger prediction
if st.button('Predict Kidney Stone Disease'):
    processed_input = preprocess_input(age, ua, creat, phosph, calcium, albumin, gender_selection)
    prediction = ada_model.predict(processed_input)
    prediction_proba = ada_model.predict_proba(processed_input)[0]

    st.write('---')
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f"
### Kidney Stone Disease: Yes (Probability: {prediction_proba[1]:.2f})
")
    else:
        st.success(f"
### Kidney Stone Disease: No (Probability: {prediction_proba[0]:.2f})
")

    # Store the result for daily entries
    if 'daily_entries' not in st.session_state:
        st.session_state.daily_entries = []
    
    current_entry = {
        'AGE': age,
        'UA': ua,
        'CREAT': creat,
        'PHOSPH': phosph,
        'CALCIUM': calcium,
        'ALBUMIN': albumin,
        'GENDER': gender_selection,
        'Prediction': 'Yes' if prediction[0] == 1 else 'No',
        'Probability_Yes': prediction_proba[1],
        'Probability_No': prediction_proba[0]
    }
    st.session_state.daily_entries.append(current_entry)


st.write('---')
st.subheader('Daily Entries')
# Display all recorded daily entries
if 'daily_entries' in st.session_state and st.session_state.daily_entries:
    daily_entries_df = pd.DataFrame(st.session_state.daily_entries)
    st.dataframe(daily_entries_df)

    # Provide a 'Download Entries as CSV' button
    csv = daily_entries_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Daily Entries as CSV",
        data=csv,
        file_name='kidney_stone_daily_entries.csv',
        mime='text/csv',
    )
else:
    st.info('No entries recorded yet.')
