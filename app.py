import streamlit as st
import pickle
import pandas as pd
import os

# List of teams and cities
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Check if 'pipe.pkl' exists
if os.path.exists('pipe.pkl'):
    pipe = pickle.load(open('pipe.pkl', 'rb'))
else:
    st.error("Model file 'pipe.pkl' not found. Please ensure the file is in the correct directory.")
    st.stop()  # Stop execution if model file is missing

# Set the title of the app
st.title(" Shubham IPL Win Predictor")

# Add instructions for the user
st.sidebar.header("Instructions")
st.sidebar.write("""
    - Select the batting and bowling teams from the dropdown.
    - Choose the city where the match is being hosted.
    - Enter the target score and current match details (score, overs completed, wickets).
    - Press 'Predict Probability' to see the win probability for both teams.
""")

# Layout for inputs
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Select the host city
selected_city = st.selectbox("Select host city", sorted(cities))

# Input for the target score
target = st.number_input('Target', min_value=1, label_visibility="visible")

# Input for match details
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.1, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10)

# Validation and prediction logic
if st.button("Predict Probability"):
    # Check if overs are greater than 0
    if overs <= 0:
        st.error("Please enter a valid value for overs completed (greater than 0).")
    elif target <= score:
        st.error("Target must be greater than the current score.")
    else:
        # Calculate the required metrics
        runs_left = target - score
        balls_left = 120 - (overs * 6)  # 120 balls in total (20 overs)
        remaining_wickets = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        # Prepare input for the model
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [remaining_wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict win probabilities
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display the result
        st.header(f"{batting_team} - {round(win * 100)}%")
        st.header(f"{bowling_team} - {round(loss * 100)}%")
