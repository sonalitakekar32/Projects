
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load model and data
with open("model_v3.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]
scaler_full = data["scaler_full"]
scaler_similarity = data["scaler_similarity"]
similarity_features = data["similarity_features"]

# Sidebar for User Inputs
with st.sidebar:
    st.header("Customize Your Preferences")
    enrollment = st.slider('Select Enrollment Level', 0.0, 1.0, 0.5, step=0.01)
    rating = st.slider('Select Desired Rating', 0.0, 1.0, 0.5, step=0.01)

# Data Inspection
st.subheader("Dataset Overview")
st.write("### Dataset Preview:")
st.write(df.head())

st.write("### Dataset Info:")
st.text(f"Shape: {df.shape}")
st.write(f"Columns: {df.columns.tolist()}")
st.write(f"Unique Users: {df['user_id'].nunique()}")

# Popularity-Based Recommendation
def get_popular_courses(df, top_n=5):
    return df.nlargest(top_n, 'enrollment_numbers')[['course_name', 'enrollment_numbers']]

# Content-Based Filtering
def get_similar_courses_based_on_input(enrollment, rating, top_n=5):
    user_input = np.array([[enrollment, rating]])
    user_input_scaled = scaler_similarity.transform(user_input)
    similarities = cosine_similarity(user_input_scaled, df[similarity_features].values).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return df.iloc[top_indices][['course_id', 'course_name', 'enrollment_numbers', 'rating']]

# Personalized Recommendations
def display_personalized_recommendations(enrollment, rating):
    st.subheader("Top 5 Personalized Course Recommendations:")
    personalized_courses = get_similar_courses_based_on_input(enrollment, rating)
    if personalized_courses.empty:
        st.write("No recommendations found based on your preferences. Please try adjusting the sliders.")
    else:
        st.write(personalized_courses)

# Display Popular Courses
def display_popular_courses():
    st.subheader("Top 5 Popular Courses Based on Enrollment:")
    popular_courses = get_popular_courses(df)
    if popular_courses.empty:
        st.write("No popular courses found.")
    else:
        st.write(popular_courses)

# Additional Recommendations
def display_additional_recommendations(enrollment, rating):
    if st.button('Show More Recommendations'):
        st.subheader("Additional Course Recommendations (Content-Based):")
        additional_courses = get_similar_courses_based_on_input(enrollment, rating, top_n=10)
        st.write(additional_courses)
# Run Example
def run_example():
    display_personalized_recommendations(enrollment, rating)
    display_popular_courses()
    display_additional_recommendations(enrollment, rating)

# Execute the app
run_example()
