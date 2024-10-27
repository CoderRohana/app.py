# Import necessary libraries
import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load and preprocess the data
@st.cache
def load_data():
    df = pd.read_csv('amazon.csv')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['rating'], inplace=True)
    return df

# Function to build and train the model
@st.cache
def train_model(data):
    reader = Reader(rating_scale=(1, 5))
    data_surprise = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)
    trainset = data_surprise.build_full_trainset()
    algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    algo.fit(trainset)
    return algo

# Main application function
def main():
    st.title("Personalized Recommendation System for Online Shopping")
    st.write("This app provides personalized product recommendations based on user preferences.")

    # Load data
    df = load_data()

    # Display sample data
    st.subheader("Sample Data")
    st.write(df.head())

    # Train the recommendation model
    algo = train_model(df)

    # User input for recommendations
    user_id = st.text_input("Enter User ID", "")
    product_id = st.text_input("Enter Product ID (for rating prediction)", "")

    # Make a prediction if inputs are provided
    if user_id and product_id:
        pred = algo.predict(user_id, product_id)
        st.write(f"Predicted rating for user {user_id} on product {product_id}: {pred.est:.2f}")

    # Display top popular products
    st.subheader("Top 10 Popular Products")
    popular_products = df['product_id'].value_counts().head(10)
    st.bar_chart(popular_products)

if __name__ == "__main__":
    main()
