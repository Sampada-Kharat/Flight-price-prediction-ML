# Flight-price-prediction-ml
Flight ticket price prediction using machine learning with feature engineering, model training, and Streamlit deployment.

Flight Price Prediction using Machine Learning

Overview
This project predicts airline ticket prices based on flight details such as airline, source city, destination, duration, stops, and booking time.
A Machine Learning regression model is trained on historical flight data and deployed using a Streamlit web application.

The goal of this project is to demonstrate:
•	End-to-end ML workflow

•	Feature engineering & model training

•	Model deployment using Streamlit

Features
•	Predict flight ticket price instantly

•	Interactive web interface

•	Multiple airline and city inputs

•	Real-time ML predictions

•	Deployed ML model integration

Machine Learning Workflow
1.	Data Cleaning & Preprocessing
2.	Exploratory Data Analysis (EDA)
3.	Feature Engineering
4.	Model Training & Evaluation
5.	Hyperparameter Tuning
6.	Model Serialization
7.	Web App Deployment

 Models Used
•	Linear Regression

•	Random Forest Regressor

•	Gradient Boosting Regressor

•	Linear SVR

Best-performing model saved using joblib.

Tech Stack
•	Python

•	Pandas, NumPy

•	Scikit-learn

•	Matplotlib & Seaborn

•	Streamlit

•	Joblib

 How to Run Locally
1. Clone Repository
2. Install Dependencies
pip install -r requirements.txt
3. Run Streamlit App
streamlit run app.py

Input Features
•	Airline

•	Source City

•	Destination City

•	Departure Time

•	Arrival Time

•	Number of Stops

•	Days Left Before Departure

•	Duration

•	Flight Class

Output
Predicted flight ticket price in INR (₹).

 Future Improvements
•	Deploy on cloud (Render / AWS / HuggingFace Spaces)

•	Add real-time flight API data

•	Improve feature engineering

•	Model explainability using SHAP
________________________________________
 Author
Sampada Kharat
linkedin.com/in/sampada-kharat-6a76b03ab
Machine Learning Enthusiast

