🏥 AI-Driven Medical Insurance Cost Prediction
(Machine Learning – Regression Analysis)

This project uses Artificial Intelligence and Machine Learning techniques to predict medical insurance costs based on individual health and demographic data.
The objective is to show how AI-driven predictive models can help insurance companies estimate premiums accurately and support fair, data-driven decisions.

📌 Project Overview

Medical insurance cost estimation is a complex task influenced by multiple factors such as age, health conditions, lifestyle habits, and region.
This project analyzes historical medical insurance data and builds a regression-based AI model to predict insurance charges for individuals using key personal and health attributes.

🎯 Objectives

Analyze historical medical insurance cost data

Identify factors that significantly affect insurance charges

Build an accurate and interpretable AI-based prediction model

Evaluate model performance using standard regression metrics

📊 Dataset Information

Source: Kaggle – Medical Insurance Cost Dataset

Total Records: 1,338

Features: 7 input attributes

Target Variable: charges (medical insurance cost)

Key Features:

age – Age of the policyholder

sex – Gender

bmi – Body Mass Index

children – Number of dependents

smoker – Smoking status

region – Residential region

🧹 Data Preprocessing

Removed records with missing or inconsistent values

Encoded categorical variables using Label Encoding / One-Hot Encoding

Scaled numerical features using StandardScaler

Split data into 80% training and 20% testing sets

🤖 Model Used
Regression-Based Machine Learning Model

(Linear Regression / Multiple Linear Regression)

Chosen because:

Suitable for continuous value prediction

Easy to interpret and explain

Fast training and evaluation

Provides meaningful insights for insurance pricing

📏 Model Evaluation

R² Score: ≈ 0.85 – 0.90

Mean Absolute Error (MAE): Low prediction error

Root Mean Squared Error (RMSE): Indicates strong model accuracy

The model demonstrates reliable and consistent performance on unseen test data.

💡 Key Insights

Smoking status has the highest impact on insurance cost

Age and BMI significantly influence premium amounts

Number of children and region have moderate impact

AI-based regression models can effectively support premium prediction

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Jupyter Notebook / Google Colab

🚀 Applications

Personalized insurance premium estimation

Risk assessment for insurance companies

Fraud detection and cost optimization

Decision support for insurance providers

👥 Team Members
Name	Contribution
👥 Team Members
Name	Contribution
Shravani Joshi	Data Collection & Preprocessing, Model Development (Regression), Feature Encoding, Model Training & Evaluation
Ragini Gholap	Exploratory Data Analysis (EDA), Visualization, Model Testing, Documentation & Presentation

(Update names as per your group if needed)

▶️ How to Use the Project
Step 1: Clone the Repository
git clone https://github.com/your-username/ai-medical-insurance-cost-prediction.git
cd ai-medical-insurance-cost-prediction

Step 2: Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

Step 3: Run the Notebook

Open the .ipynb file in Jupyter Notebook or Google Colab

Run all cells to preprocess data and train the model

Step 4: Save the Model
model.pkl

Step 5: Load the Model for Prediction
import pickle

model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict(input_data)
print(prediction)

Step 6: (Optional) Deploy Using Streamlit
streamlit run app.py

Step 7: Use the Application

Enter user details (age, BMI, smoking status, region, etc.)

Get predicted Medical Insurance Cost

📌 Conclusion

This project demonstrates how AI-driven regression models, combined with effective preprocessing and feature analysis, can accurately predict medical insurance costs and assist insurers in making fair and informed decisions.

⭐ If you found this project useful, consider starring the repository!



