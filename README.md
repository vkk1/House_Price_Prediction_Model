🏠 House Price Prediction Model

This project implements a Machine Learning model to predict housing prices using structured data. It covers the full pipeline—data preprocessing, feature engineering, model training with XGBoost, hyperparameter tuning, and evaluation with multiple performance metrics.

📂 Project Overview

Dataset: Housing dataset (Housing.csv)

Goal: Predict house prices based on features such as area, number of bedrooms, bathrooms, and amenities.

Approach:

Data cleaning and preprocessing

Feature engineering (e.g., price per square foot, rooms per bathroom)

Label encoding for categorical variables

Model training with XGBoost

Hyperparameter tuning using RandomizedSearchCV

Evaluation with R², MAE, and MAPE

⚙️ Tech Stack

Python

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

📊 Results

Training Data: R², MAE, MAPE calculated

Testing Data: R², MAE, MAPE calculated

Scatter plots can be used to compare actual vs. predicted prices.

🚀 Getting Started

Clone the repository:

git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction


Install dependencies:

pip install -r requirements.txt


Run the script:

python House_Price_Prediction_Model.py
