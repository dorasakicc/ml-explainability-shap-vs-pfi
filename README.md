# Explainability of Machine Learning Models  
## Comparing SHAP and Permutation Feature Importance

This project explores explainability techniques for machine learning models predicting household appliance energy consumption. The main goal is to compare two popular model-agnostic and model-specific explainability methods: Permutation Feature Importance (PFI) and SHAP.

## Project Overview

Two tree-based regression models, **Random Forest** and **XGBoost**, are trained to predict energy consumption based on environmental and temporal features. A time-aware train–test split is used to simulate a realistic forecasting scenario. Due to strong skewness in the target variable, a logarithmic transformation is applied to improve model learning.

After training, model behavior is analyzed using:
- **Permutation Feature Importance (PFI)** to measure how much each feature affects overall model performance
- **SHAP (SHapley Additive exPlanations)** to provide both global and local explanations of predictions

The project investigates whether both explainability methods lead to consistent conclusions about which features drive model predictions.

## Key Findings

- The **hour of the day** is the most influential variable, indicating strong daily usage patterns.
- **Indoor temperature and humidity features** also play a significant role in predicting energy consumption.
- SHAP provides detailed instance-level explanations, showing how individual features increase or decrease specific predictions.
- Both PFI and SHAP produce consistent importance rankings, increasing confidence in the reliability of the explanations.
- Random noise features have negligible importance, confirming that the models do not rely on irrelevant information.

## Technologies Used

- Python  
- Scikit-learn  
- XGBoost  
- SHAP  
- Pandas, NumPy, Matplotlib  

## Project Structure

The notebook includes:
1. Data preprocessing and feature engineering  
2. Model training and evaluation  
3. Permutation Feature Importance analysis  
4. SHAP global and local explanations  
5. Comparison between PFI and SHAP  
6. Error analysis and discussion  

## Notes

This project was developed for educational purposes as part of coursework focused on explainable artificial intelligence. The goal is not only to build predictive models, but also to understand and interpret how they make decisions.

## Future Work

Future improvements could include incorporating additional contextual data, applying temporal deep learning models, or exploring causal explainability methods to further understand energy consumption patterns.
